"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

from prismatic.models.action_heads import AuxHead, L1RegressionActionHead, VisualTokenCosineHead
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.flow_gr00t_jepa_action_head import FlowMatchingActionJEPAHead
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedFANProjector, FusedMLPProjector, LinearProjector, MLPProjector
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, NUM_ACTIONS_CHUNK, NUM_TOKENS, PROPRIO_DIM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def _maybe_cuda_mem_snapshot(label: str):
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "label": label,
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


def _build_prefix_condition_memory(hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    supervised_mask = labels.ne(IGNORE_INDEX)
    if not torch.any(supervised_mask):
        return hidden_states

    batch_size, _, hidden_dim = hidden_states.shape
    first_supervised = supervised_mask.to(dtype=torch.int64).argmax(dim=1)
    has_supervision = supervised_mask.any(dim=1)
    first_supervised = torch.where(
        has_supervision,
        first_supervised,
        torch.full_like(first_supervised, hidden_states.shape[1]),
    )
    max_prefix_len = int(first_supervised.max().item())
    if max_prefix_len <= 0:
        return hidden_states.new_zeros((batch_size, 1, hidden_dim))

    prefix_memory = hidden_states.new_zeros((batch_size, max_prefix_len, hidden_dim))
    for idx in range(batch_size):
        prefix_len = int(first_supervised[idx].item())
        if prefix_len > 0:
            prefix_memory[idx, :prefix_len] = hidden_states[idx, :prefix_len]
    return prefix_memory


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fusedfan-projector"):
            self.projector = FusedFANProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.vision_backbone_requires_grad = False

        # === Action Head & Aux Head (JEPA-VLA) ===
        self.use_action_head = kwargs.get("use_action_head", True)
        self.action_head_type = kwargs.get("action_head_type", "flow_gr00t").lower()
        self.flow_gr00t_use_full_llm_hidden = kwargs.get("flow_gr00t_use_full_llm_hidden", False)
        self.use_llm_ce_loss = kwargs.get("use_llm_ce_loss", False)
        self.lambda_llm_ce = float(kwargs.get("lambda_llm_ce", 0.1))
        self.lora_unfreeze_last_n_llm_layers = int(kwargs.get("lora_unfreeze_last_n_llm_layers", 0))
        self.use_action_queries = bool(kwargs.get("use_action_queries", False))
        self.use_context = bool(kwargs.get("context", False))
        self.context_action_tokens = int(kwargs.get("context_action_tokens", 4))
        if self.use_context:
            d = llm_backbone.embed_dim; da = kwargs.get("d_action", ACTION_DIM); dp = kwargs.get("d_proprio", PROPRIO_DIM); h = kwargs.get("action_horizon", NUM_ACTIONS_CHUNK)
            self.context_state_encoder = torch.nn.Sequential(torch.nn.Linear(dp, d), torch.nn.SiLU(), torch.nn.Linear(d, d))
            self.context_action_encoder = torch.nn.Sequential(torch.nn.Linear(h * da, d), torch.nn.SiLU(), torch.nn.Linear(d, self.context_action_tokens * d))
            self.context_time_embedding = torch.nn.Embedding(257, d)
            self.context_action_slot_embedding = torch.nn.Embedding(self.context_action_tokens, d)
        if self.action_head_type == "l1":
            self.action_placeholder_tokens = NUM_TOKENS
        elif self.action_head_type in {"flow_gr00t", "flow_gr00t_jepa"}:
            self.action_placeholder_tokens = kwargs.get("flow_gr00t_placeholder_tokens", NUM_TOKENS)
        else:
            self.action_placeholder_tokens = 0
        if self.use_llm_ce_loss and self.action_head_type in {"flow_gr00t", "flow_gr00t_jepa"}:
            self.action_placeholder_tokens = 0
        self.use_aux_head = kwargs.get("use_aux_head", True)
        self.lambda_aux = kwargs.get("lambda_aux", 0.2)
        self.use_visual_token_cosine_head = kwargs.get("use_visual_token_cosine_head", False)
        self.lambda_visual_token_cosine = kwargs.get("lambda_visual_token_cosine", 0.5)
        self.visual_token_cosine_use_projector_target = kwargs.get("visual_token_cosine_use_projector_target", True)
        self.visual_token_cosine_layer_idx = int(kwargs.get("visual_token_cosine_layer_idx", -1))
        self.visual_token_cosine_projection_type = str(kwargs.get("visual_token_cosine_projection_type", "mlp")).lower()
        if self.visual_token_cosine_projection_type not in {"mlp", "conv"}:
            raise ValueError(
                "Unsupported visual_token_cosine_projection_type "
                f"`{self.visual_token_cosine_projection_type}`. Use `mlp` or `conv`."
            )
        if self.use_action_queries and self.action_placeholder_tokens > 0:
            self.action_queries = torch.nn.Embedding(self.action_placeholder_tokens, llm_backbone.embed_dim)
            self.action_queries.weight.data.zero_()
        else:
            self.action_queries = None

        if self.use_action_head:
            if self.action_head_type == "flow_gr00t":
                self.action_head = FlowMatchingActionHead(
                    d_proprio=kwargs.get("d_proprio", PROPRIO_DIM),
                    d_action=kwargs.get("d_action", ACTION_DIM),
                    d_llm=llm_backbone.embed_dim,
                    horizon=kwargs.get("action_horizon", NUM_ACTIONS_CHUNK),
                    fm_hidden_size=kwargs.get("fm_hidden_size", 1024),
                    fm_action_model_type=kwargs.get("fm_action_model_type", "DiT-B"),
                    fm_num_layers=kwargs.get("fm_num_layers", 16),
                    fm_num_inference_timesteps=kwargs.get("fm_num_inference_timesteps", 4),
                    fm_num_timestep_buckets=kwargs.get("fm_num_timestep_buckets", 1000),
                    fm_noise_beta_alpha=kwargs.get("fm_noise_beta_alpha", 1.5),
                    fm_noise_beta_beta=kwargs.get("fm_noise_beta_beta", 1.0),
                    fm_noise_s=kwargs.get("fm_noise_s", 0.999),
                    fm_num_target_vision_tokens=kwargs.get("fm_num_target_vision_tokens", 32),
                    fm_add_pos_embed=kwargs.get("fm_add_pos_embed", True),
                    fm_max_seq_len=kwargs.get("fm_max_seq_len", 1024),
                    fm_state_dropout=kwargs.get("fm_state_dropout", 0.5),
                )
            elif self.action_head_type == "flow_gr00t_jepa":
                self.action_head = FlowMatchingActionJEPAHead(
                    d_proprio=kwargs.get("d_proprio", PROPRIO_DIM),
                    d_action=kwargs.get("d_action", ACTION_DIM),
                    d_llm=llm_backbone.embed_dim,
                    d_jepa=kwargs.get("d_jepa", vision_backbone.embed_dim),
                    horizon=kwargs.get("action_horizon", NUM_ACTIONS_CHUNK),
                    jepa_horizon=kwargs.get("fm_jepa_horizon", kwargs.get("aux_T", 4)),
                    fm_hidden_size=kwargs.get("fm_hidden_size", 1024),
                    fm_action_model_type=kwargs.get("fm_action_model_type", "DiT-B"),
                    fm_num_layers=kwargs.get("fm_num_layers", 16),
                    fm_num_inference_timesteps=kwargs.get("fm_num_inference_timesteps", 4),
                    fm_num_timestep_buckets=kwargs.get("fm_num_timestep_buckets", 1000),
                    fm_noise_beta_alpha=kwargs.get("fm_noise_beta_alpha", 1.5),
                    fm_noise_beta_beta=kwargs.get("fm_noise_beta_beta", 1.0),
                    fm_noise_s=kwargs.get("fm_noise_s", 0.999),
                    fm_num_target_vision_tokens=kwargs.get("fm_num_target_vision_tokens", 32),
                    fm_add_pos_embed=kwargs.get("fm_add_pos_embed", True),
                    fm_max_seq_len=kwargs.get("fm_max_seq_len", 1024),
                    fm_state_dropout=kwargs.get("fm_state_dropout", 0.5),
                    fm_jepa_loss_weight=kwargs.get("fm_jepa_loss_weight", 1.0),
                )
            elif self.action_head_type == "l1":
                self.action_head = L1RegressionActionHead(
                    d_proprio=kwargs.get("d_proprio", PROPRIO_DIM),
                    d_action=kwargs.get("d_action", ACTION_DIM),
                    d_llm=llm_backbone.embed_dim,
                    hidden_dim=llm_backbone.embed_dim,
                    horizon=kwargs.get("action_horizon", NUM_ACTIONS_CHUNK),
                    num_blocks=kwargs.get("l1_num_blocks", 24),
                    use_pro_version=kwargs.get("l1_use_pro_version", True),
                )
            else:
                raise ValueError(f"Unsupported action_head_type: {self.action_head_type}")
        else:
            self.action_head = None

        if self.use_aux_head:
            self.aux_head = AuxHead(
                d_llm=llm_backbone.embed_dim,
                d_jepa=kwargs.get("d_jepa", 1024),
                num_views_max=kwargs.get("num_views_max", 3),
                d_aux=kwargs.get("d_aux", 768),
                n_heads=kwargs.get("n_heads_aux", 12),
                num_layers=kwargs.get("num_layers_aux", 12),
                ffn_ratio=kwargs.get("ffn_ratio_aux", 4),
                T=kwargs.get("aux_T", 4),
                H=kwargs.get("aux_H", 14),
                W=kwargs.get("aux_W", 14),
            )
        else:
            self.aux_head = None

        if self.use_visual_token_cosine_head:
            visual_token_target_dim = (
                llm_backbone.embed_dim
                if self.visual_token_cosine_use_projector_target
                else kwargs.get("d_jepa", vision_backbone.embed_dim)
            )
            self.visual_token_cosine_head = VisualTokenCosineHead(
                d_llm=llm_backbone.embed_dim,
                d_target=visual_token_target_dim,
                projection_type=self.visual_token_cosine_projection_type,
            )
        else:
            self.visual_token_cosine_head = None

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        if self.action_queries is not None:
            self.all_module_keys.append("action_queries")
        if self.use_context:
            self.all_module_keys.extend(["context_state_encoder", "context_action_encoder", "context_time_embedding", "context_action_slot_embedding"])
        if self.action_head is not None:
            self.all_module_keys.append("action_head")
        if self.aux_head is not None:
            self.all_module_keys.append("aux_head")
        if self.visual_token_cosine_head is not None:
            self.all_module_keys.append("visual_token_cosine_head")
        self.trainable_module_keys = []

        if self.action_head is not None:
            overwatch.info(
                "Initialized Action Head `%s` (placeholder tokens = %d, full_llm_hidden = %s)",
                self.action_head_type,
                self.action_placeholder_tokens,
                self.flow_gr00t_use_full_llm_hidden,
            )

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def _get_last_n_llm_trainable_modules(self, num_layers: int) -> List[nn.Module]:
        if num_layers <= 0:
            return []

        llm = self.llm_backbone.llm
        candidate_models = [
            getattr(llm, "model", None),
            getattr(getattr(llm, "model", None), "model", None),
            getattr(llm, "base_model", None),
            getattr(getattr(llm, "base_model", None), "model", None),
            getattr(getattr(getattr(llm, "base_model", None), "model", None), "model", None),
        ]

        for model in candidate_models:
            if model is None:
                continue

            layers = getattr(model, "layers", None)
            if layers is None:
                continue

            layers = list(layers)
            if not layers:
                continue

            modules: List[nn.Module] = list(layers[-num_layers:])
            final_norm = getattr(model, "norm", None)
            if final_norm is not None:
                modules.append(final_norm)
            return modules

        fallback_modules = getattr(self.llm_backbone, "last_layer_finetune_modules", None)
        if fallback_modules is not None:
            return list(fallback_modules)

        raise AttributeError(
            "Could not locate decoder layers for LLM backbone when enabling LoRA + last-N-layer training. "
            f"LLM wrapper type={type(self.llm_backbone).__name__}, model type={type(llm).__name__}"
        )

    def _unfreeze_last_n_llm_layers(self, num_layers: int) -> int:
        modules = self._get_last_n_llm_trainable_modules(num_layers)
        for module in modules:
            module.requires_grad_(True)
        return len(modules)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        load_visual_token_cosine_head: bool = True,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        if vlm.action_queries is not None and "action_queries" in model_state_dict:
            vlm.action_queries.load_state_dict(model_state_dict["action_queries"])
        if vlm.action_head is not None and "action_head" in model_state_dict:
            vlm.action_head.load_state_dict(model_state_dict["action_head"])
        if vlm.aux_head is not None and "aux_head" in model_state_dict:
            vlm.aux_head.load_state_dict(model_state_dict["aux_head"])
        if vlm.visual_token_cosine_head is not None and "visual_token_cosine_head" in model_state_dict:
            if load_visual_token_cosine_head:
                missing, unexpected = vlm.visual_token_cosine_head.load_state_dict(
                    model_state_dict["visual_token_cosine_head"],
                    strict=False,
                )
                if missing or unexpected:
                    overwatch.info(
                        "Visual Token Cosine Head checkpoint mismatch ignored "
                        "(missing=%s unexpected=%s)",
                        missing,
                        unexpected,
                    )
            else:
                overwatch.info("Skipping visual_token_cosine_head checkpoint load by request.")

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def _inject_action_query_embeddings(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_queries is None or self.use_llm_ce_loss:
            return input_embeddings

        placeholder_mask = input_ids.eq(ACTION_TOKEN_BEGIN_IDX)
        if not torch.any(placeholder_mask):
            return input_embeddings

        placeholder_count = int(placeholder_mask.sum(dim=1).min().item())
        if placeholder_count <= 0:
            return input_embeddings
        if not torch.all(placeholder_mask.sum(dim=1).eq(placeholder_count)):
            raise ValueError("Action placeholder count must match across the batch when `use_action_queries=True`.")
        if placeholder_count > self.action_queries.num_embeddings:
            raise ValueError(
                f"Action placeholder count {placeholder_count} exceeds action query table size "
                f"{self.action_queries.num_embeddings}."
            )

        query_embeddings = self.action_queries.weight[:placeholder_count]
        query_embeddings = query_embeddings.unsqueeze(0).expand(input_embeddings.shape[0], -1, -1)
        query_embeddings = query_embeddings.to(dtype=input_embeddings.dtype, device=input_embeddings.device)

        output_embeddings = input_embeddings.clone()
        output_embeddings[placeholder_mask] = query_embeddings.reshape(-1, query_embeddings.shape[-1])
        return output_embeddings

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        def log_action_head_trainable(prefix: str = "[TRAINABLE] 🔥 =>>") -> None:
            if self.action_head is not None:
                overwatch.info(
                    f"{prefix} Action Head `{self.action_head_type}` "
                    f"(placeholders={self.action_placeholder_tokens})",
                    ctx_level=1,
                )

        def log_action_queries_trainable(prefix: str = "[TRAINABLE] 🔥 =>>") -> None:
            if self.action_queries is not None:
                overwatch.info(
                    f"{prefix} Action Queries (`{self.action_queries.num_embeddings}` x `{self.action_queries.embedding_dim}`)",
                    ctx_level=1,
                )

        if self.action_queries is not None:
            self.action_queries.requires_grad_(False)

        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"finetune", "vla-train", "vla-llm-projector-train"}:
            train_projector = stage == "vla-llm-projector-train"
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(train_projector)
            if self.action_queries is not None:
                self.action_queries.requires_grad_(True)
            if self.action_head is not None:
                self.action_head.requires_grad_(True)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(True)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]
            if train_projector:
                self.trainable_module_keys.append("projector")
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")
            if self.action_head is not None:
                self.trainable_module_keys.append("action_head")
            if self.aux_head is not None:
                self.trainable_module_keys.append("aux_head")
            if self.visual_token_cosine_head is not None:
                self.trainable_module_keys.append("visual_token_cosine_head")

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            projector_status = "[TRAINABLE] 🔥 =>>" if train_projector else "[Frozen]    🥶 =>>"
            overwatch.info(f"{projector_status} Projector `{self.arch_specifier}`", ctx_level=1)
            if self.action_head is not None:
                log_action_head_trainable()
            if self.action_queries is not None:
                log_action_queries_trainable()
            if self.aux_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Aux Head", ctx_level=1)
            if self.visual_token_cosine_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Visual Token Cosine Head", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(False)
            if self.action_queries is not None:
                self.action_queries.requires_grad_(True)
            if self.action_head is not None:
                self.action_head.requires_grad_(True)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(True)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "llm_backbone"]
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")
            if self.action_head is not None:
                self.trainable_module_keys.append("action_head")
            if self.aux_head is not None:
                self.trainable_module_keys.append("aux_head")
            if self.visual_token_cosine_head is not None:
                self.trainable_module_keys.append("visual_token_cosine_head")

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            if self.action_queries is not None:
                log_action_queries_trainable()
            log_action_head_trainable()
            if self.aux_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Aux Head", ctx_level=1)
            if self.visual_token_cosine_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Visual Token Cosine Head", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            if self.action_queries is not None:
                self.action_queries.requires_grad_(True)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            if self.action_queries is not None:
                log_action_queries_trainable("[TRAINABLE]                 🔥   =>>")
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            if self.action_queries is not None:
                self.action_queries.requires_grad_(True)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)
            if self.action_head is not None:
                self.action_head.requires_grad_(True)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(True)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "llm_backbone"]
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")
            if self.action_head is not None:
                self.trainable_module_keys.append("action_head")
            if self.aux_head is not None:
                self.trainable_module_keys.append("aux_head")
            if self.visual_token_cosine_head is not None:
                self.trainable_module_keys.append("visual_token_cosine_head")

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            if self.action_queries is not None:
                log_action_queries_trainable("[TRAINABLE]                 🔥   =>>")
            log_action_head_trainable("[TRAINABLE]                 🔥   =>>")
            if self.aux_head is not None:
                overwatch.info(f"[TRAINABLE]                 🔥   =>> Aux Head", ctx_level=1)
            if self.visual_token_cosine_head is not None:
                overwatch.info(f"[TRAINABLE]                 🔥   =>> Visual Token Cosine Head", ctx_level=1)
            # fmt: on

        elif stage in {"vla-lora-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)

            lora_param_names = []
            for name, param in self.llm_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    lora_param_names.append(name)

            if self.action_queries is not None:
                self.action_queries.requires_grad_(True)
            if self.action_head is not None:
                self.action_head.requires_grad_(True)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(True)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(True)

            self.trainable_module_keys = ["llm_backbone"]
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")
            if self.action_head is not None:
                self.trainable_module_keys.append("action_head")
            if self.aux_head is not None:
                self.trainable_module_keys.append("aux_head")
            if self.visual_token_cosine_head is not None:
                self.trainable_module_keys.append("visual_token_cosine_head")

            self.vision_backbone_requires_grad = False

            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> LLM LoRA Adapters (`{len(lora_param_names)}` parameter groups matched)",
                ctx_level=1,
            )
            if self.action_queries is not None:
                log_action_queries_trainable()
            if self.action_head is not None:
                log_action_head_trainable()
            if self.aux_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Aux Head", ctx_level=1)
            if self.visual_token_cosine_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Visual Token Cosine Head", ctx_level=1)

        elif stage in {"vla-lora-last-n-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)

            lora_param_names = []
            for name, param in self.llm_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    lora_param_names.append(name)

            unfrozen_module_count = self._unfreeze_last_n_llm_layers(self.lora_unfreeze_last_n_llm_layers)

            if self.action_queries is not None:
                self.action_queries.requires_grad_(True)
            if self.action_head is not None:
                self.action_head.requires_grad_(True)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(True)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(True)

            self.trainable_module_keys = ["llm_backbone"]
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")
            if self.action_head is not None:
                self.trainable_module_keys.append("action_head")
            if self.aux_head is not None:
                self.trainable_module_keys.append("aux_head")
            if self.visual_token_cosine_head is not None:
                self.trainable_module_keys.append("visual_token_cosine_head")

            self.vision_backbone_requires_grad = False

            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> LLM LoRA Adapters (`{len(lora_param_names)}` parameter groups matched)",
                ctx_level=1,
            )
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> Last {self.lora_unfreeze_last_n_llm_layers} LLM Layers "
                f"(`{unfrozen_module_count}` module blocks enabled)",
                ctx_level=1,
            )
            if self.action_queries is not None:
                log_action_queries_trainable()
            if self.action_head is not None:
                log_action_head_trainable()
            if self.aux_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Aux Head", ctx_level=1)
            if self.visual_token_cosine_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Visual Token Cosine Head", ctx_level=1)

        elif stage in {"vla-vlm-peft-train", "vla-vlm-peft-frozen-vision-train"}:
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            if self.action_queries is not None:
                self.action_queries.requires_grad_(False)
            if self.action_head is not None:
                self.action_head.requires_grad_(False)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(False)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(False)

            lora_param_names = []
            modules_to_save_names = []
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    lora_param_names.append(name)
                elif "modules_to_save" in name:
                    param.requires_grad_(True)
                    modules_to_save_names.append(name)

            if self.action_head is not None:
                self.action_head.requires_grad_(True)
            if self.aux_head is not None:
                self.aux_head.requires_grad_(True)
            if self.visual_token_cosine_head is not None:
                self.visual_token_cosine_head.requires_grad_(True)

            trainable_backbone_keys = ["projector", "llm_backbone"]
            if stage == "vla-vlm-peft-train":
                trainable_backbone_keys.insert(0, "vision_backbone")

            self.trainable_module_keys = trainable_backbone_keys
            if self.action_queries is not None:
                self.trainable_module_keys.append("action_queries")
            if self.action_head is not None:
                self.trainable_module_keys.append("action_head")
            if self.aux_head is not None:
                self.trainable_module_keys.append("aux_head")
            if self.visual_token_cosine_head is not None:
                self.trainable_module_keys.append("visual_token_cosine_head")

            vision_lora_trainable = stage == "vla-vlm-peft-train"
            self.vision_backbone_requires_grad = vision_lora_trainable

            overwatch.info(f"[Frozen]    🥶 =>> Base Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Base LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> Base Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(
                f"[TRAINABLE] 🔥 =>> Full-VLM PEFT Adapters (`{len(lora_param_names)}` parameter groups matched)",
                ctx_level=1,
            )
            if not vision_lora_trainable:
                overwatch.info("[TRAINABLE] 🔥 =>> Vision Backbone excluded from PEFT targets", ctx_level=1)
            if modules_to_save_names:
                overwatch.info(
                    f"[TRAINABLE] 🔥 =>> Modules-To-Save (`{len(modules_to_save_names)}` parameter groups matched)",
                    ctx_level=1,
                )
            if self.action_queries is not None:
                log_action_queries_trainable()
            if self.action_head is not None:
                log_action_head_trainable()
            if self.aux_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Aux Head", ctx_level=1)
            if self.visual_token_cosine_head is not None:
                overwatch.info(f"[TRAINABLE] 🔥 =>> Visual Token Cosine Head", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> projector + action/aux heads
        head_classes = {LinearProjector, MLPProjector, FusedMLPProjector, FusedFANProjector}
        if self.action_head is not None:
            from prismatic.models.action_heads import ActionHeadBackbone, SelfAttnBlock, CrossAttnBlock
            from prismatic.models.action_heads import MLPResNet, MLPResNetBlock, MLPResNetBlockPro
            head_classes.update({ActionHeadBackbone, SelfAttnBlock, CrossAttnBlock, MLPResNet, MLPResNetBlock, MLPResNetBlockPro})
        if self.aux_head is not None:
            from prismatic.models.action_heads import AuxDecoderBlock
            head_classes.add(AuxDecoderBlock)

        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes=head_classes,
        )

        # Return union (_or_) over constituent policies
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        future_pixel_values: Optional[torch.FloatTensor] = None,
        pair_pixel_values: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_valid_mask: Optional[torch.BoolTensor] = None,
        action_valid_dim: Optional[int] = None,
        proprio: Optional[torch.FloatTensor] = None,
        action_expert_only: bool = False,
        context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Run a forward pass through the Predictor (V-JEPA + LLM).

        Returns a dict with:
            - loss: LLM causal loss (for backward compatibility)
            - llm_hidden: last layer hidden states [B, L, D_llm]
            - vjepa_target: future frame V-JEPA embeddings [B, V, 4, 14, 14, D_jepa] (if future_pixel_values provided)
        """

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        memory_stats = [] if getattr(self, "debug_memory_stats", False) else None

        # Run Visual Feature Extraction (current frames)
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_vision_encode")) is not None:
            memory_stats.append(snap)
        current_vjepa = patch_features

        # Encode future frames for aux target (if provided)
        vjepa_target = None
        if future_pixel_values is not None and hasattr(self.vision_backbone, "encode_future"):
            if isinstance(future_pixel_values, dict):
                vjepa_target = self.vision_backbone.encode_future(
                    {k: future_pixel_values[k][multimodal_indices] for k in future_pixel_values}
                )
            else:
                vjepa_target = self.vision_backbone.encode_future(future_pixel_values[multimodal_indices])
            if not getattr(self, "_printed_vjepa_target_debug", False):
                overwatch.info(
                    "Future JEPA target shape: %s (patch-wise target preserved; temporal dim reflects downsampled frames after tubelet aggregation)",
                    tuple(vjepa_target.shape),
                )
                self._printed_vjepa_target_debug = True

        pair_vjepa_target = None
        if (
            self.visual_token_cosine_head is not None
            and pair_pixel_values is not None
            and hasattr(self.vision_backbone, "encode_future")
        ):
            if isinstance(pair_pixel_values, dict):
                pair_vjepa_target = self.vision_backbone.encode_future(
                    {k: pair_pixel_values[k][multimodal_indices] for k in pair_pixel_values}
                )
            else:
                pair_vjepa_target = self.vision_backbone.encode_future(pair_pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim]
        projected_patch_embeddings = self.projector(patch_features)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_projector")) is not None:
            memory_stats.append(snap)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        input_embeddings = self._inject_action_query_embeddings(input_ids, input_embeddings)
        if self.use_context and context is not None:
            cp = context["proprio"].to(input_embeddings.device, input_embeddings.dtype)
            ca = context["actions"].to(input_embeddings.device, input_embeddings.dtype).flatten(1)
            dt = context["delta_t"].to(input_embeddings.device).clamp(-128, 128) + 128
            te = self.context_time_embedding(dt).unsqueeze(1)
            st = self.context_state_encoder(cp).unsqueeze(1) + te
            at = self.context_action_encoder(ca).view(ca.shape[0], self.context_action_tokens, -1) + te
            slots = torch.arange(self.context_action_tokens, device=at.device)
            ctx = torch.cat([st, at + self.context_action_slot_embedding(slots).unsqueeze(0)], dim=1)
            n = self.action_placeholder_tokens
            if n <= 0: raise RuntimeError("context requires action placeholder tokens")
            input_embeddings = torch.cat([input_embeddings[:, :-n], ctx, input_embeddings[:, -n:]], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask[:, :-n], torch.ones(ctx.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype), attention_mask[:, -n:]], dim=1)
            if labels is not None:
                labels = torch.cat([labels[:, :-n], torch.full(ctx.shape[:2], IGNORE_INDEX, device=labels.device, dtype=labels.dtype), labels[:, -n:]], dim=1)

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Add Unimodal Handling ===
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward with output_hidden_states=True. If all labels are
        # ignored, skip LM CE loss entirely; continuous heads provide training.
        llm_labels = fused_labels
        if llm_labels is not None and not torch.any(llm_labels != IGNORE_INDEX):
            llm_labels = None
        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=llm_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_llm_forward")) is not None:
            memory_stats.append(snap)

        # === Action Head & Aux Head Forward ===
        llm_hidden = llm_output.hidden_states[-1] if llm_output.hidden_states is not None else None
        loss_llm_ce = llm_output.loss if llm_output.loss is not None else None
        if action_expert_only:
            loss_llm_ce = None
        total_loss = (
            self.lambda_llm_ce * loss_llm_ce
            if loss_llm_ce is not None
            else torch.tensor(0.0, device=input_ids.device)
        )
        loss_action = None
        loss_jepa = None
        loss_aux = None
        loss_visual_token_cosine = None
        aux_pred = None
        pred_jepa_delta = None

        if llm_hidden is not None:
            # Determine number of views from pixel_values shape
            V = 1
            if pixel_values is not None:
                if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
                    V = pixel_values.shape[1]
                elif isinstance(pixel_values, dict):
                    pv_example = next(iter(pixel_values.values()))
                    if pv_example.dim() == 5:
                        V = pv_example.shape[1]

            action_placeholder_tokens = min(self.action_placeholder_tokens, max(0, llm_hidden.shape[1] - 1))
            z_action = llm_hidden[:, -1, :]  # [B, D_llm]
            vision_token_count = projected_patch_embeddings.shape[1]
            vision_memory = llm_hidden[:, 1 : 1 + vision_token_count, :]
            aux_vision_memory = vision_memory
            if V > 1:
                tokens_per_view = vision_token_count // V
                if tokens_per_view * V != vision_token_count:
                    raise ValueError(
                        f"Vision token count {vision_token_count} is not divisible by num views {V}."
                    )
                aux_vision_memory = vision_memory[:, :tokens_per_view, :]
            action_memory = None
            if action_placeholder_tokens > 0:
                action_memory = llm_hidden[:, -action_placeholder_tokens:, :]
            if self.use_llm_ce_loss and fused_labels is not None and self.action_head_type in {"flow_gr00t", "flow_gr00t_jepa"}:
                action_memory = _build_prefix_condition_memory(llm_hidden, fused_labels)
            if (
                self.flow_gr00t_use_full_llm_hidden
                and not self.use_llm_ce_loss
                and self.action_head_type in {"flow_gr00t", "flow_gr00t_jepa"}
            ):
                action_memory = llm_hidden
            if action_memory is None:
                aux_memory = aux_vision_memory
            else:
                aux_memory = torch.cat([aux_vision_memory, action_memory], dim=1)

            if action_expert_only:
                z_action = z_action.detach()
                vision_memory = vision_memory.detach()
                aux_vision_memory = aux_vision_memory.detach()
                aux_memory = aux_memory.detach()
                if action_memory is not None:
                    action_memory = action_memory.detach()
                current_vjepa = current_vjepa.detach() if current_vjepa is not None else None

            # Action Head (Flow Matching)
            if self.action_head is not None and actions is not None and proprio is not None:
                    # Ensure actions has the right shape [B, H_a, D_action]
                    if actions.dim() == 4 and actions.shape[1] == 1:
                        actions = actions.squeeze(1)
                    if actions.dim() == 2:
                        actions = actions.unsqueeze(1)  # [B, 1, D_action]
                    
                    # Ensure proprio has the right shape [B, D_proprio]
                    if proprio.dim() == 3 and proprio.shape[1] == 1:
                        proprio = proprio.squeeze(1)

                    if self.action_head_type == "flow_gr00t":
                        if action_memory is None:
                            raise RuntimeError("flow_gr00t requires a non-empty action condition sequence.")
                        loss_action, _ = self.action_head(
                            action_memory,
                            proprio,
                            actions,
                            action_valid_mask=action_valid_mask,
                            action_valid_dim=action_valid_dim,
                        )
                    elif self.action_head_type == "flow_gr00t_jepa":
                        if action_memory is None:
                            raise RuntimeError("flow_gr00t_jepa requires a non-empty action condition sequence.")
                        if vjepa_target is None:
                            raise RuntimeError("flow_gr00t_jepa requires future JEPA targets during training.")
                        joint_loss, _, pred_jepa_delta, loss_action, loss_jepa = self.action_head(
                            action_memory,
                            proprio,
                            actions,
                            current_vjepa=current_vjepa,
                            future_jepa_target=vjepa_target,
                            num_views=V,
                            action_valid_mask=action_valid_mask,
                            action_valid_dim=action_valid_dim,
                        )
                        total_loss = total_loss + (loss_action if action_expert_only else joint_loss)
                    elif self.action_head_type == "l1":
                        loss_action, _ = self.action_head(
                            z_action,
                            proprio,
                            actions,
                            hidden_states=llm_output.hidden_states,
                            task_token_count=projected_patch_embeddings.shape[1],
                            action_token_count=action_placeholder_tokens,
                            phase="Training" if self.training else "Inference",
                            action_valid_mask=action_valid_mask,
                        )
                    else:
                        raise ValueError(f"Unsupported action_head_type: {self.action_head_type}")
                    if self.action_head_type != "flow_gr00t_jepa":
                        total_loss = total_loss + loss_action

            # Aux Head (Future JEPA Embedding Prediction)
            if self.aux_head is not None and vjepa_target is not None and self.training and not action_expert_only:
                target_views = vjepa_target.shape[1]
                if not getattr(self, "_printed_aux_patchwise_debug", False):
                    overwatch.info(
                        "Aux head consuming patch-wise JEPA target with shape %s (V=%d, T=%d, H=%d, W=%d, D=%d)",
                        tuple(vjepa_target.shape),
                        vjepa_target.shape[1],
                        vjepa_target.shape[2],
                        vjepa_target.shape[3],
                        vjepa_target.shape[4],
                        vjepa_target.shape[5],
                    )
                    self._printed_aux_patchwise_debug = True
                aux_pred = self.aux_head(aux_memory, V=target_views)
                # Per-patch layer-normalized MSE (no learnable params)
                pred_n = F.layer_norm(aux_pred, aux_pred.shape[-1:])
                target_n = F.layer_norm(vjepa_target, vjepa_target.shape[-1:])
                loss_aux = F.mse_loss(pred_n, target_n)
                total_loss = total_loss + self.lambda_aux * loss_aux
                if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_aux_head")) is not None:
                    memory_stats.append(snap)

            if (
                self.visual_token_cosine_head is not None
                and pair_vjepa_target is not None
                and self.training
                and not action_expert_only
            ):
                if llm_output.hidden_states is None:
                    raise RuntimeError("visual_token_cosine_head requires LLM hidden states.")
                cosine_layer_idx = self.visual_token_cosine_layer_idx
                num_hidden_layers = len(llm_output.hidden_states)
                if not -num_hidden_layers <= cosine_layer_idx < num_hidden_layers:
                    raise ValueError(
                        f"visual_token_cosine_layer_idx={cosine_layer_idx} is out of range for "
                        f"{num_hidden_layers} hidden-state tensors."
                    )
                cosine_llm_hidden = llm_output.hidden_states[cosine_layer_idx]
                if not getattr(self, "_printed_visual_token_cosine_layer_debug", False):
                    overwatch.info(
                        "Visual-token cosine supervision using LLM hidden_states[%s] with shape %s",
                        cosine_layer_idx,
                        tuple(cosine_llm_hidden.shape),
                    )
                    self._printed_visual_token_cosine_layer_debug = True
                if pair_vjepa_target.shape[2] != 1:
                    raise ValueError(
                        "visual_token_cosine_head expects a JEPA target with temporal token dim 1, "
                        f"got {tuple(pair_vjepa_target.shape)}"
                    )
                cosine_vision_memory = cosine_llm_hidden[:, 1 : 1 + vision_token_count, :]
                cosine_pair_target_grid = pair_vjepa_target.squeeze(2)
                target_views = cosine_pair_target_grid.shape[1]
                target_height = cosine_pair_target_grid.shape[2]
                target_width = cosine_pair_target_grid.shape[3]
                if self.visual_token_cosine_projection_type == "conv":
                    target_spatial = cosine_pair_target_grid.reshape(
                        cosine_pair_target_grid.shape[0],
                        target_views,
                        target_height * target_width,
                        cosine_pair_target_grid.shape[-1],
                    )
                    target_spatial = target_spatial - target_spatial.mean(dim=2, keepdim=True)
                    target_spatial = target_spatial / target_spatial.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
                    cosine_pair_target_grid = target_spatial.reshape_as(cosine_pair_target_grid)
                cosine_pair_target = cosine_pair_target_grid.reshape(
                    cosine_pair_target_grid.shape[0],
                    target_views * target_height * target_width,
                    cosine_pair_target_grid.shape[-1],
                )
                if not getattr(self, "_printed_visual_token_cosine_shape_debug", False):
                    overwatch.info(
                        "Visual-token cosine check: projection=%s input_views=%d target_views=%d pred_tokens=%d target_tokens=%d",
                        self.visual_token_cosine_projection_type,
                        V,
                        target_views,
                        cosine_vision_memory.shape[1],
                        cosine_pair_target.shape[1],
                    )
                    self._printed_visual_token_cosine_shape_debug = True
                if self.visual_token_cosine_use_projector_target:
                    target_visual_tokens = self.projector(cosine_pair_target).detach()
                else:
                    target_visual_tokens = cosine_pair_target.detach()

                if cosine_vision_memory.shape[1] != target_visual_tokens.shape[1]:
                    raise ValueError(
                        f"Visual token count mismatch for cosine supervision: pred_seq={cosine_vision_memory.shape[1]} "
                        f"target={tuple(target_visual_tokens.shape)}"
                    )
                loss_visual_token_cosine, visual_pred = self.visual_token_cosine_head(
                    cosine_vision_memory,
                    target_visual_tokens,
                    num_views=target_views,
                    spatial_hw=(target_height, target_width),
                )
                total_loss = total_loss + self.lambda_visual_token_cosine * loss_visual_token_cosine
                if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_visual_token_cosine_head")) is not None:
                    memory_stats.append(snap)

        # Build Predictor output dict
        output = {
            "loss": total_loss,
            "logits": llm_output.logits,
            "llm_hidden": llm_hidden,
            "llm_hidden_states": llm_output.hidden_states,
            "vjepa_target": vjepa_target,
            "pair_vjepa_target": pair_vjepa_target,
            "current_vjepa": current_vjepa,
            "task_token_count": projected_patch_embeddings.shape[1],
            "action_token_count": action_placeholder_tokens if llm_hidden is not None else 0,
        }
        if aux_pred is not None:
            output["aux_pred"] = aux_pred
        if loss_action is not None:
            output["loss_action"] = loss_action
        if loss_llm_ce is not None:
            output["loss_llm_ce"] = loss_llm_ce
        if loss_jepa is not None:
            output["loss_jepa"] = loss_jepa
        if pred_jepa_delta is not None:
            output["pred_jepa_delta"] = pred_jepa_delta
        if loss_aux is not None:
            output["loss_aux"] = loss_aux
        if loss_visual_token_cosine is not None:
            output["loss_visual_token_cosine"] = loss_visual_token_cosine
        if memory_stats is not None:
            output["memory_stats"] = memory_stats
        return output

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text
