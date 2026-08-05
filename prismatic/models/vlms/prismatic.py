"""
prismatic.py

PyTorch module implementing the fixed JEPA-WAM policy.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Type, Union

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from prismatic.models.action_heads import VisualTokenCosineHead
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import MLPProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, NUM_TOKENS, PROPRIO_DIM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


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

        # The released base VLM uses the two-layer GELU projector.
        self.arch_specifier = arch_specifier
        if not arch_specifier.endswith("gelu-mlp"):
            raise ValueError(f"The public JEPA-WAM recipe requires a GELU MLP projector, got `{arch_specifier}`.")
        self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)

        # Trackers
        self.vision_backbone_requires_grad = False

        # Fixed public heads: Flow-GR00T plus final-layer visual-token cosine alignment.
        self.action_placeholder_tokens = kwargs.get("flow_gr00t_placeholder_tokens", NUM_TOKENS)
        self.lambda_visual_token_cosine = kwargs.get("lambda_visual_token_cosine", 0.5)
        self.action_head = FlowMatchingActionHead(
            d_proprio=kwargs.get("d_proprio", PROPRIO_DIM),
            d_action=kwargs.get("d_action", ACTION_DIM),
            d_llm=llm_backbone.embed_dim,
            horizon=kwargs.get("action_horizon", NUM_ACTIONS_CHUNK),
            fm_hidden_size=kwargs.get("fm_hidden_size", 1024),
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
        self.visual_token_cosine_head = VisualTokenCosineHead(
            d_llm=llm_backbone.embed_dim,
            d_target=kwargs.get("d_jepa", vision_backbone.embed_dim),
        )

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = [
            "vision_backbone",
            "llm_backbone",
            "projector",
            "action_head",
            "visual_token_cosine_head",
        ]
        self.trainable_module_keys = []
        overwatch.info("Initialized Flow-GR00T action head with %d placeholder tokens", self.action_placeholder_tokens)

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
        if "action_head" in model_state_dict:
            vlm.action_head.load_state_dict(model_state_dict["action_head"])
        if "visual_token_cosine_head" in model_state_dict:
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

    def freeze_for_training(self) -> None:
        """Freeze the fixed base model and train only Qwen LoRA plus the two public heads."""
        self.vision_backbone.requires_grad_(False)
        self.projector.requires_grad_(False)
        self.llm_backbone.requires_grad_(False)

        lora_param_names = []
        for name, param in self.llm_backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                lora_param_names.append(name)
        if not lora_param_names:
            raise RuntimeError("Qwen must be wrapped with LoRA before calling `freeze_for_training`.")

        self.action_head.requires_grad_(True)
        self.visual_token_cosine_head.requires_grad_(True)
        self.trainable_module_keys = ["llm_backbone", "action_head", "visual_token_cosine_head"]
        self.vision_backbone_requires_grad = False

        overwatch.info(f"[Frozen] =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
        overwatch.info(f"[Frozen] =>> Projector `{self.arch_specifier}`", ctx_level=1)
        overwatch.info(
            f"[TRAINABLE] =>> Qwen LoRA (`{len(lora_param_names)}` parameter groups matched)",
            ctx_level=1,
        )
        overwatch.info(
            f"[TRAINABLE] =>> Flow-GR00T Action Head (placeholders={self.action_placeholder_tokens})",
            ctx_level=1,
        )
        overwatch.info("[TRAINABLE] =>> Visual Token Cosine Head", ctx_level=1)

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> projector and fixed action/alignment heads
        head_classes = {MLPProjector, FlowMatchingActionHead, VisualTokenCosineHead}

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

    @staticmethod
    def _select_action_memory(
        llm_hidden: torch.Tensor,
        fused_attention_mask: torch.Tensor,
        num_action_tokens: int,
    ) -> torch.Tensor:
        """Gather the final non-padding tokens from every sample."""
        valid_lengths = fused_attention_mask.long().sum(dim=1)
        if torch.any(valid_lengths < num_action_tokens):
            raise ValueError("Input sequence is shorter than the configured action placeholder span.")

        offsets = torch.arange(num_action_tokens, device=llm_hidden.device)
        positions = valid_lengths[:, None] - num_action_tokens + offsets[None, :]
        gather_index = positions[..., None].expand(-1, -1, llm_hidden.shape[-1])
        return llm_hidden.gather(1, gather_index)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Union[torch.FloatTensor, Dict[str, torch.Tensor]],
        pair_pixel_values: Optional[Union[torch.FloatTensor, Dict[str, torch.Tensor]]] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_valid_mask: Optional[torch.BoolTensor] = None,
        action_valid_dim: Optional[int] = None,
        proprio: Optional[torch.FloatTensor] = None,
    ) -> dict:
        """Run the fixed JEPA-WAM action and visual-alignment forward pass."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected input_ids and attention_mask with matching [B, L] shapes.")

        memory_stats = [] if getattr(self, "debug_memory_stats", False) else None

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            patch_features = self.vision_backbone(pixel_values)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_vision_encode")) is not None:
            memory_stats.append(snap)

        pair_vjepa_target = None
        if pair_pixel_values is not None:
            if not hasattr(self.vision_backbone, "encode_pair"):
                raise TypeError("The configured vision backbone does not implement paired-frame encoding.")
            with torch.no_grad():
                pair_vjepa_target = self.vision_backbone.encode_pair(pair_pixel_values)

        projected_patch_embeddings = self.projector(patch_features)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_projector")) is not None:
            memory_stats.append(snap)

        projected_patch_attention_mask = torch.ones(
            projected_patch_embeddings.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        fused_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            dim=1,
        )
        fused_attention_mask = torch.cat(
            [
                attention_mask[:, :1],
                projected_patch_attention_mask,
                attention_mask[:, 1:],
            ],
            dim=1,
        )

        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_llm_forward")) is not None:
            memory_stats.append(snap)
        if llm_output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states.")

        llm_hidden = llm_output.hidden_states[-1]
        vision_token_count = projected_patch_embeddings.shape[1]
        vision_memory = llm_hidden[:, 1 : 1 + vision_token_count, :]
        action_memory = self._select_action_memory(
            llm_hidden,
            fused_attention_mask,
            self.action_placeholder_tokens,
        )

        if isinstance(pixel_values, torch.Tensor):
            num_views = pixel_values.shape[1] if pixel_values.ndim == 5 else 1
        else:
            example = next(iter(pixel_values.values()))
            num_views = example.shape[1] if example.ndim == 5 else 1
        if vision_token_count % num_views != 0:
            raise ValueError(
                f"Vision token count {vision_token_count} is not divisible by num views {num_views}."
            )

        total_loss = llm_hidden.new_zeros(())
        loss_action = None
        loss_visual_token_cosine = None

        if (actions is None) != (proprio is None):
            raise ValueError("Actions and proprio must be provided together.")
        if actions is not None:
            if actions.ndim == 4 and actions.shape[1] == 1:
                actions = actions.squeeze(1)
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)
            if proprio.ndim == 3 and proprio.shape[1] == 1:
                proprio = proprio.squeeze(1)

            loss_action, _ = self.action_head(
                action_memory,
                proprio,
                actions,
                action_valid_mask=action_valid_mask,
                action_valid_dim=action_valid_dim,
            )
            total_loss = total_loss + loss_action

        if pair_vjepa_target is not None and self.training:
            if pair_vjepa_target.shape[2] != 1:
                raise ValueError(
                    "Visual-token cosine supervision expects one temporal target token, "
                    f"got {tuple(pair_vjepa_target.shape)}."
                )
            target_grid = pair_vjepa_target.squeeze(2)
            target_visual_tokens = target_grid.reshape(
                target_grid.shape[0],
                target_grid.shape[1] * target_grid.shape[2] * target_grid.shape[3],
                target_grid.shape[-1],
            ).detach()
            if vision_memory.shape[1] != target_visual_tokens.shape[1]:
                raise ValueError(
                    f"Visual token count mismatch: prediction={vision_memory.shape[1]}, "
                    f"target={target_visual_tokens.shape[1]}."
                )

            loss_visual_token_cosine, _ = self.visual_token_cosine_head(
                vision_memory,
                target_visual_tokens,
            )
            total_loss = total_loss + self.lambda_visual_token_cosine * loss_visual_token_cosine
            if memory_stats is not None and (
                snap := _maybe_cuda_mem_snapshot("after_visual_token_cosine_head")
            ) is not None:
                memory_stats.append(snap)

        output = {"loss": total_loss, "llm_hidden": llm_hidden}
        if loss_action is not None:
            output["loss_action"] = loss_action
        if loss_visual_token_cosine is not None:
            output["loss_visual_token_cosine"] = loss_visual_token_cosine
        if memory_stats is not None:
            output["memory_stats"] = memory_stats
        return output
