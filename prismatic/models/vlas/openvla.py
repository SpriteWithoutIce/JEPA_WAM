"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer
        self._context_proprio = None
        self._context_actions = None

    def reset_context(self) -> None:
        """Clear episode-local action context; call after every environment reset."""
        self._context_proprio = None
        self._context_actions = None

    @torch.inference_mode()
    def predict_action(
        self,
        image: Union[Img, List[Img]],
        instruction: str,
        unnorm_key: Optional[str] = None,
        proprio: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        reset_context: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        if reset_context:
            self.reset_context()
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        use_continuous_head = (
            self.action_head is not None
            and proprio is not None
            and hasattr(self.action_head, "sample_action")
        )
        use_llm_ce_loss = getattr(self, "use_llm_ce_loss", False)

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        # Match the training prompt format for continuous JEPA-VLA heads: the
        # dataset path adds an empty assistant turn before appending action placeholders.
        if use_continuous_head:
            prompt_builder.add_turn(role="gpt", message="")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        tokenized = tokenizer(prompt_text, truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            # do nothing here. I think...
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image(s)
        if isinstance(image, list):
            # Routed multi-view backbones (e.g. primary->JEPA, wrist->DINO) expose
            # view-specific transforms so inference matches the training data path.
            if hasattr(image_transform, "transform_primary") and hasattr(image_transform, "transform_wrist"):
                primary_tensor = image_transform.transform_primary(image[0])
                wrist_tensors = [image_transform.transform_wrist(img) for img in image[1:]]
                pixel_values = {k: v[None, ...].to(self.device) for k, v in primary_tensor.items()}
                for wrist_tensor in wrist_tensors:
                    for key, value in wrist_tensor.items():
                        if key in pixel_values:
                            raise ValueError(
                                "Routed multi-view inference expects distinct transform keys per view, "
                                f"but key `{key}` appeared more than once."
                            )
                        pixel_values[key] = value[None, ...].to(self.device)
            else:
                img_tensors = [image_transform(img) for img in image]
                if all(isinstance(t, torch.Tensor) for t in img_tensors):
                    pixel_values = torch.stack(img_tensors, dim=0)[None, ...].to(self.device)  # [1, V, 3, H, W]
                elif all(isinstance(t, dict) for t in img_tensors):
                    transform_keys = img_tensors[0].keys()
                    if not all(t.keys() == transform_keys for t in img_tensors):
                        raise ValueError("List image transform dict outputs must share identical keys.")
                    pixel_values = {
                        k: torch.stack([t[k] for t in img_tensors], dim=0)[None, ...].to(self.device)
                        for k in transform_keys
                    }
                else:
                    raise ValueError("List image transform must return Tensor or dict for each image.")
        else:
            pixel_values = image_transform(image)
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values[None, ...].to(self.device)
            elif isinstance(pixel_values, dict):
                pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        num_views = 1
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            num_views = pixel_values.shape[1]
        elif isinstance(pixel_values, dict):
            pv_example = next(iter(pixel_values.values()))
            if pv_example.dim() == 5:
                num_views = pv_example.shape[1]

        if use_continuous_head:
            action_placeholder_tokens = getattr(self, "action_placeholder_tokens", 0)
            if action_placeholder_tokens > 0 and not use_llm_ce_loss:
                placeholder_ids = torch.full(
                    (input_ids.shape[0], action_placeholder_tokens),
                    fill_value=ACTION_TOKEN_BEGIN_IDX,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, placeholder_ids], dim=1)
                placeholder_mask = torch.ones(
                    (attention_mask.shape[0], action_placeholder_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, placeholder_mask], dim=1)

            if isinstance(proprio, torch.Tensor):
                proprio_t = proprio.to(self.device, dtype=torch.float32)
            else:
                proprio_t = torch.tensor(np.asarray(proprio), device=self.device, dtype=torch.float32)
            if proprio_t.dim() == 1:
                proprio_t = proprio_t.unsqueeze(0)

            if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
                key = self._check_unnorm_key(self.norm_stats, unnorm_key)
                proprio_norm_stats = self.norm_stats[key]["proprio"]
                proprio_high = torch.tensor(np.array(proprio_norm_stats["q99"]), device=self.device, dtype=torch.float32)
                proprio_low = torch.tensor(np.array(proprio_norm_stats["q01"]), device=self.device, dtype=torch.float32)
                mask = torch.tensor(
                    np.array(proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))),
                    device=self.device,
                    dtype=torch.bool,
                )
                stats_dim = proprio_high.shape[-1]
                if proprio_t.shape[-1] < stats_dim:
                    raise ValueError(
                        f"Proprio dim {proprio_t.shape[-1]} is smaller than normalization stats dim {stats_dim}."
                    )
                normalized_proprio_prefix = torch.where(
                    mask,
                    2 * (proprio_t[..., :stats_dim] - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                    proprio_t[..., :stats_dim],
                )
                normalized_proprio = (
                    torch.cat([normalized_proprio_prefix, proprio_t[..., stats_dim:]], dim=-1)
                    if proprio_t.shape[-1] > stats_dim
                    else normalized_proprio_prefix
                )
            else:
                normalized_proprio = proprio_t

            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            autocast_dtype = self.llm_backbone.half_precision_dtype
            context = None
            if getattr(self, "use_context", False) and self._context_proprio is not None:
                context = {
                    "proprio": self._context_proprio.to(self.device),
                    "actions": self._context_actions.to(self.device),
                    "delta_t": torch.full((1,), -int(self.action_head.action_horizon), device=self.device, dtype=torch.long),
                }
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
                outputs = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=None,
                    output_hidden_states=True,
                    return_dict=True,
                    context=context,
                )
                llm_hidden = outputs["llm_hidden"]
                hidden_states = outputs.get("llm_hidden_states", None)
                if hidden_states is None:
                    # Fallback to only final hidden state repeated once.
                    hidden_states = (llm_hidden,)
                task_token_count = outputs.get("task_token_count", self.vision_backbone.num_patches)
                action_token_count = outputs.get("action_token_count", getattr(self, "action_placeholder_tokens", 0))
                z_action = llm_hidden[:, -1, :]
                use_full_llm_hidden = getattr(self, "flow_gr00t_use_full_llm_hidden", False)

                if self.action_head_type == "flow_gr00t":
                    if use_llm_ce_loss or use_full_llm_hidden:
                        action_condition = llm_hidden
                    else:
                        if action_token_count <= 0:
                            raise RuntimeError("flow_gr00t inference requires action placeholder tokens.")
                        action_condition = llm_hidden[:, -action_token_count:, :]
                    normalized_actions = self.action_head.predict_action(
                        action_condition,
                        normalized_proprio,
                    )
                elif self.action_head_type == "flow_gr00t_jepa":
                    if use_llm_ce_loss or use_full_llm_hidden:
                        action_condition = llm_hidden
                    else:
                        if action_token_count <= 0:
                            raise RuntimeError("flow_gr00t_jepa inference requires action placeholder tokens.")
                        action_condition = llm_hidden[:, -action_token_count:, :]
                    normalized_actions = self.action_head.predict_action(
                        action_condition,
                        normalized_proprio,
                        current_vjepa=outputs["current_vjepa"],
                        num_views=num_views,
                    )
                elif hasattr(self.action_head, "predict_action"):
                    normalized_actions = self.action_head.predict_action(
                        z_action,
                        normalized_proprio,
                        hidden_states=hidden_states,
                        task_token_count=task_token_count,
                        action_token_count=action_token_count,
                        phase="Inference",
                    )
                else:
                    normalized_actions = self.action_head.sample_action(
                        z_action,
                        normalized_proprio,
                    )
            # Store normalized values: training context is normalized in the RLDS pipeline.
            if getattr(self, "use_context", False):
                self._context_proprio = normalized_proprio.detach().float()
                self._context_actions = normalized_actions.detach().float()
            normalized_actions = normalized_actions.detach().float().cpu().numpy()[0]

            action_norm_stats = self.get_action_stats(unnorm_key)
            if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
            elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            else:
                raise ValueError("Unsupported action/proprio normalization type detected!")
            stats_dim = action_high.shape[-1]
            if normalized_actions.shape[-1] < stats_dim:
                raise ValueError(
                    f"Action dim {normalized_actions.shape[-1]} is smaller than normalization stats dim {stats_dim}."
                )
            actions_prefix = np.where(
                mask,
                0.5 * (normalized_actions[..., :stats_dim] + 1) * (action_high - action_low + 1e-8) + action_low,
                normalized_actions[..., :stats_dim],
            )
            actions = (
                np.concatenate([actions_prefix, normalized_actions[..., stats_dim:]], axis=-1)
                if normalized_actions.shape[-1] > stats_dim
                else actions_prefix
            )
            return actions

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
