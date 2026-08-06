"""Load local JEPA-WAM checkpoints produced by the public training recipe."""

import json
import os
from pathlib import Path
from typing import Optional, Union

import torch

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlas import OpenVLA
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

overwatch = initialize_overwatch(__name__)


def _normalize_lora_target_modules(target_modules):
    if isinstance(target_modules, tuple):
        return list(target_modules)
    if isinstance(target_modules, str) and "," in target_modules:
        return [item.strip() for item in target_modules.split(",") if item.strip()]
    return target_modules


def _apply_lora_to_llm_backbone(llm_backbone, vla_cfg: dict, is_trainable: bool) -> None:
    from peft import LoraConfig, get_peft_model

    if hasattr(llm_backbone.llm, "peft_config"):
        return
    llm_backbone.llm = get_peft_model(
        llm_backbone.llm,
        LoraConfig(
            r=int(vla_cfg.get("lora_rank", 32)),
            lora_alpha=int(vla_cfg.get("lora_alpha", 64)),
            target_modules=_normalize_lora_target_modules(vla_cfg.get("lora_target_modules", "all-linear")),
            lora_dropout=float(vla_cfg.get("lora_dropout", 0.1)),
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            inference_mode=not is_trainable,
        ),
    )


def _resolve_base_config(base_vlm: Union[str, Path]) -> tuple[Path, dict]:
    path = Path(os.path.expanduser(str(base_vlm)))
    run_dir = path if path.is_dir() else path.parent.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Base VLM config not found: `{config_path}`")
    with open(config_path, "r") as handle:
        return run_dir, json.load(handle)["model"]


def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    load_for_training: bool = False,
    base_vlm: Optional[Union[str, Path]] = None,
    llm_checkpoint_path: Optional[str] = None,
    vjepa_checkpoint_path: Optional[str] = None,
    load_visual_token_cosine_head: bool = True,
    **_: object,
) -> OpenVLA:
    checkpoint_path = Path(os.path.expanduser(str(model_id_or_path)))
    if not checkpoint_path.is_file() or checkpoint_path.suffix != ".pt" or checkpoint_path.parent.name != "checkpoints":
        raise ValueError("JEPA-WAM loading requires a local `runs/.../checkpoints/*.pt` checkpoint.")

    run_dir = checkpoint_path.parents[1]
    config_path = run_dir / "config.json"
    statistics_path = run_dir / "dataset_statistics.json"
    if not config_path.exists() or not statistics_path.exists():
        raise ValueError(f"Checkpoint run directory must contain config.json and dataset_statistics.json: `{run_dir}`")

    with open(config_path, "r") as handle:
        full_cfg = json.load(handle)
    with open(statistics_path, "r") as handle:
        norm_stats = json.load(handle)
    vla_cfg = full_cfg["vla"]

    base_source = base_vlm or vla_cfg.get("base_vlm")
    if not base_source:
        raise ValueError("Pass a local base VLM run directory through `base_vlm`.")
    _, model_cfg = _resolve_base_config(base_source)

    vision_id = model_cfg.get("vision_backbone_id")
    llm_id = model_cfg.get("llm_backbone_id")
    vision_checkpoint = vjepa_checkpoint_path or vla_cfg.get("vjepa_checkpoint_path") or model_cfg.get(
        "vision_checkpoint_path"
    )
    llm_checkpoint = llm_checkpoint_path or full_cfg.get("llm_checkpoint_path") or model_cfg.get("llm_local_path")
    if not vision_checkpoint or not llm_checkpoint:
        raise ValueError("Both V-JEPA and Qwen checkpoint paths are required to reconstruct JEPA-WAM.")

    overwatch.info(f"Loading JEPA-WAM checkpoint `{checkpoint_path}`")
    vision_backbone, _ = get_vision_backbone_and_transform(
        vision_id,
        model_cfg.get("image_resize_strategy", "resize-naive"),
        checkpoint_path=str(vision_checkpoint),
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        llm_id,
        llm_max_length=int(model_cfg.get("llm_max_length", 32_768)),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        custom_hf_path=str(llm_checkpoint),
    )
    _apply_lora_to_llm_backbone(llm_backbone, vla_cfg, is_trainable=load_for_training)

    model = OpenVLA.from_pretrained(
        checkpoint_path,
        model_cfg.get("model_id", "prism-qwen25-vjepa21-vitl-384px+0_5b"),
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.get("arch_specifier", "no-align+gelu-mlp"),
        freeze_weights=not load_for_training,
        load_visual_token_cosine_head=load_visual_token_cosine_head,
        norm_stats=norm_stats,
        d_action=int(vla_cfg.get("d_action", 7)),
        d_proprio=int(vla_cfg.get("d_proprio", 8)),
        action_horizon=int(vla_cfg.get("action_horizon", NUM_ACTIONS_CHUNK)),
        flow_gr00t_placeholder_tokens=int(vla_cfg.get("flow_gr00t_placeholder_tokens", 64)),
        fm_hidden_size=int(vla_cfg.get("fm_hidden_size", 1024)),
        fm_num_layers=int(vla_cfg.get("fm_num_layers", 16)),
        fm_num_inference_timesteps=int(vla_cfg.get("fm_num_inference_timesteps", 4)),
        fm_num_timestep_buckets=int(vla_cfg.get("fm_num_timestep_buckets", 1000)),
        fm_noise_beta_alpha=float(vla_cfg.get("fm_noise_beta_alpha", 1.5)),
        fm_noise_beta_beta=float(vla_cfg.get("fm_noise_beta_beta", 1.0)),
        fm_noise_s=float(vla_cfg.get("fm_noise_s", 0.999)),
        fm_num_target_vision_tokens=int(vla_cfg.get("fm_num_target_vision_tokens", 32)),
        fm_add_pos_embed=bool(vla_cfg.get("fm_add_pos_embed", True)),
        fm_max_seq_len=int(vla_cfg.get("fm_max_seq_len", 1024)),
        fm_state_dropout=float(vla_cfg.get("fm_state_dropout", 0.5)),
        lambda_visual_token_cosine=float(vla_cfg.get("lambda_visual_token_cosine", 0.5)),
        d_jepa=vision_backbone.embed_dim,
    )
    return model
