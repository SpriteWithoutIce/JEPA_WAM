"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.rotation_utils import resolve_vla_action_proprio_dims
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS, ActionTokenizer
from prismatic.vla.future_utils import compute_downsampled_future_frame_count

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"


def _normalize_lora_target_modules(target_modules):
    if isinstance(target_modules, tuple):
        return list(target_modules)
    if isinstance(target_modules, list):
        return target_modules
    if isinstance(target_modules, str):
        normalized = target_modules.strip()
        if normalized == "all-linear":
            return normalized
        if "," in normalized:
            return [module.strip() for module in normalized.split(",") if module.strip()]
        return normalized
    return target_modules


def _apply_lora_to_llm_backbone(llm_backbone, vla_cfg: dict, is_trainable: bool) -> None:
    from peft import LoraConfig, get_peft_model

    llm = llm_backbone.llm
    if hasattr(llm, "peft_config"):
        return

    target_modules = _normalize_lora_target_modules(vla_cfg.get("lora_target_modules", "all-linear"))

    llm_backbone.llm = get_peft_model(
        llm,
        LoraConfig(
            r=vla_cfg.get("lora_rank", 32),
            lora_alpha=vla_cfg.get("lora_alpha", 64),
            target_modules=target_modules,
            lora_dropout=vla_cfg.get("lora_dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            inference_mode=not is_trainable,
        ),
    )


def _resolve_vla_local_paths(model_id_or_path: Union[str, Path]) -> Tuple[Optional[Path], Optional[Path], Path, Path]:
    path = Path(model_id_or_path)

    if path.is_file():
        overwatch.info(f"Loading from local checkpoint path `{path}`")
        assert (path.suffix == ".pt") and (path.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = path.parents[1]
        return path, None, run_dir / "config.json", run_dir / "dataset_statistics.json"

    if path.is_dir():
        overwatch.info(f"Loading from local export/checkpoint directory `{path}`")
        config_json = next((candidate for candidate in [path / "config.json", path.parent.parent / "config.json"] if candidate.exists()), None)
        dataset_statistics_json = next(
            (candidate for candidate in [path / "dataset_statistics.json", path.parent.parent / "dataset_statistics.json"] if candidate.exists()),
            None,
        )
        if config_json is None or dataset_statistics_json is None:
            raise ValueError(
                f"Could not find `config.json` and `dataset_statistics.json` for local VLA directory `{path}`"
            )
        return None, path, config_json, dataset_statistics_json

    raise ValueError(f"Local path `{model_id_or_path}` is neither a checkpoint file nor a checkpoint directory.")


def _load_base_vlm_checkpoint_state(base_vlm_run_dir: Union[str, Path]) -> dict:
    run_dir = Path(base_vlm_run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest-checkpoint.pt"
    checkpoint_pt = latest_checkpoint if latest_checkpoint.exists() else None
    if checkpoint_pt is None:
        checkpoint_candidates = sorted(checkpoint_dir.glob("step-*.pt"))
        if not checkpoint_candidates:
            raise ValueError(f"Could not find a base VLM checkpoint under `{checkpoint_dir}`")
        checkpoint_pt = checkpoint_candidates[-1]
    return torch.load(checkpoint_pt, map_location="cpu")["model"]


def _find_component_checkpoint(export_dir: Path, stem: str) -> Optional[Path]:
    direct = export_dir / f"{stem}--checkpoint.pt"
    if direct.exists():
        return direct
    matches = sorted(export_dir.glob(f"{stem}--*.pt"))
    return matches[-1] if matches else None


def _resolve_base_vlm_override(base_vlm_override: Optional[Union[str, Path]]) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    if base_vlm_override is None:
        return None, None, None

    base_vlm_raw = os.path.expanduser(str(base_vlm_override))
    base_vlm_path = Path(base_vlm_raw)

    if base_vlm_path.is_dir():
        config_path = base_vlm_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Base VLM override directory is missing `config.json`: `{base_vlm_path}`")
        with open(config_path, "r") as f:
            base_cfg = json.load(f)["model"]
        return base_cfg["model_id"], str(base_vlm_path), base_cfg

    if base_vlm_path.is_file():
        run_dir = base_vlm_path.parent.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"Base VLM override checkpoint is missing sibling `config.json`: `{base_vlm_path}`")
        with open(config_path, "r") as f:
            base_cfg = json.load(f)["model"]
        return base_cfg["model_id"], str(run_dir), base_cfg

    return str(base_vlm_override), str(base_vlm_override), None


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    image_sequence_len: Optional[int] = None,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""

    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json = run_dir / "config.json"
        checkpoint_dir = run_dir / "checkpoints"
        latest_checkpoint = checkpoint_dir / "latest-checkpoint.pt"
        if latest_checkpoint.exists():
            checkpoint_pt = latest_checkpoint
        else:
            checkpoint_candidates = sorted(checkpoint_dir.glob("step-*.pt"))
            assert checkpoint_candidates, f"Missing checkpoint under `{checkpoint_dir}`"
            checkpoint_pt = checkpoint_candidates[-1]
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    if image_sequence_len is None:
        image_sequence_len = model_cfg.get("image_sequence_len", 1)

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        image_sequence_len,
        checkpoint_path=model_cfg.get("vision_checkpoint_path"),
        dino_local_path=model_cfg.get("dino_local_path"),
        siglip_local_path=model_cfg.get("siglip_local_path"),
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        custom_hf_path=model_cfg.get("llm_local_path"),
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )

    return vlm


# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
    image_sequence_len: Optional[int] = None,
    base_vlm: Optional[Union[str, Path]] = None,
    llm_checkpoint_path: Optional[str] = None,
    vjepa_checkpoint_path: Optional[str] = None,
    dino_local_path: Optional[str] = None,
    siglip_local_path: Optional[str] = None,
    use_aux_head: Optional[bool] = None,
    load_visual_token_cosine_head: bool = True,
) -> OpenVLA:
    """Loads a pretrained OpenVLA from either local disk or the HuggingFace Hub."""
    checkpoint_pt, export_dir = None, None
    local_path = Path(model_id_or_path)
    if local_path.exists():
        checkpoint_pt, export_dir, config_json, dataset_statistics_json = _resolve_vla_local_paths(local_path)
        assert config_json.exists(), f"Missing `config.json` for `{model_id_or_path = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{model_id_or_path = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`VLA_HF_HUB_REPO`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # Identify Checkpoint to Load (via `step_to_load`)
        step_to_load = f"{step_to_load:06d}" if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/step-{step_to_load if step_to_load is not None else ''}*.pt")
        if (len(valid_ckpts) == 0) or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        # Call to `glob` will sort steps in ascending order (if `step_to_load` is None); just grab last element
        target_ckpt = Path(valid_ckpts[-1]).name

        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            relpath = Path(model_type) / model_id_or_path
            config_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=VLA_HF_HUB_REPO, filename=f"{(relpath / 'checkpoints' / target_ckpt)!s}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        full_cfg = json.load(f)
        vla_cfg = full_cfg["vla"]
        if "d_action" not in vla_cfg or "d_proprio" not in vla_cfg:
            resolved_d_action, resolved_d_proprio = resolve_vla_action_proprio_dims(
                vla_cfg.get("data_mix", "libero"),
                vla_cfg.get("rotation_representation", "axis_angle"),
                default_action_dim=vla_cfg.get("d_action", 7),
                default_proprio_dim=vla_cfg.get("d_proprio", 8),
            )
            vla_cfg.setdefault("d_action", resolved_d_action)
            vla_cfg.setdefault("d_proprio", resolved_d_proprio)
        config_base_vlm = vla_cfg["base_vlm"]

    base_cfg = None
    base_vlm_source = None
    base_vlm_source_label = "checkpoint config"
    resolved_base_vlm, resolved_base_vlm_source, resolved_base_cfg = _resolve_base_vlm_override(base_vlm)
    if resolved_base_vlm is not None:
        base_vlm = resolved_base_vlm
        base_vlm_source = resolved_base_vlm_source
        base_cfg = resolved_base_cfg
        vla_cfg["base_vlm"] = str(base_vlm if base_vlm_source is None else base_vlm_source)
        base_vlm_source_label = "eval CLI override"
    else:
        base_vlm = config_base_vlm
        base_vlm_source = str(config_base_vlm) if isinstance(config_base_vlm, (str, Path)) else None
        expanded_base_vlm = os.path.expanduser(str(base_vlm))
        if os.path.isdir(expanded_base_vlm):
            with open(Path(expanded_base_vlm) / "config.json", "r") as f:
                base_cfg = json.load(f)["model"]
                base_vlm = base_cfg["model_id"]
        elif os.path.isfile(expanded_base_vlm):
            with open(Path(expanded_base_vlm).parent.parent / "config.json", "r") as f:
                base_cfg = json.load(f)["model"]
                base_vlm = base_cfg["model_id"]
                base_vlm_source = str(Path(expanded_base_vlm).parent.parent)

    overwatch.info(f"Base vlm ({base_vlm_source_label}): {base_vlm}")
    try:
        model_cfg = ModelConfig.get_choice_class(base_vlm)()
    except KeyError as exc:
        raise ValueError(
            f"Could not resolve base VLM `{base_vlm}`. Pass `--base_vlm` with a valid local base VLM run dir, "
            "checkpoint `.pt`, or a registered model ID."
        ) from exc
    if base_cfg is not None:
        # When the VLA was trained from a local base VLM run, use the exact saved base-model
        # config rather than only the registry defaults keyed by `model_id`.
        for key, value in base_cfg.items():
            if hasattr(model_cfg, key):
                setattr(model_cfg, key, value)

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    if image_sequence_len is None:
        if hasattr(model_cfg, "image_sequence_len"):
            image_sequence_len = model_cfg.image_sequence_len
        else:
            image_sequence_len = 1

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    if vjepa_checkpoint_path:
        vision_ckpt_path = vjepa_checkpoint_path
        vision_ckpt_source = "eval CLI override"
        vla_cfg["vjepa_checkpoint_path"] = vjepa_checkpoint_path
    elif vla_cfg.get("vjepa_checkpoint_path"):
        vision_ckpt_path = vla_cfg.get("vjepa_checkpoint_path")
        vision_ckpt_source = "checkpoint config"
    else:
        vision_ckpt_path = getattr(model_cfg, "vision_checkpoint_path", None)
        vision_ckpt_source = "base VLM config"

    if dino_local_path:
        resolved_dino_local_path = dino_local_path
    else:
        resolved_dino_local_path = vla_cfg.get("dino_local_path") or getattr(model_cfg, "dino_local_path", None)

    if siglip_local_path:
        resolved_siglip_local_path = siglip_local_path
    else:
        resolved_siglip_local_path = vla_cfg.get("siglip_local_path") or getattr(model_cfg, "siglip_local_path", None)

    if model_cfg.vision_backbone_id.startswith("vjepa") or model_cfg.vision_backbone_id.startswith("jepasiglip"):
        if vision_ckpt_path is None:
            raise ValueError(
                "Resuming a V-JEPA based model requires `vla.vjepa_checkpoint_path` in config "
                "or an explicit override in training config."
            )
        overwatch.info(f"Using V-JEPA checkpoint from {vision_ckpt_source}: `{vision_ckpt_path}`")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
        image_sequence_len,
        checkpoint_path=vision_ckpt_path,
        dino_local_path=resolved_dino_local_path,
        siglip_local_path=resolved_siglip_local_path,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    if llm_checkpoint_path:
        resolved_llm_checkpoint_path = llm_checkpoint_path
        llm_checkpoint_source = "eval CLI override"
        full_cfg["llm_checkpoint_path"] = llm_checkpoint_path
    else:
        resolved_llm_checkpoint_path = full_cfg.get("llm_checkpoint_path")
        llm_checkpoint_source = "checkpoint config" if resolved_llm_checkpoint_path else "HF/default config"

    if resolved_llm_checkpoint_path is not None:
        overwatch.info(f"Using LLM checkpoint from {llm_checkpoint_source}: `{resolved_llm_checkpoint_path}`")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
        custom_hf_path=resolved_llm_checkpoint_path,
    )

    # Create Action Tokenizer
    ac_tokenizer = vla_cfg["action_tokenizer"] if "action_tokenizer" in vla_cfg else "action_tokenizer"
    action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS[ac_tokenizer](llm_backbone.get_tokenizer())

    # Reconstruct JEPA-VLA heads with the exact training-time config when available.
    vla_head_kwargs = {}
    for key in (
        "use_action_head",
        "action_head_type",
        "d_action",
        "d_proprio",
        "use_aux_head",
        "d_a",
        "n_heads_action",
        "num_layers_action",
        "ffn_ratio_action",
        "beta_alpha",
        "beta_beta",
        "l1_use_pro_version",
        "l1_num_blocks",
        "fm_hidden_size",
        "fm_action_model_type",
        "fm_num_layers",
        "fm_num_inference_timesteps",
        "fm_num_timestep_buckets",
        "fm_noise_beta_alpha",
        "fm_noise_beta_beta",
        "fm_noise_s",
        "fm_num_target_vision_tokens",
        "fm_add_pos_embed",
        "fm_max_seq_len",
        "fm_state_dropout",
        "fm_jepa_loss_weight",
        "flow_gr00t_placeholder_tokens",
        "flow_gr00t_use_full_llm_hidden",
        "use_llm_ce_loss",
        "lambda_llm_ce",
        "lora_unfreeze_last_n_llm_layers",
        "use_vlm_peft",
        "use_action_queries",
        "d_aux",
        "n_heads_aux",
        "num_layers_aux",
        "ffn_ratio_aux",
        "lambda_aux",
        "use_visual_token_cosine_head",
        "lambda_visual_token_cosine",
    ):
        if key in vla_cfg:
            vla_head_kwargs[key] = vla_cfg[key]
    if use_aux_head is not None:
        vla_head_kwargs["use_aux_head"] = use_aux_head

    # Match JEPA head geometry to the instantiated vision backbone instead of relying on
    # PrismaticVLM defaults (which assume 224px => 14x14 spatial tokens).
    patch_size = getattr(vision_backbone, "patch_size", 16)
    tubelet_size = getattr(vision_backbone, "tubelet_size", 2)
    aux_target_dim = (
        vision_backbone.vjepa_backbone.embed_dim if hasattr(vision_backbone, "vjepa_backbone") else vision_backbone.embed_dim
    )
    aux_spatial_side = vision_backbone.default_image_size // patch_size
    effective_future_frames = compute_downsampled_future_frame_count(
        int(vla_cfg.get("future_obs_window_size", 0)),
        int(vla_cfg.get("future_obs_downsample_stride", 1)),
    )
    aux_temporal_tokens = max(1, effective_future_frames // tubelet_size)
    vla_head_kwargs.setdefault("d_jepa", aux_target_dim)
    vla_head_kwargs.setdefault("fm_jepa_horizon", aux_temporal_tokens)
    vla_head_kwargs.setdefault("aux_T", aux_temporal_tokens)
    vla_head_kwargs.setdefault("aux_H", aux_spatial_side)
    vla_head_kwargs.setdefault("aux_W", aux_spatial_side)

    use_lora = vla_cfg.get("use_lora", False)
    use_vlm_peft = vla_cfg.get("use_vlm_peft", False)

    if checkpoint_pt is not None:
        if use_lora:
            _apply_lora_to_llm_backbone(llm_backbone, vla_cfg, is_trainable=load_for_training)

        overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
        return OpenVLA.from_pretrained(
            checkpoint_pt,
            model_cfg.model_id,
            vision_backbone,
            llm_backbone,
            arch_specifier=model_cfg.arch_specifier,
            freeze_weights=not load_for_training,
            load_visual_token_cosine_head=load_visual_token_cosine_head,
            norm_stats=norm_stats,
            action_tokenizer=action_tokenizer,
            **vla_head_kwargs,
        )

    assert export_dir is not None, "Expected either a native checkpoint file or an export directory."
    if not os.path.isdir(base_vlm_source):
        raise ValueError(
            "Directory-style VLA exports currently require `vla.base_vlm` in config to point to a local base VLM run."
        )

    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Export Directory")
    vla = OpenVLA(
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
        **vla_head_kwargs,
    )

    base_state_dict = _load_base_vlm_checkpoint_state(base_vlm_source)
    vla.projector.load_state_dict(base_state_dict["projector"])
    vla.llm_backbone.load_state_dict(base_state_dict["llm_backbone"])
    if (vision_ckpt := _find_component_checkpoint(export_dir, "vision_backbone")) is not None:
        vla.vision_backbone.load_state_dict(torch.load(vision_ckpt, map_location="cpu"))
    if (projector_ckpt := _find_component_checkpoint(export_dir, "projector")) is not None:
        vla.projector.load_state_dict(torch.load(projector_ckpt, map_location="cpu"))
    if vla.action_queries is not None and (action_queries_ckpt := _find_component_checkpoint(export_dir, "action_queries")) is not None:
        vla.action_queries.load_state_dict(torch.load(action_queries_ckpt, map_location="cpu"))
    if use_vlm_peft:
        from peft import PeftModel

        adapter_dir = export_dir / "lora_adapter"
        if not adapter_dir.exists():
            raise ValueError(f"Expected PEFT adapter directory at `{adapter_dir}`")
        vla = PeftModel.from_pretrained(
            vla,
            adapter_dir,
            is_trainable=load_for_training,
        )
    elif use_lora:
        from peft import PeftModel

        adapter_dir = export_dir / "lora_adapter"
        if not adapter_dir.exists():
            raise ValueError(f"Expected LoRA adapter directory at `{adapter_dir}`")
        vla.llm_backbone.llm = PeftModel.from_pretrained(
            vla.llm_backbone.llm,
            adapter_dir,
            is_trainable=load_for_training,
        )
    elif (llm_ckpt := _find_component_checkpoint(export_dir, "llm_backbone")) is not None:
        vla.llm_backbone.load_state_dict(torch.load(llm_ckpt, map_location="cpu"))

    if vla.action_head is not None and (action_ckpt := _find_component_checkpoint(export_dir, "action_head")) is not None:
        vla.action_head.load_state_dict(torch.load(action_ckpt, map_location="cpu"))
    if vla.aux_head is not None and (aux_ckpt := _find_component_checkpoint(export_dir, "aux_head")) is not None:
        vla.aux_head.load_state_dict(torch.load(aux_ckpt, map_location="cpu"))
    if (
        vla.visual_token_cosine_head is not None
        and (visual_ckpt := _find_component_checkpoint(export_dir, "visual_token_cosine_head")) is not None
    ):
        if load_visual_token_cosine_head:
            missing, unexpected = vla.visual_token_cosine_head.load_state_dict(
                torch.load(visual_ckpt, map_location="cpu"),
                strict=False,
            )
            if missing or unexpected:
                overwatch.info(
                    "Visual Token Cosine Head export mismatch ignored "
                    "(missing=%s unexpected=%s)",
                    missing,
                    unexpected,
                )
        else:
            overwatch.info("Skipping visual_token_cosine_head export load by request.")

    if not load_for_training:
        vla.requires_grad_(False)
        vla.eval()

    return vla
