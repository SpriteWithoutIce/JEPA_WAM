"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import importlib.util
import logging
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import draccus
import numpy as np
import tqdm
import yaml
# Add LIBERO to path before other imports
LIBERO_PATH = os.environ.get("LIBERO_PATH", "/root/linyihan/LIBERO")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)
from libero.libero import benchmark

import wandb
import swanlab

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    _is_native_prismatic_checkpoint_path,
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.util.rotation_utils import (
    AXIS_ANGLE,
    get_libero_eval_action_dim,
    get_libero_eval_proprio_dim,
    quat_to_rot6d_np,
    rot6d_to_axis_angle_np,
    validate_rotation_representation,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL.value: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT.value: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL.value: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10.value: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90.value: 400,  # longest training demo has 373 steps
}

BASE_TASK_SUITE_ORDER = (
    TaskSuite.LIBERO_SPATIAL.value,
    TaskSuite.LIBERO_OBJECT.value,
    TaskSuite.LIBERO_GOAL.value,
    TaskSuite.LIBERO_10.value,
    TaskSuite.LIBERO_90.value,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

LIBERO_PLUS_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)

SENSOR_NOISE_ENV_RESOLUTION = 224
DEFAULT_LIBERO_PRO_PERTURBATION_MAPPING = {
    "use_environment": "env",
    "use_swap": "swap",
    "use_object": "object",
    "use_language": "lan",
    "use_task": "task",
}
LIBERO_PRO_COMBINED_FLAG_ORDER = (
    "use_swap",
    "use_object",
    "use_language",
    "use_task",
    "use_environment",
)
DEFAULT_TASK_MAX_STEPS = 400



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    base_vlm: Optional[str] = None                   # Optional base VLM run dir/checkpoint/model ID override
    llm_checkpoint_path: Optional[str] = None        # Local LLM path (for native .pt checkpoints)
    vjepa_checkpoint_path: Optional[str] = None      # Optional local V-JEPA checkpoint override
    dino_local_path: Optional[str] = None            # Optional local DINO checkpoint/directory override
    siglip_local_path: Optional[str] = None          # Optional local SigLIP checkpoint/directory override
    use_aux_head: Optional[bool] = None              # Override native checkpoint aux-head construction if needed
    load_visual_token_cosine_head: bool = False      # Visual-cosine head is unused at eval; skip loading by default
    action_head_type: str = "l1"                     # Continuous action head type: "l1" or "flow_gr00t"
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    stitch_primary_and_wrist_images: bool = False    # If True, stitch primary+wrist into one policy image
    use_proprio: bool = True                         # Whether to include proprio state in input
    rotation_representation: str = AXIS_ANGLE       # "axis_angle" or "rot6d"
    eval_action_dim: Optional[int] = None            # Optional env action dim after policy prediction (e.g. 7 for LIBERO)
    eval_proprio_dim: Optional[int] = None           # Optional policy proprio dim; pads LIBERO proprio with zeros if needed

    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    evaluation_config_path: Optional[str] = None     # Optional LIBERO-PRO perturbation config
    libero_plus_categories: str = "all"             # "all" or comma-separated LIBERO-plus categories to run

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    log_per_task_metrics: bool = False               # Whether to log per-task metrics keyed by task description

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"
    policy_image_size: Optional[int] = None          # Override policy input image size when known from checkpoint


@dataclass
class EvalStats:
    total_episodes: int = 0
    total_successes: int = 0
    category_totals: Dict[str, int] = field(default_factory=dict)
    category_successes: Dict[str, int] = field(default_factory=dict)
    difficulty_totals: Dict[int, int] = field(default_factory=dict)
    difficulty_successes: Dict[int, int] = field(default_factory=dict)


def _get_task_classification_path() -> Path:
    return Path(LIBERO_PATH) / "libero" / "libero" / "benchmark" / "task_classification.json"


def load_task_classification(task_suite_name: str) -> Dict[str, Dict[str, Union[str, int]]]:
    """Load LIBERO-plus task metadata if available; return empty dict for original LIBERO."""
    classification_path = _get_task_classification_path()
    if not classification_path.exists():
        return {}

    try:
        with open(classification_path, "r") as f:
            classification = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read LIBERO-plus task classification from %s: %s", classification_path, exc)
        return {}

    suite_entries = classification.get(task_suite_name)
    if not isinstance(suite_entries, list):
        logger.info("No LIBERO-plus classification entries found for suite `%s`.", task_suite_name)
        return {}

    task_metadata = {}
    for entry in suite_entries:
        name = entry.get("name")
        if name is None:
            continue
        task_metadata[name] = {
            "category": entry.get("category"),
            "difficulty_level": entry.get("difficulty_level"),
            "id": entry.get("id"),
        }
    return task_metadata


def maybe_log_libero_plus_mode(cfg: GenerateConfig, task_metadata_by_name: Dict[str, Dict[str, Union[str, int]]]) -> None:
    if not task_metadata_by_name:
        return

    logger.info(
        "Detected LIBERO-plus task classification for suite `%s` with %d tasks.",
        cfg.task_suite_name,
        len(task_metadata_by_name),
    )
    if cfg.num_trials_per_task != 1:
        logger.warning(
            "LIBERO-plus is typically evaluated with `num_trials_per_task=1`, but current config uses %d.",
            cfg.num_trials_per_task,
        )


def _success_rate(successes: int, total: int) -> float:
    return float(successes) / float(total) if total > 0 else 0.0


def resolve_selected_libero_plus_categories(categories_spec: str) -> Optional[set[str]]:
    categories_spec = str(categories_spec or "all").strip()
    if categories_spec.lower() in {"all", "*"}:
        return None

    alias_map = {
        "camera": "Camera Viewpoints",
        "camera viewpoints": "Camera Viewpoints",
        "robot": "Robot Initial States",
        "robot initial states": "Robot Initial States",
        "language": "Language Instructions",
        "language instructions": "Language Instructions",
        "light": "Light Conditions",
        "light conditions": "Light Conditions",
        "background": "Background Textures",
        "background textures": "Background Textures",
        "sensor": "Sensor Noise",
        "sensor noise": "Sensor Noise",
        "objects": "Objects Layout",
        "objects layout": "Objects Layout",
    }

    selected = set()
    for raw_part in categories_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        canonical = alias_map.get(part.lower(), part)
        if canonical not in LIBERO_PLUS_CATEGORIES:
            raise ValueError(
                f"Unsupported LIBERO-plus category `{part}`. "
                f"Choose from {LIBERO_PLUS_CATEGORIES} or use `all`."
            )
        selected.add(canonical)

    return selected or None


def log_libero_plus_summary(stats: EvalStats, log_file=None) -> None:
    if not stats.category_totals:
        return

    log_message("LIBERO-plus category summary:", log_file)
    for category in LIBERO_PLUS_CATEGORIES:
        total = stats.category_totals.get(category, 0)
        if total == 0:
            continue
        success_rate = _success_rate(stats.category_successes.get(category, 0), total)
        log_message(f"{category}: {success_rate:.4f} ({success_rate * 100:.1f}%) [{stats.category_successes.get(category, 0)}/{total}]", log_file)

    if stats.difficulty_totals:
        log_message("LIBERO-plus difficulty summary:", log_file)
        for level in sorted(stats.difficulty_totals):
            total = stats.difficulty_totals[level]
            success_rate = _success_rate(stats.difficulty_successes.get(level, 0), total)
            log_message(f"Level-{level}: {success_rate:.4f} ({success_rate * 100:.1f}%) [{stats.difficulty_successes.get(level, 0)}/{total}]", log_file)


def get_task_env_resolution(cfg: GenerateConfig, category: Optional[Union[str, int]]) -> int:
    """Choose simulator render resolution for a task.

    Sensor-noise tasks in LIBERO-plus are authored around 224px image perturbations.
    We therefore render those tasks at 224, let the benchmark add noise there, and
    only then resize to the policy input size in `prepare_observation`.
    """
    if category == "Sensor Noise":
        return SENSOR_NOISE_ENV_RESOLUTION
    return cfg.env_img_res


def infer_base_task_suite_name(task_suite_name: str) -> str:
    """Map a derived benchmark suite back to its base training suite when possible."""
    suite_name = str(task_suite_name).lower()
    for base_suite in BASE_TASK_SUITE_ORDER:
        if suite_name == base_suite or suite_name.startswith(f"{base_suite}_") or suite_name.startswith(f"{base_suite}-"):
            return base_suite
    return suite_name


def resolve_task_max_steps(task_suite_name: str) -> int:
    """Return rollout length for standard and derived LIBERO benchmark suites."""
    suite_name = str(task_suite_name).lower()
    if suite_name in TASK_MAX_STEPS:
        return TASK_MAX_STEPS[suite_name]

    base_suite_name = infer_base_task_suite_name(suite_name)
    if base_suite_name in TASK_MAX_STEPS:
        return TASK_MAX_STEPS[base_suite_name]

    logger.warning(
        "No task max-step entry found for suite `%s`; falling back to %d steps.",
        suite_name,
        DEFAULT_TASK_MAX_STEPS,
    )
    return DEFAULT_TASK_MAX_STEPS


def load_libero_pro_perturbation_module():
    """Load LIBERO-PRO's optional perturbation helper without affecting non-PRO runs."""
    perturbation_path = Path(LIBERO_PATH) / "perturbation.py"
    if not perturbation_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("libero_pro_perturbation", perturbation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load perturbation module from {perturbation_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_libero_pro_suite(cfg: GenerateConfig) -> str:
    """Resolve and generate LIBERO-PRO perturbation suites when requested."""
    if not cfg.evaluation_config_path:
        return str(cfg.task_suite_name).lower()

    evaluation_config_path = Path(os.path.expanduser(str(cfg.evaluation_config_path)))
    assert evaluation_config_path.exists(), f"LIBERO-PRO evaluation config not found: {evaluation_config_path}"

    perturbation = load_libero_pro_perturbation_module()
    assert perturbation is not None, (
        "Received `evaluation_config_path`, but no `perturbation.py` exists under LIBERO_PATH. "
        "Point LIBERO_PATH to the LIBERO-PRO repository root."
    )

    with open(evaluation_config_path, "r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f) or {}

    requested_suite_name = str(cfg.task_suite_name).lower()
    libero_root = Path(LIBERO_PATH)
    evaluation_cfg["bddl_files_path"] = str(libero_root / "libero" / "libero" / "bddl_files")
    evaluation_cfg["script_path"] = str(libero_root / "notebooks" / "generate_init_states.py")
    evaluation_cfg["init_file_dir"] = str(libero_root / "libero" / "libero" / "init_files")
    evaluation_cfg["task_suite_name"] = requested_suite_name
    evaluation_cfg.setdefault("seed", cfg.seed)

    enabled_flags = [
        flag_name
        for flag_name in DEFAULT_LIBERO_PRO_PERTURBATION_MAPPING
        if bool(evaluation_cfg.get(flag_name, False))
    ]

    if not enabled_flags:
        logger.info(
            "LIBERO-PRO config loaded from %s, but no perturbation flags are enabled; evaluating `%s` directly.",
            evaluation_config_path,
            requested_suite_name,
        )
        return requested_suite_name

    if "use_task" in enabled_flags and len(enabled_flags) > 1:
        raise ValueError("LIBERO-PRO `use_task=True` cannot be combined with other perturbations.")

    perturbation_mapping = {
        **DEFAULT_LIBERO_PRO_PERTURBATION_MAPPING,
        **(evaluation_cfg.get("perturbation_mapping", {}) or {}),
    }

    if len(enabled_flags) > 1:
        resolved_suite_name = f"{requested_suite_name}_temp"
        expected_log = ",".join(str(bool(evaluation_cfg.get(flag_name, False))) for flag_name in LIBERO_PRO_COMBINED_FLAG_ORDER)
        generated_bddl_dir = Path(evaluation_cfg["bddl_files_path"]) / resolved_suite_name
        generated_init_dir = Path(evaluation_cfg["init_file_dir"]) / resolved_suite_name
        log_path = generated_bddl_dir / "log.txt"

        should_regenerate = True
        if generated_bddl_dir.exists() and generated_init_dir.exists() and log_path.exists():
            try:
                should_regenerate = log_path.read_text(encoding="utf-8").strip() != expected_log
            except Exception:
                should_regenerate = True

        if should_regenerate:
            shutil.rmtree(generated_bddl_dir, ignore_errors=True)
            shutil.rmtree(generated_init_dir, ignore_errors=True)
            generated_bddl_dir.mkdir(parents=True, exist_ok=True)
            generated_init_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text(expected_log, encoding="utf-8")
            perturbation.create_env(configs=evaluation_cfg)
    else:
        perturb_key = enabled_flags[0]
        perturb_suffix = perturbation_mapping.get(perturb_key)
        assert perturb_suffix, f"Missing LIBERO-PRO perturbation mapping for flag `{perturb_key}`."

        resolved_suite_name = f"{requested_suite_name}_{perturb_suffix}"
        generated_bddl_dir = Path(evaluation_cfg["bddl_files_path"]) / resolved_suite_name
        generated_init_dir = Path(evaluation_cfg["init_file_dir"]) / resolved_suite_name
        if not generated_bddl_dir.exists() or not generated_init_dir.exists():
            perturbation.create_env(configs=evaluation_cfg)

    logger.info(
        "Resolved LIBERO-PRO suite `%s` -> `%s` using flags %s.",
        requested_suite_name,
        resolved_suite_name,
        ", ".join(enabled_flags),
    )
    return resolved_suite_name



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    ckpt_path = os.path.expanduser(str(cfg.pretrained_checkpoint))
    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"
    assert _is_native_prismatic_checkpoint_path(ckpt_path) or os.path.isdir(ckpt_path), (
        "Pass either a native Prismatic checkpoint `.pt`, a native export directory, "
        "or a local HF/OpenVLA checkpoint directory."
    )

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    if cfg.evaluation_config_path is not None:
        evaluation_config_path = os.path.expanduser(str(cfg.evaluation_config_path))
        assert os.path.exists(evaluation_config_path), f"Evaluation config path does not exist: {evaluation_config_path}"


def align_eval_config_with_training_checkpoint(cfg: GenerateConfig) -> None:
    """Align eval-side defaults with the saved native checkpoint config."""
    ckpt_path = os.path.expanduser(str(cfg.pretrained_checkpoint))
    if not _is_native_prismatic_checkpoint_path(ckpt_path):
        return

    ckpt = Path(ckpt_path)
    run_dir = ckpt.parents[1] if ckpt.is_file() else ckpt.parent.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.warning("Native checkpoint config not found at %s; keeping eval CLI settings.", config_path)
        return

    try:
        with open(config_path, "r") as f:
            train_cfg = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read training config from %s: %s", config_path, exc)
        return

    vla_cfg = train_cfg.get("vla", {})
    trained_num_images = int(vla_cfg.get("image_sequence_len", cfg.num_images_in_input))
    trained_use_wrist = bool(vla_cfg.get("use_wrist_image", trained_num_images > 1))
    trained_stitch_primary_wrist = bool(
        vla_cfg.get("stitch_primary_and_wrist_images", cfg.stitch_primary_and_wrist_images)
    )
    trained_rotation_representation = validate_rotation_representation(
        vla_cfg.get("rotation_representation", cfg.rotation_representation)
    )
    trained_policy_size = None

    base_vlm = cfg.base_vlm or vla_cfg.get("base_vlm")
    if isinstance(base_vlm, str) and os.path.isdir(base_vlm):
        base_config_path = Path(base_vlm) / "config.json"
        if base_config_path.exists():
            try:
                with open(base_config_path, "r") as f:
                    base_cfg = json.load(f).get("model", {})
                vision_backbone_id = str(base_cfg.get("vision_backbone_id", ""))
                if "384px" in vision_backbone_id:
                    trained_policy_size = 384
                elif "336px" in vision_backbone_id:
                    trained_policy_size = 336
                elif "256px" in vision_backbone_id:
                    trained_policy_size = 256
                elif "224px" in vision_backbone_id:
                    trained_policy_size = 224
            except Exception as exc:
                logger.warning("Failed to read base VLM config from %s: %s", base_config_path, exc)

    if trained_stitch_primary_wrist:
        trained_num_images = 1

    if cfg.num_images_in_input != trained_num_images:
        logger.warning(
            "Overriding eval num_images_in_input from %s to %s to match saved training config.",
            cfg.num_images_in_input,
            trained_num_images,
        )
        cfg.num_images_in_input = trained_num_images

    cfg.stitch_primary_and_wrist_images = trained_stitch_primary_wrist
    cfg.rotation_representation = trained_rotation_representation

    if trained_policy_size is not None:
        cfg.policy_image_size = trained_policy_size
        if cfg.env_img_res < trained_policy_size:
            logger.warning(
                "Overriding env_img_res from %s to %s to avoid upsampling simulator frames for the policy.",
                cfg.env_img_res,
                trained_policy_size,
            )
            cfg.env_img_res = trained_policy_size

    logger.info(
        "Loaded native training config from %s | image_sequence_len=%s use_wrist_image=%s "
        "stitch_primary_and_wrist_images=%s rotation_representation=%s policy_image_size=%s "
        "action_head_type=%s use_aux_head=%s use_lora=%s",
        config_path,
        trained_num_images,
        trained_use_wrist,
        cfg.stitch_primary_and_wrist_images,
        cfg.rotation_representation,
        trained_policy_size,
        vla_cfg.get("action_head_type"),
        vla_cfg.get("use_aux_head"),
        vla_cfg.get("use_lora"),
    )



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    native_checkpoint = _is_native_prismatic_checkpoint_path(cfg.pretrained_checkpoint)
    cfg.rotation_representation = validate_rotation_representation(cfg.rotation_representation)
    proprio_dim = get_libero_eval_proprio_dim(cfg.rotation_representation)
    action_dim = get_libero_eval_action_dim(cfg.rotation_representation)

    # Load model
    model = get_model(cfg)
    if hasattr(model, "set_version"):
        model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio and not native_checkpoint:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=proprio_dim,
        )

    # Load action head if needed
    action_head = None
    if cfg.action_head_type == "l1" and not native_checkpoint:
        action_head = get_action_head(cfg, model.llm_dim, action_dim=action_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    requested_unnorm_key = str(cfg.unnorm_key or cfg.task_suite_name)
    candidate_keys = [requested_unnorm_key]

    inferred_base_key = infer_base_task_suite_name(requested_unnorm_key)
    if inferred_base_key not in candidate_keys:
        candidate_keys.append(inferred_base_key)

    candidate_keys.extend(
        f"{candidate_key}_no_noops"
        for candidate_key in list(candidate_keys)
        if f"{candidate_key}_no_noops" not in candidate_keys
    )

    unnorm_key = next((candidate_key for candidate_key in candidate_keys if candidate_key in model.norm_stats), None)
    assert unnorm_key is not None, (
        f"Action un-norm key not found in VLA `norm_stats`. Tried: {candidate_keys}"
    )

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        swanlab.init(
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def _right_pad_1d(array: np.ndarray, target_dim: Optional[int], name: str) -> np.ndarray:
    array = np.asarray(array)
    if target_dim is None or array.shape[-1] == target_dim:
        return array
    if array.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad `{name}` with dim {array.shape[-1]} down to target dim {target_dim}.")
    return np.concatenate([array, np.zeros(target_dim - array.shape[-1], dtype=array.dtype)], axis=-1)


def prepare_observation(
    obs,
    resize_size,
    rotation_representation,
    stitch_primary_and_wrist_images=False,
    eval_proprio_dim: Optional[int] = None,
):
    """Prepare observation for policy input."""
    rotation_representation = validate_rotation_representation(rotation_representation)
    # Keep LIBERO benchmark-side preprocessing consistent with the original evaluator.
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by policy evaluation pipeline.
    if stitch_primary_and_wrist_images:
        img_resized = resize_image_for_policy(np.concatenate([img, wrist_img], axis=1), resize_size)
        wrist_img_resized = None
    else:
        img_resized = resize_image_for_policy(img, resize_size)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    proprio = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"])
            if rotation_representation == AXIS_ANGLE
            else quat_to_rot6d_np(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    observation = {
        "full_image": img_resized,
        "state": _right_pad_1d(proprio, eval_proprio_dim, "proprio"),
    }
    if wrist_img_resized is not None:
        observation["wrist_image"] = wrist_img_resized

    return observation, img  # Return both processed observation and original image for replay



def process_action(action, model_family, rotation_representation):
    """Process action before sending to environment."""
    rotation_representation = validate_rotation_representation(rotation_representation)
    if rotation_representation != AXIS_ANGLE:
        action = np.concatenate([action[:3], rot6d_to_axis_angle_np(action[3:9]), action[-1:]], axis=0)

    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def adapt_action_for_env(action: np.ndarray, eval_action_dim: Optional[int]) -> np.ndarray:
    if eval_action_dim is None or action.shape[-1] == eval_action_dim:
        return action
    if action.shape[-1] < eval_action_dim:
        raise ValueError(f"Policy action dim {action.shape[-1]} is smaller than eval_action_dim={eval_action_dim}.")
    return action[..., :eval_action_dim]



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = resolve_task_max_steps(cfg.task_suite_name)

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(
                obs,
                resize_size,
                cfg.rotation_representation,
                cfg.stitch_primary_and_wrist_images,
                cfg.eval_proprio_dim,
            )
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )

                action_queue.extend(actions) 

            # Get action from queue
            action = action_queue.popleft()
            # action = actions[0]

            action = adapt_action_for_env(action, cfg.eval_action_dim)

            # Process action
            action = process_action(action, cfg.model_family, cfg.rotation_representation)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images




def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    stats: Optional[EvalStats] = None,
    task_metadata_by_name: Optional[Dict[str, Dict[str, Union[str, int]]]] = None,
    log_file=None,
    save_version=None
):
    """Run evaluation for a single task."""
    # Get task
    # task_id = 8
    task = task_suite.get_task(task_id)
    task_metadata_by_name = task_metadata_by_name or {}
    task_metadata = task_metadata_by_name.get(task.name, {})
    category = task_metadata.get("category")
    difficulty_level = task_metadata.get("difficulty_level")
    selected_categories = resolve_selected_libero_plus_categories(cfg.libero_plus_categories)
    if selected_categories is not None and category not in selected_categories:
        return stats

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    task_env_res = get_task_env_resolution(cfg, category)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=task_env_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
        if category is not None or difficulty_level is not None:
            log_message(
                f"Task metadata => category: {category or 'N/A'}, difficulty: {difficulty_level or 'N/A'}",
                log_file,
            )
        if task_env_res != cfg.env_img_res:
            log_message(
                f"Task env resolution override => render at {task_env_res}, then resize to policy input {resize_size}.",
                log_file,
            )

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        if stats is not None:
            stats.total_episodes += 1
            if category is not None:
                stats.category_totals[str(category)] = stats.category_totals.get(str(category), 0) + 1
            if difficulty_level is not None:
                level = int(difficulty_level)
                stats.difficulty_totals[level] = stats.difficulty_totals.get(level, 0) + 1
        if success:
            task_successes += 1
            if stats is not None:
                stats.total_successes += 1
                if category is not None:
                    stats.category_successes[str(category)] = stats.category_successes.get(str(category), 0) + 1
                if difficulty_level is not None:
                    level = int(difficulty_level)
                    stats.difficulty_successes[level] = stats.difficulty_successes.get(level, 0) + 1

        total_episodes = stats.total_episodes if stats is not None else task_episodes

        # Save replay video
        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            save_version=save_version,
            task_suite_name=cfg.task_suite_name,
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        total_episodes = stats.total_episodes if stats is not None else task_episodes
        total_successes = stats.total_successes if stats is not None else task_successes
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_episodes = stats.total_episodes if stats is not None else task_episodes
    total_successes = stats.total_successes if stats is not None else task_successes
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    
    # close env
    env.close()
    del env

    # Log task metrics only when explicitly requested; LIBERO-plus task
    # descriptions are all distinct, which otherwise creates one chart per task.
    if cfg.use_wandb and cfg.log_per_task_metrics:
        swanlab.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return stats



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)
    cfg.task_suite_name = str(cfg.task_suite_name).lower()
    align_eval_config_with_training_checkpoint(cfg)
    resolved_task_suite_name = prepare_libero_pro_suite(cfg)
    if resolved_task_suite_name != cfg.task_suite_name:
        logger.info("Resolved benchmark suite from `%s` to `%s`.", cfg.task_suite_name, resolved_task_suite_name)
        cfg.task_suite_name = resolved_task_suite_name

    available_task_suites = benchmark.get_benchmark_dict()
    assert cfg.task_suite_name in available_task_suites, (
        f"Invalid task suite: {cfg.task_suite_name}. "
        f"Available suites include: {sorted(available_task_suites.keys())[:20]}"
    )

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # for name, param in model.named_parameters():
    #     if 'action_queries' in name: 
    #         print(f"{name}: {param}")

    # Keep benchmark-side resize behavior unchanged.
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = available_task_suites
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_metadata_by_name = load_task_classification(cfg.task_suite_name)
    maybe_log_libero_plus_mode(cfg, task_metadata_by_name)
    stats = EvalStats()

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    for task_id in tqdm.tqdm(range(num_tasks)):
        stats = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            stats,
            task_metadata_by_name,
            log_file,
            cfg.save_version
        )

    # Calculate final success rate
    final_success_rate = float(stats.total_successes) / float(stats.total_episodes) if stats.total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {stats.total_episodes}", log_file)
    log_message(f"Total successes: {stats.total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_libero_plus_summary(stats, log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        payload = {
            "success_rate/total": final_success_rate,
            "num_episodes/total": stats.total_episodes,
        }
        for category, total in stats.category_totals.items():
            payload[f"success_rate/category/{category}"] = _success_rate(stats.category_successes.get(category, 0), total)
        for level, total in stats.difficulty_totals.items():
            payload[f"success_rate/difficulty/level_{level}"] = _success_rate(stats.difficulty_successes.get(level, 0), total)
        swanlab.log(payload)
        # wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()
