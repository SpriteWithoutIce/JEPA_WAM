"""Evaluate a local JEPA-WAM checkpoint on LIBERO-Plus."""

import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import draccus
import numpy as np
import tqdm

LIBERO_PATH = os.environ.get("LIBERO_PATH")
if not LIBERO_PATH:
    raise RuntimeError("Set LIBERO_PATH to a LIBERO-Plus checkout before running evaluation.")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import _is_native_prismatic_checkpoint_path
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL.value: 220,
    TaskSuite.LIBERO_OBJECT.value: 280,
    TaskSuite.LIBERO_GOAL.value: 300,
    TaskSuite.LIBERO_10.value: 520,
    TaskSuite.LIBERO_90.value: 400,
}

LIBERO_PLUS_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)

CATEGORY_ALIASES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "sensor": "Sensor Noise",
    "noise": "Sensor Noise",
    "objects": "Objects Layout",
    "layout": "Objects Layout",
}

SENSOR_NOISE_ENV_RESOLUTION = 224


@dataclass
class GenerateConfig:
    pretrained_checkpoint: Union[str, Path] = ""
    base_vlm: Union[str, Path] = ""
    llm_checkpoint_path: Union[str, Path] = ""
    vjepa_checkpoint_path: Union[str, Path] = ""
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    libero_plus_categories: str = "all"
    num_trials_per_task: int = 1
    max_tasks: Optional[int] = None
    max_episode_steps: Optional[int] = None
    num_steps_wait: int = 10
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    env_img_res: int = 384
    unnorm_key: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    save_rollouts: bool = True
    seed: int = 7


@dataclass
class EvalStats:
    total_episodes: int = 0
    total_successes: int = 0
    category_totals: Dict[str, int] = field(default_factory=dict)
    category_successes: Dict[str, int] = field(default_factory=dict)
    difficulty_totals: Dict[int, int] = field(default_factory=dict)
    difficulty_successes: Dict[int, int] = field(default_factory=dict)


def _classification_path() -> Path:
    return Path(LIBERO_PATH) / "libero" / "libero" / "benchmark" / "task_classification.json"


def _load_task_classification(task_suite_name: str) -> Dict[str, dict]:
    path = _classification_path()
    if not path.is_file():
        raise ValueError(f"LIBERO-Plus task classification not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        classification = json.load(handle)
    entries = classification.get(task_suite_name)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No LIBERO-Plus classification entries found for `{task_suite_name}` in {path}")

    return {str(entry["name"]): entry for entry in entries if entry.get("name")}


def _resolve_categories(spec: str) -> Optional[set[str]]:
    normalized = str(spec or "all").strip()
    if normalized.lower() in {"all", "*"}:
        return None

    selected = set()
    for raw_category in normalized.split(","):
        category = raw_category.strip()
        if not category:
            continue
        canonical = CATEGORY_ALIASES.get(category.lower(), category)
        if canonical not in LIBERO_PLUS_CATEGORIES:
            raise ValueError(
                f"Unsupported LIBERO-Plus category `{category}`. "
                f"Choose from {LIBERO_PLUS_CATEGORIES} or use `all`."
            )
        selected.add(canonical)
    return selected or None


def _validate_config(cfg: GenerateConfig) -> None:
    if not _is_native_prismatic_checkpoint_path(cfg.pretrained_checkpoint):
        raise ValueError("pretrained_checkpoint must be a local `runs/.../checkpoints/*.pt` file.")
    for name in ("base_vlm", "llm_checkpoint_path", "vjepa_checkpoint_path"):
        path = Path(os.path.expanduser(str(getattr(cfg, name))))
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    if cfg.task_suite_name not in TASK_MAX_STEPS:
        raise ValueError(f"Unsupported suite `{cfg.task_suite_name}`; choose from {sorted(TASK_MAX_STEPS)}.")
    if cfg.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive.")
    if cfg.max_tasks is not None and cfg.max_tasks <= 0:
        raise ValueError("max_tasks must be positive when provided.")
    if cfg.max_episode_steps is not None and cfg.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive when provided.")
    if cfg.num_open_loop_steps <= 0:
        raise ValueError("num_open_loop_steps must be positive.")


def _resolve_unnorm_key(cfg: GenerateConfig, model) -> None:
    candidates = [cfg.unnorm_key, cfg.task_suite_name, f"{cfg.task_suite_name}_no_noops"]
    cfg.unnorm_key = next((key for key in candidates if key and key in model.norm_stats), None)
    if cfg.unnorm_key is None:
        raise ValueError(f"No matching unnorm key in checkpoint statistics: {sorted(model.norm_stats)}")


def _prepare_observation(obs) -> dict:
    proprio = np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    )
    return {
        "full_image": get_libero_image(obs),
        "wrist_image": get_libero_wrist_image(obs),
        "state": proprio,
    }


def _process_action(action: np.ndarray) -> np.ndarray:
    return invert_gripper_action(normalize_gripper_action(action, binarize=True))


def _run_episode(cfg: GenerateConfig, env, task_description: str, model, initial_state):
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    replay_images = []

    episode_steps = cfg.max_episode_steps or TASK_MAX_STEPS[cfg.task_suite_name]
    for step in range(episode_steps + cfg.num_steps_wait):
        if step < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                return True, replay_images
            continue

        replay_images.append(get_libero_image(obs))
        if not action_queue:
            action_queue.extend(get_action(cfg, model, _prepare_observation(obs), task_description))
        action = _process_action(np.asarray(action_queue.popleft()))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, replay_images

    return False, replay_images


def _record_episode(stats: EvalStats, success: bool, category: Optional[str], difficulty: Optional[int]) -> None:
    stats.total_episodes += 1
    stats.total_successes += int(success)
    if category is not None:
        stats.category_totals[category] = stats.category_totals.get(category, 0) + 1
        stats.category_successes[category] = stats.category_successes.get(category, 0) + int(success)
    if difficulty is not None:
        stats.difficulty_totals[difficulty] = stats.difficulty_totals.get(difficulty, 0) + 1
        stats.difficulty_successes[difficulty] = stats.difficulty_successes.get(difficulty, 0) + int(success)


def _log(message: str, log_file) -> None:
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


def _log_summary(stats: EvalStats, log_file) -> float:
    success_rate = stats.total_successes / stats.total_episodes if stats.total_episodes else 0.0
    _log(f"Overall: {stats.total_successes}/{stats.total_episodes} = {success_rate:.4f}", log_file)

    for category in LIBERO_PLUS_CATEGORIES:
        total = stats.category_totals.get(category, 0)
        if total:
            successes = stats.category_successes.get(category, 0)
            _log(f"{category}: {successes}/{total} = {successes / total:.4f}", log_file)
    for difficulty in sorted(stats.difficulty_totals):
        total = stats.difficulty_totals[difficulty]
        successes = stats.difficulty_successes.get(difficulty, 0)
        _log(f"Difficulty {difficulty}: {successes}/{total} = {successes / total:.4f}", log_file)
    return success_rate


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    _validate_config(cfg)
    selected_categories = _resolve_categories(cfg.libero_plus_categories)
    task_metadata = _load_task_classification(cfg.task_suite_name)
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    _resolve_unnorm_key(cfg, model)

    log_dir = Path(cfg.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"libero-plus-{cfg.task_suite_name}-{DATE_TIME}.txt"
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    stats = EvalStats()
    evaluated_tasks = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        _log(f"Suite: {cfg.task_suite_name}", log_file)
        _log(f"Categories: {cfg.libero_plus_categories}", log_file)
        for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc=cfg.task_suite_name):
            task = task_suite.get_task(task_id)
            metadata = task_metadata.get(task.name, {})
            category = metadata.get("category")
            difficulty_value = metadata.get("difficulty_level")
            difficulty = int(difficulty_value) if difficulty_value is not None else None
            if selected_categories is not None and category not in selected_categories:
                continue
            if cfg.max_tasks is not None and evaluated_tasks >= cfg.max_tasks:
                break

            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) < cfg.num_trials_per_task:
                raise ValueError(
                    f"Task `{task.name}` only has {len(initial_states)} initial states, "
                    f"but {cfg.num_trials_per_task} trials were requested."
                )

            env_resolution = SENSOR_NOISE_ENV_RESOLUTION if category == "Sensor Noise" else cfg.env_img_res
            env, task_description = get_libero_env(task, resolution=env_resolution)
            task_successes = 0
            try:
                for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), leave=False):
                    success, replay_images = _run_episode(
                        cfg,
                        env,
                        task_description,
                        model,
                        initial_states[episode_idx],
                    )
                    task_successes += int(success)
                    _record_episode(stats, success, category, difficulty)
                    if cfg.save_rollouts:
                        save_rollout_video(
                            replay_images,
                            stats.total_episodes,
                            success,
                            task_description,
                            log_file=log_file,
                            task_suite_name=cfg.task_suite_name,
                        )
            finally:
                env.close()

            _log(
                f"{task_id:03d} [{category or 'Unclassified'}] {task_description}: "
                f"{task_successes}/{cfg.num_trials_per_task}",
                log_file,
            )
            evaluated_tasks += 1

        if stats.total_episodes == 0:
            raise ValueError("No LIBERO-Plus tasks matched the requested category filter.")
        return _log_summary(stats, log_file)


if __name__ == "__main__":
    eval_libero()
