"""RLDS camera and state mappings for the four public LIBERO suites."""

LIBERO_DATASETS = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)

_LIBERO_CONFIG = {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["EEF_state", "gripper_state"],
}

OXE_DATASET_CONFIGS = {name: dict(_LIBERO_CONFIG) for name in LIBERO_DATASETS}
