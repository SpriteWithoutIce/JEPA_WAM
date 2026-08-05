"""Materialize the fixed four-suite LIBERO RLDS mixture."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
) -> Dict[str, Any]:
    if dataset_name not in OXE_DATASET_CONFIGS:
        raise ValueError(f"Unsupported public dataset `{dataset_name}`.")

    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    camera_views = ("primary", "wrist")
    missing = set(camera_views) - set(dataset_kwargs["image_obs_keys"])
    if missing:
        raise ValueError(f"Dataset `{dataset_name}` is missing camera views {sorted(missing)}.")

    dataset_kwargs["image_obs_keys"] = {
        key: value for key, value in dataset_kwargs["image_obs_keys"].items() if key in camera_views
    }
    dataset_kwargs.pop("depth_obs_keys")
    dataset_kwargs["language_key"] = "language_instruction"

    dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
    dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]
    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
) -> Tuple[List[Dict[str, Any]], List[float]]:
    dataset_kwargs = [
        make_oxe_dataset_kwargs(
            name,
            data_root_dir,
        )
        for name, _ in mixture_spec
    ]
    return dataset_kwargs, [weight for _, weight in mixture_spec]
