#!/usr/bin/env python3
"""Populate per-dataset caches and write one shared RoboTwin ALOHA statistic."""

import argparse
import json
from pathlib import Path

import numpy as np

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import merge_dataset_statistics


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-mix", default="aloha_robotwin_all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mixture_spec = OXE_NAMED_MIXTURES[args.data_mix]
    dataset_kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        args.data_root,
        mixture_spec,
        load_camera_views=("primary", "left_wrist", "right_wrist"),
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )

    all_statistics = {}
    for index, kwargs in enumerate(dataset_kwargs, start=1):
        name = kwargs["name"]
        print(f"[STATS {index}/{len(dataset_kwargs)}] {name}", flush=True)
        _, statistics = make_dataset_from_rlds(
            **kwargs,
            train=True,
            num_parallel_calls=1,
            num_parallel_reads=1,
        )
        all_statistics[name] = statistics

    shared = merge_dataset_statistics(all_statistics)
    output = args.output or args.data_root / f"{args.data_mix}_shared_dataset_statistics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump({args.data_mix: shared}, file, indent=2, default=json_default)
    print(f"[STATS] Saved shared statistics to {output}")


if __name__ == "__main__":
    main()
