"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""


from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import numpy as np
import torch
import tensorflow as tf
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.future_utils import downsample_future_sequence
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def _stack_pixel_values(values):
    example = values[0]
    if isinstance(example, torch.Tensor):
        return torch.stack(values)
    if isinstance(example, dict):
        return {key: _stack_pixel_values([value[key] for value in values]) for key in example}
    raise ValueError(f"Unsupported pixel value type `{type(example)}`")


def _future_target_pixel_values(image_transform, images):
    values = [image_transform(img) for img in images]
    if isinstance(values[0], dict):
        return {"vjepa": _stack_pixel_values([value["vjepa"] for value in values])}
    return _stack_pixel_values(values)


def _frame_to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, (bytes, bytearray, np.bytes_)):
        return Image.open(BytesIO(frame)).convert("RGB")
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    raise ValueError(f"Unsupported frame type for PIL conversion: {type(frame)}")


def _stitch_images_horizontally(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _compose_robotwin_aloha_mosaic(
    primary: Image.Image,
    left_wrist: Image.Image,
    right_wrist: Image.Image,
) -> Image.Image:
    """Compose the three RoboTwin camera views into one 256x256 image."""
    primary = primary.convert("RGB").resize((256, 128), resample=Image.Resampling.BICUBIC)
    left_wrist = left_wrist.convert("RGB").resize((128, 128), resample=Image.Resampling.BICUBIC)
    right_wrist = right_wrist.convert("RGB").resize((128, 128), resample=Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (256, 256))
    canvas.paste(primary, (0, 0))
    canvas.paste(left_wrist, (0, 128))
    canvas.paste(right_wrist, (128, 128))
    return canvas


def _flatten_fast_token_ids(token_ids: Any) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(x) for x in token_ids.reshape(-1).tolist()]
    if isinstance(token_ids, np.ndarray):
        return [int(x) for x in token_ids.reshape(-1).tolist()]
    if isinstance(token_ids, (list, tuple)):
        flat: list[int] = []
        for item in token_ids:
            flat.extend(_flatten_fast_token_ids(item))
        return flat
    return [int(token_ids)]


def _fast_tokens_to_extra_string(token_ids: list[int]) -> str:
    return "".join(f"<|extra_{int(token_id)}|>" for token_id in token_ids)


def _extract_fast_token_ids(fast_tokenizer: Any, actions: np.ndarray) -> list[int]:
    token_output = fast_tokenizer(actions)
    if hasattr(token_output, "input_ids"):
        token_output = token_output.input_ids
    elif isinstance(token_output, dict) and "input_ids" in token_output:
        token_output = token_output["input_ids"]
    if isinstance(token_output, (list, tuple)) and len(token_output) == 1:
        token_output = token_output[0]
    return _flatten_fast_token_ids(token_output)


@dataclass
class VLABatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False
    action_head_type: str = "flow_gr00t"
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    use_llm_ce_loss: bool = False
    future_obs_window_size: int = 8
    future_obs_downsample_stride: int = 1
    context: bool = False
    visual_token_pair_offset: int = 0
    stitch_primary_and_wrist_images: bool = False
    robotwin_aloha_mosaic: bool = False
    fast_tokenizer: Optional[Any] = None

    @staticmethod
    def _find_wrist_keys(observation: Dict[str, Any]) -> list[str]:
        return sorted(k for k in observation.keys() if k.startswith("image_") and "wrist" in k)

    @staticmethod
    def _find_pair_wrist_keys(observation: Dict[str, Any]) -> list[str]:
        return sorted(k for k in observation.keys() if k.startswith("pair_image_") and "wrist" in k)

    def _get_stitched_current_image(self, rlds_batch: Dict[str, Any], primary_img: Image.Image) -> Image.Image:
        wrist_keys = self._find_wrist_keys(rlds_batch["observation"])
        if not wrist_keys:
            raise ValueError("Expected a wrist image for stitched primary+wrist input, but none were found.")
        wrist_img = _frame_to_pil(rlds_batch["observation"][wrist_keys[0]][0])
        return _stitch_images_horizontally(primary_img, wrist_img)

    def _get_stitched_pair_images(self, rlds_batch: Dict[str, Any], pair_primary_imgs: list[Image.Image]) -> list[Image.Image]:
        pair_wrist_keys = self._find_pair_wrist_keys(rlds_batch["observation"])
        if not pair_wrist_keys:
            raise ValueError("Expected paired wrist images for stitched visual-token cosine supervision, but none were found.")

        pair_wrist_imgs = [_frame_to_pil(frame) for frame in rlds_batch["observation"][pair_wrist_keys[0]]]
        if len(pair_primary_imgs) != len(pair_wrist_imgs):
            raise ValueError(
                "Primary/wrist paired-frame count mismatch for stitched cosine supervision: "
                f"{len(pair_primary_imgs)} vs {len(pair_wrist_imgs)}"
            )
        return [_stitch_images_horizontally(primary, wrist) for primary, wrist in zip(pair_primary_imgs, pair_wrist_imgs)]

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the JEPA-WAM collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        action_valid_mask = rlds_batch.get("action_valid_mask")
        
        # Observation contains [current_frame, future_frame_1, ...] when future_obs_window_size > 0.
        obs = rlds_batch["observation"]
        img = _frame_to_pil(obs["image_primary"][0])
        raw_future_imgs = (
            [_frame_to_pil(obs["image_primary"][i]) for i in range(1, len(obs["image_primary"]))]
            if self.future_obs_window_size > 0
            else []
        )
        future_imgs = downsample_future_sequence(raw_future_imgs, self.future_obs_downsample_stride)
        if self.future_obs_window_size > 0 and not getattr(self, "_printed_future_downsample_debug", False):
            print(
                "[VLABatchTransform] future supervision: "
                f"raw_future_frames={len(raw_future_imgs)} "
                f"downsample_stride={self.future_obs_downsample_stride} "
                f"downsampled_future_frames={len(future_imgs)}"
            )
            self._printed_future_downsample_debug = True
        
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        prompt_builder = self.prompt_builder_fn("openvla")
        if self.use_minivlm:
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn("openvla")

        user_message = f"What action should the robot take to {lang}?"
        prefix_builder = self.prompt_builder_fn("openvla")
        prefix_builder.add_turn("human", user_message)
        prefix_prompt = prefix_builder.get_prompt()
        prefix_input_ids = self.base_tokenizer(prefix_prompt, add_special_tokens=True).input_ids

        if self.use_llm_ce_loss:
            if self.fast_tokenizer is None:
                raise RuntimeError("FAST tokenizer must be initialized when `use_llm_ce_loss=True`.")
            fast_token_ids = _extract_fast_token_ids(self.fast_tokenizer, np.asarray(actions, dtype=np.float32))
            assistant_target = _fast_tokens_to_extra_string(fast_token_ids)
            prompt_builder.add_turn("human", user_message)
            prompt_builder.add_turn("gpt", assistant_target)
            prompt = prompt_builder.get_prompt()
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            labels = list(input_ids)
            labels[: len(prefix_input_ids)] = [IGNORE_INDEX] * len(prefix_input_ids)
        else:
            prompt_builder.add_turn("human", user_message)
            prompt_builder.add_turn("gpt", "")
            prompt = prompt_builder.get_prompt()
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            if self.action_head_type.lower() == "l1":
                input_ids = input_ids + [ACTION_TOKEN_BEGIN_IDX] * NUM_TOKENS
            elif self.action_head_type.lower() in {"flow_gr00t", "flow_gr00t_jepa"}:
                input_ids = input_ids + [ACTION_TOKEN_BEGIN_IDX] * self.flow_gr00t_placeholder_tokens
            labels = [IGNORE_INDEX] * len(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        routed_multiview_transform = (
            self.use_wrist_image
            and hasattr(self.image_transform, "transform_primary")
            and hasattr(self.image_transform, "transform_wrist")
        )
        if self.robotwin_aloha_mosaic:
            wrist_keys = self._find_wrist_keys(obs)
            if len(wrist_keys) != 2:
                raise ValueError(
                    "RoboTwin ALOHA mosaic requires exactly two wrist cameras, "
                    f"found {wrist_keys}."
                )
            left_wrist = _frame_to_pil(obs[wrist_keys[0]][0])
            right_wrist = _frame_to_pil(obs[wrist_keys[1]][0])
            mosaic = _compose_robotwin_aloha_mosaic(img, left_wrist, right_wrist)
            pixel_values = self.image_transform(mosaic)
        elif self.stitch_primary_and_wrist_images:
            pixel_values = self.image_transform(self._get_stitched_current_image(rlds_batch, img))
        elif routed_multiview_transform:
            pixel_values = self.image_transform.transform_primary(img)
        else:
            pixel_values = self.image_transform(img)

        # [CRITICAL] The default JEPA-WAM path uses the LLM as a context encoder
        # only. Fast-token CE supervision opt-in keeps assistant labels intact.
        if not self.use_llm_ce_loss:
            labels[:] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
        )
        if self.context:
            return_dict["context_proprio"] = rlds_batch["context_proprio"]
            return_dict["context_actions"] = rlds_batch["context_actions"]
            return_dict["context_delta_t"] = rlds_batch["context_delta_t"]
        if action_valid_mask is not None:
            return_dict["action_valid_mask"] = action_valid_mask
        if self.future_obs_window_size > 0 and len(future_imgs) > 0:
            return_dict["future_pixel_values"] = _future_target_pixel_values(self.image_transform, future_imgs)

        if self.visual_token_pair_offset > 0 and "pair_image_primary" in obs:
            pair_primary_imgs = [_frame_to_pil(frame) for frame in obs["pair_image_primary"]]
            if self.robotwin_aloha_mosaic:
                pair_wrist_keys = self._find_pair_wrist_keys(obs)
                if len(pair_wrist_keys) != 2:
                    raise ValueError(
                        "RoboTwin ALOHA mosaic pair target requires exactly two wrist cameras, "
                        f"found {pair_wrist_keys}."
                    )
                pair_left_wrist = [_frame_to_pil(frame) for frame in obs[pair_wrist_keys[0]]]
                pair_right_wrist = [_frame_to_pil(frame) for frame in obs[pair_wrist_keys[1]]]
                pair_counts = (len(pair_primary_imgs), len(pair_left_wrist), len(pair_right_wrist))
                if len(set(pair_counts)) != 1:
                    raise ValueError(
                        "Primary/left-wrist/right-wrist paired-frame count mismatch for RoboTwin ALOHA mosaic: "
                        f"{pair_counts}."
                    )
                pair_mosaics = [
                    _compose_robotwin_aloha_mosaic(primary, left, right)
                    for primary, left, right in zip(pair_primary_imgs, pair_left_wrist, pair_right_wrist)
                ]
                return_dict["pair_pixel_values"] = _future_target_pixel_values(self.image_transform, pair_mosaics)
            elif self.stitch_primary_and_wrist_images:
                pair_target_imgs = self._get_stitched_pair_images(rlds_batch, pair_primary_imgs)
                return_dict["pair_pixel_values"] = _future_target_pixel_values(
                    self.image_transform, pair_target_imgs
                )
            else:
                return_dict["pair_pixel_values"] = _future_target_pixel_values(self.image_transform, pair_primary_imgs)
            if not getattr(self, "_printed_visual_token_target_debug", False):
                print(
                    "[VLABatchTransform] visual-token cosine target: "
                    f"mode=current+future offset={self.visual_token_pair_offset}"
                )
                self._printed_visual_token_target_debug = True

        # Add additional inputs. Current observations may use wrist/multi-view
        # inputs, but future aux targets stay third-person only.
        if self.robotwin_aloha_mosaic or self.stitch_primary_and_wrist_images:
            pass
        elif self.use_wrist_image and routed_multiview_transform:
            wrist_keys = self._find_wrist_keys(rlds_batch["observation"])
            if not wrist_keys:
                raise ValueError("Expected a wrist image for routed multi-view transform, but none were found.")

            img_wrist = _frame_to_pil(rlds_batch["observation"][wrist_keys[0]][0])
            pixel_values.update(self.image_transform.transform_wrist(img_wrist))
        elif self.use_wrist_image:
            all_wrist_pixels = []
            for k in self._find_wrist_keys(rlds_batch["observation"]):
                img_wrist = _frame_to_pil(rlds_batch["observation"][k][0])
                pixel_values_wrist = self.image_transform(img_wrist)
                all_wrist_pixels.append(pixel_values_wrist)

            if all_wrist_pixels:
                return_dict["pixel_values_wrist"] = _stack_pixel_values(all_wrist_pixels)
            if self.visual_token_pair_offset > 0:
                all_pair_wrist_pixels = []
                for k in rlds_batch["observation"].keys():
                    if k.startswith("pair_image_") and "wrist" in k:
                        pair_imgs = [_frame_to_pil(frame) for frame in rlds_batch["observation"][k]]
                        all_pair_wrist_pixels.append(_future_target_pixel_values(self.image_transform, pair_imgs))
                if all_pair_wrist_pixels:
                    return_dict["pair_pixel_values_wrist"] = _stack_pixel_values(all_pair_wrist_pixels)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict
    
    

# Backward-compatible name for older RLDS-only entry points.
RLDSBatchTransform = VLABatchTransform


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        future_obs_window_size: int = 8,
        future_obs_downsample_stride: int = 1,
        context: bool = False,
        strict_epoch_mode: bool = False,
        shared_dataset_statistics: bool = False,
        rank_shard_dataset_sources: bool = False,
        visual_token_pair_offset: int = 0,
        rotation_representation: str = "axis_angle",
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.strict_epoch_mode = strict_epoch_mode
        frame_transform_threads = int(os.getenv("VLA_RLDS_FRAME_TRANSFORM_THREADS", "16"))
        private_threadpool_size = int(os.getenv("VLA_RLDS_PRIVATE_THREADPOOL_SIZE", "0"))
        max_intra_op_parallelism = int(os.getenv("VLA_RLDS_MAX_INTRA_OP_PARALLELISM", "0"))
        if frame_transform_threads <= 0:
            raise ValueError("VLA_RLDS_FRAME_TRANSFORM_THREADS must be positive.")
        if private_threadpool_size < 0 or max_intra_op_parallelism < 0:
            raise ValueError("RLDS TensorFlow thread limits must be non-negative.")

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
            rotation_representation=rotation_representation,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                future_obs_window_size=future_obs_window_size,      # Future frame extraction for aux target
                pair_target_offset=visual_token_pair_offset,
                context=getattr(batch_transform, "context", False),        # Paired frame target for visual-token cosine loss
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=frame_transform_threads,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            shared_dataset_statistics_key=self.data_mix if shared_dataset_statistics else None,
            rank_shard_dataset_sources=rank_shard_dataset_sources,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        dataset_obj, dataset_length, dataset_statistics = self.make_dataset(rlds_config)
        if private_threadpool_size > 0 or max_intra_op_parallelism > 0:
            options = tf.data.Options()
            if private_threadpool_size > 0:
                options.threading.private_threadpool_size = private_threadpool_size
                options.autotune.cpu_budget = private_threadpool_size
            if max_intra_op_parallelism > 0:
                options.threading.max_intra_op_parallelism = max_intra_op_parallelism
            if isinstance(dataset_obj, list):
                dataset_obj = [dataset.with_options(options) for dataset in dataset_obj]
            else:
                dataset_obj = dataset_obj.with_options(options)
        if int(os.getenv("RANK", "0")) == 0:
            print(
                "[RLDSDataset] TensorFlow threading: "
                f"frame_transforms={frame_transform_threads} "
                f"private_pool={private_threadpool_size} "
                f"max_intra_op={max_intra_op_parallelism}"
            )
        self.dataset = dataset_obj
        self.global_dataset_length = dataset_length
        self.dataset_length = dataset_length
        self.dataset_statistics = dataset_statistics

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(strict_epoch_mode=self.strict_epoch_mode, **rlds_config)

    @staticmethod
    def _rank_world_size() -> Tuple[int, int]:
        env_rank = os.getenv("RANK")
        env_world_size = os.getenv("WORLD_SIZE")
        if env_rank is not None and env_world_size is not None:
            return int(env_rank), int(env_world_size)
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _local_length_from_global(self, global_length: int) -> int:
        rank, world_size = self._rank_world_size()
        return (global_length + world_size - rank - 1) // world_size

    def __iter__(self) -> Dict[str, Any]:
        datasets = self.dataset if isinstance(self.dataset, list) else [self.dataset]
        for dataset in datasets:
            for rlds_batch in dataset.as_numpy_iterator():
                yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        if self.strict_epoch_mode:
            return self._local_length_from_global(self.global_dataset_length)
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
