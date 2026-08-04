#!/usr/bin/python3
"""
run_fake_cobot_client.py

Lightweight client that mimics the cobot inference loop without requiring ROS.
It fabricates proprioception and image observations with the same shapes as the
real robot logs (e.g., qpos (14,), camera frames 480x640x3) and streams them to
the remote OpenVLA server over MsgPack HTTP.
"""

import argparse
import logging
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
np.set_printoptions(precision=8, suppress=True)

from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import (
    DATE_TIME,
    MsgPackHttpClientPolicy,
    get_image_resize_size,
    set_seed_everywhere,
)

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class FakeOpenVLAConfig:
    # fmt: off
    #################################################################################################################
    # Server parameters
    #################################################################################################################
    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = "0.0.0.0"  # Remote VLA server URL
    unnorm_key: str = ""                             # Dataset key for action un-normalization

    #################################################################################################################
    # Model parameters
    #################################################################################################################
    model_family: str = "openvla"
    center_crop: bool = True
    num_open_loop_steps: int = 25

    #################################################################################################################
    # Fake stream parameters
    #################################################################################################################
    sequence_length: int = 260                       # Matches collected episode length
    num_joints: int = 14                             # qpos dimensionality per frame
    image_height: int = 480
    image_width: int = 640
    max_publish_step: int = 1000                     # Safety upper bound for loop

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    seed: int = 7
    task_label: str = "open the box"
    # fmt: on


class FakeRobotStream:
    """Generates deterministic fake proprio and image observations."""

    def __init__(self, cfg: FakeOpenVLAConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.step = 0

    def reset(self):
        self.step = 0

    def get_observation(self):
        """
        Returns a dictionary shaped like the real ROS observation:
        - images: cam_high, cam_left_wrist, cam_right_wrist in 480x640x3 uint8
        - qpos: 14D vector (matches dataset episode shape (260, 14))
        """
        if self.step >= self.cfg.sequence_length:
            return None

        t = self.step
        base_angles = np.linspace(-np.pi, np.pi, self.cfg.num_joints)
        qpos = np.sin(base_angles + 0.05 * t).astype(np.float32)
        qpos += 0.01 * self.rng.standard_normal(self.cfg.num_joints).astype(np.float32)

        observation = {
            "images": {
                "cam_high": self._generate_image(t, 0.0),
                "cam_left_wrist": self._generate_image(t, 0.25),
                "cam_right_wrist": self._generate_image(t, 0.5),
            },
            "qpos": qpos,
        }
        self.step += 1
        return observation

    def _generate_image(self, step: int, phase: float) -> np.ndarray:
        """Create a simple gradient pattern with slight temporal variation."""
        height, width = self.cfg.image_height, self.cfg.image_width
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        base = (x + y + 0.02 * step + phase) % 1.0
        img = np.stack(
            [
                base,
                np.roll(base, 1, axis=1),
                np.roll(base, 2, axis=1),
            ],
            axis=-1,
        )
        noise = self.rng.uniform(-0.02, 0.02, size=img.shape)
        img = np.clip(img + noise, 0.0, 1.0)
        return (img * 255).astype(np.uint8)


def setup_logging(cfg: FakeOpenVLAConfig):
    """Set up logging to file."""
    run_id = f"OPENVLA-FAKE-INFERENCE-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")
    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def validate_config(cfg: FakeOpenVLAConfig):
    assert cfg.use_vla_server, "Fake client still requires --use_vla_server for remote inference."
    assert cfg.vla_server_url, "A valid --vla_server_url must be provided."
    assert cfg.unnorm_key, "A valid --unnorm_key matching the remote policy is required."
    assert cfg.sequence_length > 0, "--sequence_length must be positive."
    assert cfg.num_joints > 0, "--num_joints must be positive."


def get_server_endpoint(cfg: FakeOpenVLAConfig):
    """Normalize different URL formats into a base URL for MsgPack client."""
    server_url = str(cfg.vla_server_url).strip()
    if server_url.startswith("http"):
        return server_url.rstrip("/")

    host_and_path = server_url.split("/", 1)[0]
    if ":" in host_and_path:
        host, port = host_and_path.split(":", 1)
    else:
        host, port = host_and_path, "8777"

    ip_address = socket.gethostbyname(host)
    protocol = "https" if any(keyword in host for keyword in ("nat-notebook", "ngrok")) else "http"
    return f"{protocol}://{ip_address}:{port}"


def prepare_observation_for_server(obs_data, task_description, resize_size, unnorm_key):
    """Format fake observation into the payload required by the server."""
    img_front = resize_image_for_policy(obs_data["images"]["cam_high"], resize_size)
    left_wrist = resize_image_for_policy(obs_data["images"]["cam_left_wrist"], resize_size)
    right_wrist = resize_image_for_policy(obs_data["images"]["cam_right_wrist"], resize_size)

    observation = {
        "full_image": img_front,
        "left_wrist_image": left_wrist,
        "right_wrist_image": right_wrist,
        "state": obs_data["qpos"],
        "instruction": task_description,
        "unnorm_key": unnorm_key,
    }
    return observation


def run_inference_loop(cfg: FakeOpenVLAConfig, log_file=None):
    """Main fake inference loop."""
    resize_size = get_image_resize_size(cfg)
    server_endpoint = get_server_endpoint(cfg)
    client = MsgPackHttpClientPolicy(host=server_endpoint)
    log_message(f"Connecting to OpenVLA server at: {client.infer_url}", log_file)
    log_message(f"Task: {cfg.task_label}", log_file)

    stream = FakeRobotStream(cfg)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    total_model_query_time = 0.0
    t = 0

    try:
        while t < min(cfg.max_publish_step, cfg.sequence_length):
            obs_data = stream.get_observation()
            if obs_data is None:
                log_message("Fake stream exhausted. Stopping episode.", log_file)
                break

            if len(action_queue) == 0:
                observation = prepare_observation_for_server(
                    obs_data,
                    cfg.task_label,
                    resize_size,
                    cfg.unnorm_key,
                )
                log_message("Requerying OpenVLA server...", log_file)
                model_query_start_time = time.time()
                try:
                    response = client.infer(observation)
                except Exception as exc:
                    log_message(f"Error querying server: {exc}", log_file)
                    break
                actions = np.array(response["actions"])
                actions = actions[: cfg.num_open_loop_steps]
                total_model_query_time += time.time() - model_query_start_time
                action_queue.extend(actions)
                log_message(f"Received {len(actions)} actions from server", log_file)

            action = action_queue.popleft()
            log_message(f"Step {t}: action={action}", log_file)
            t += 1

        log_message("\nEpisode completed:", log_file)
        log_message(f"Total steps: {t}", log_file)
        log_message(f"Total model query time: {total_model_query_time:.2f} sec", log_file)
    except KeyboardInterrupt:
        log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)


def main():
    parser = argparse.ArgumentParser(description="Fake OpenVLA remote inference client (no ROS).")

    # Server parameters
    parser.add_argument("--use_vla_server", action="store_true", default=True, help="Use VLA server")
    parser.add_argument("--vla_server_url", type=str, default="0.0.0.0", help="VLA server URL")
    parser.add_argument("--unnorm_key", type=str, default="", help="Dataset key for action un-normalization")

    # Model parameters
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family")
    parser.add_argument("--center_crop", action="store_true", help="Center crop images")
    parser.add_argument("--num_open_loop_steps", type=int, default=25, help="Open loop steps")

    # Fake stream parameters
    parser.add_argument("--sequence_length", type=int, default=260, help="Length of fake episode")
    parser.add_argument("--num_joints", type=int, default=14, help="Dimensionality of qpos vector")
    parser.add_argument("--image_height", type=int, default=480, help="Source image height")
    parser.add_argument("--image_width", type=int, default=640, help="Source image width")
    parser.add_argument("--max_publish_step", type=int, default=1000, help="Maximum steps to execute")

    # Utils
    parser.add_argument("--run_id_note", type=str, help="Run ID note")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs", help="Log directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--task_label", type=str, default="open the box", help="Task description")

    args = parser.parse_args()

    cfg = FakeOpenVLAConfig(
        use_vla_server=args.use_vla_server,
        vla_server_url=args.vla_server_url,
        unnorm_key=args.unnorm_key,
        model_family=args.model_family,
        center_crop=args.center_crop,
        num_open_loop_steps=args.num_open_loop_steps,
        sequence_length=args.sequence_length,
        num_joints=args.num_joints,
        image_height=args.image_height,
        image_width=args.image_width,
        max_publish_step=args.max_publish_step,
        run_id_note=args.run_id_note,
        local_log_dir=args.local_log_dir,
        seed=args.seed,
        task_label=args.task_label,
    )

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    log_file, _, _ = setup_logging(cfg)
    try:
        run_inference_loop(cfg, log_file)
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
