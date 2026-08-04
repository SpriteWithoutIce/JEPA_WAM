#!/usr/bin/python3
"""
inference_openvla_oft.py

A hybrid inference system that combines local ROS data collection with remote OpenVLA server inference.
This allows local robot control while leveraging cloud-based OpenVLA models.
"""

import torch
import numpy as np
import os
import pickle
import argparse
import json
import select
from einops import rearrange
import socket
import sys
import time
import threading
import math
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("/home/agilex/openvla-oft")
sys.path.append(".")

from experiments.robot.aloha.aloha_utils import (
    # get_aloha_env,
    get_aloha_image,
    get_aloha_wrist_images,
    get_next_task_label,
    # save_rollout_video,
)
from experiments.robot.openvla_utils import resize_image_for_policy
from experiments.robot.robot_utils import (
    DATE_TIME,
    MsgPackHttpClientPolicy,
    get_image_resize_size,
    set_seed_everywhere,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class OpenVLAConfig:
    # fmt: off
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                    # Number of actions to execute open-loop before requerying policy
    unnorm_key: str = ""                             # Dataset key for action un-normalization

    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = "0.0.0.0"  # Remote VLA server URL

    #################################################################################################################
    # Robot control parameters
    #################################################################################################################
    max_publish_step: int = 500                      # Max number of steps per episode
    publish_rate: int = 40                           # Control frequency (Hz)
    num_trials: int = 10                             # Number of inference trials to record
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)
    pos_lookahead_step: int = 25                     # Number of steps to look ahead
    
    #################################################################################################################
    # ROS topic parameters
    #################################################################################################################
    img_front_topic: str = '/camera_f/color/image_raw'
    img_left_topic: str = '/camera_l/color/image_raw' 
    img_right_topic: str = '/camera_r/color/image_raw'
    puppet_arm_left_topic: str = '/puppet/joint_left'
    puppet_arm_right_topic: str = '/puppet/joint_right'
    puppet_arm_left_cmd_topic: str = '/master/joint_left'
    puppet_arm_right_cmd_topic: str = '/master/joint_right'
    robot_base_topic: str = '/odom_raw'
    robot_base_cmd_topic: str = '/cmd_vel'
    
    #################################################################################################################
    # Robot base and movement parameters  
    #################################################################################################################
    use_robot_base: bool = False                     # Whether to use robot base movement
    arm_steps_length: list = None                    # Step sizes for each joint
    use_actions_interpolation: bool = False          # Whether to use action interpolation
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    seed: int = 42                                    # Random Seed
    task_label: str = ""                             # Default task label used when prompting operator
    
    # fmt: on
    
    def __post_init__(self):
        if self.arm_steps_length is None:
            self.arm_steps_length = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]


class RosOperator:
    """ROS interface for robot control and data collection."""
    
    def __init__(self, cfg: OpenVLAConfig):
        self.cfg = cfg
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        """Initialize data structures."""
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        """Publish joint commands to both arms."""
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        """Publish base velocity commands."""
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def get_observation(self):
        """Get synchronized observation from all sensors."""
        if (len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or 
            len(self.img_front_deque) == 0 or len(self.puppet_arm_left_deque) == 0 or 
            len(self.puppet_arm_right_deque) == 0):
            return None
            
        if self.cfg.use_robot_base and len(self.robot_base_deque) == 0:
            return None

        # Get the latest timestamp from all sensors
        frame_time = min([
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(), 
            self.img_front_deque[-1].header.stamp.to_sec()
        ])

        # Check if all data is synchronized
        if (self.img_left_deque[-1].header.stamp.to_sec() < frame_time or
            self.img_right_deque[-1].header.stamp.to_sec() < frame_time or
            self.img_front_deque[-1].header.stamp.to_sec() < frame_time or
            self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time or
            self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time):
            return None
            
        if (self.cfg.use_robot_base and 
            self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return None

        # Pop old data and get synchronized frames
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        robot_base = None
        if self.cfg.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        # Construct observation dict
        observation = {
            'images': {
                'cam_high': img_front,
                'cam_left_wrist': img_left, 
                'cam_right_wrist': img_right
            },
            'qpos': np.concatenate((
                np.array(puppet_arm_left.position), 
                np.array(puppet_arm_right.position)
            ), axis=0),
            'qvel': np.concatenate((
                np.array(puppet_arm_left.velocity),
                np.array(puppet_arm_right.velocity) 
            ), axis=0),
            'effort': np.concatenate((
                np.array(puppet_arm_left.effort),
                np.array(puppet_arm_right.effort)
            ), axis=0)
        }
        
        if self.cfg.use_robot_base and robot_base is not None:
            base_vel = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            observation['qpos'] = np.concatenate((observation['qpos'], base_vel), axis=0)
            
        return observation

    # ROS callback functions
    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        """Initialize ROS node and subscribers/publishers."""
        rospy.init_node('openvla_inference', anonymous=True)
        
        # Subscribers
        rospy.Subscriber(self.cfg.img_left_topic, Image, self.img_left_callback, 
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.cfg.img_right_topic, Image, self.img_right_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.cfg.img_front_topic, Image, self.img_front_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.cfg.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.cfg.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback,
                        queue_size=1000, tcp_nodelay=True)
        
        if self.cfg.use_robot_base:
            rospy.Subscriber(self.cfg.robot_base_topic, Odometry, self.robot_base_callback,
                            queue_size=1000, tcp_nodelay=True)
        
        # Publishers
        self.puppet_arm_left_publisher = rospy.Publisher(self.cfg.puppet_arm_left_cmd_topic, 
                                                        JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.cfg.puppet_arm_right_cmd_topic,
                                                         JointState, queue_size=10)
        if self.cfg.use_robot_base:
            self.robot_base_publisher = rospy.Publisher(self.cfg.robot_base_cmd_topic,
                                                       Twist, queue_size=10)


def validate_config(cfg: OpenVLAConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.use_vla_server, (
        "Must use VLA server for remote inference! Please set --use_vla_server=True"
    )
    assert cfg.vla_server_url, "A valid --vla_server_url must be provided for MsgPack remote inference."
    assert cfg.unnorm_key, "A valid --unnorm_key matching the deployed policy's dataset stats is required."


def setup_logging(cfg: OpenVLAConfig):
    """Set up logging to file.""" 
    # Create run ID
    run_id = f"OPENVLA-INFERENCE-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
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


def get_server_endpoint(cfg: OpenVLAConfig):
    """Get the server endpoint for remote inference."""
    server_url = str(cfg.vla_server_url).strip()
    if server_url.startswith("http"):
        return server_url.rstrip("/")

    # Support host[:port] inputs without protocol
    host_and_path = server_url.split("/", 1)[0]
    if ":" in host_and_path:
        host, port = host_and_path.split(":", 1)
    else:
        host, port = host_and_path, "8777"

    ip_address = socket.gethostbyname(host)
    protocol = "https" if any(keyword in host for keyword in ("nat-notebook", "ngrok")) else "http"
    return f"{protocol}://{ip_address}:{port}"


def prepare_observation_for_server(obs_data, task_description, resize_size):
    """Prepare observation for OpenVLA server input."""
    # Support both dict observations (real ROS bridge) and dm_env.TimeStep (ALOHA sim)
    if isinstance(obs_data, dict) and "images" in obs_data:
        img = obs_data["images"]["cam_high"]
        left_wrist_img = obs_data["images"]["cam_left_wrist"]
        right_wrist_img = obs_data["images"]["cam_right_wrist"]
    else:
        img = get_aloha_image(obs_data)  # Main camera image
        left_wrist_img, right_wrist_img = get_aloha_wrist_images(obs_data)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    left_wrist_img_resized = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_img, resize_size)

    # Prepare observations dict for server
    observation = {
        "full_image": img_resized,
        "left_wrist_image": left_wrist_img_resized,
        "right_wrist_image": right_wrist_img_resized,
        "state": obs_data['qpos'],
        "instruction": task_description,
    }

    return observation, img_resized, left_wrist_img_resized, right_wrist_img_resized


def manual_stop_requested() -> bool:
    """Check whether the operator requested to stop the current trial via stdin."""
    if not sys.stdin.isatty():
        return False

    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return False

    if ready:
        raw_input = sys.stdin.readline()
        if raw_input == "":
            return False
        user_signal = raw_input.rstrip("\r\n")
        if user_signal == " " or user_signal.lower() in {"space", "stop", "s"}:
            return True

    return False


def run_inference_loop(cfg: OpenVLAConfig, ros_operator: RosOperator, log_file=None, run_id: Optional[str] = None):
    """Main inference loop supporting multiple trials."""
    resize_size = get_image_resize_size(cfg)
    server_endpoint = get_server_endpoint(cfg)
    client = MsgPackHttpClientPolicy(host=server_endpoint)
    log_message(f"Connecting to OpenVLA server at: {client.infer_url}", log_file)

    previous_task_description = cfg.task_label
    STEP_DURATION_IN_SEC = 1.0 / cfg.publish_rate
    rate = rospy.Rate(cfg.publish_rate)

    # Nominal joint targets used to settle the robot before each attempt
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375,
             -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656,
              -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]

    trial_results = []
    total_successes = 0

    for trial_idx in range(cfg.num_trials):
        log_message(f"\n========== Trial {trial_idx + 1}/{cfg.num_trials} ==========", log_file)
        task_description = get_next_task_label(previous_task_description)
        previous_task_description = task_description
        log_message(f"Task: {task_description}", log_file)
        log_message("Tip: press <Space> then Enter at any time to stop this trial early.", log_file)

        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        t = 0
        curr_state = None

        ros_operator.puppet_arm_publish(left0, right0)
        time.sleep(3)
        log_message("Prepare the scene, and then press Enter to begin...", log_file)
        input()

        episode_start_time = time.time()
        total_model_query_time = 0.0

        try:
            while t < cfg.max_publish_step and not rospy.is_shutdown():
                if manual_stop_requested():
                    log_message("Manual stop requested; ending trial early.", log_file)
                    break
                step_start_time = time.time()

                obs_data = ros_operator.get_observation()
                if obs_data is None:
                    log_message("Waiting for synchronized sensor data...", log_file)
                    rate.sleep()
                    continue

                if len(action_queue) == 0:
                    log_message("Requerying OpenVLA server...", log_file)
                    observation, _, _, _ = prepare_observation_for_server(obs_data, task_description, resize_size)
                    observation["unnorm_key"] = cfg.unnorm_key

                    model_query_start_time = time.time()
                    try:
                        response = client.infer(observation)
                        actions = np.array(response["actions"])
                        actions = actions[: cfg.num_open_loop_steps]
                        total_model_query_time += time.time() - model_query_start_time
                        action_queue.extend(actions)
                        log_message(f"Received {len(actions)} actions from server", log_file)
                    except Exception as e:
                        log_message(f"Error querying server: {e}", log_file)
                        rate.sleep()
                        continue

                action = action_queue.popleft()
                log_message("-----------------------------------------------------", log_file)
                log_message(f"t: {t}", log_file)
                log_message(f"action: {action}", log_file)

                if cfg.use_relative_actions:
                    if curr_state is None:
                        curr_state = obs_data['qpos']
                    rel_action = action
                    target_state = curr_state + rel_action
                    left_action = target_state[:7]
                    right_action = target_state[7:14]
                    curr_state = target_state
                else:
                    left_action = action[:7]
                    right_action = action[7:14]

                ros_operator.puppet_arm_publish(left_action, right_action)

                if cfg.use_robot_base and len(action) > 14:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)

                t += 1

                step_elapsed_time = time.time() - step_start_time
                if step_elapsed_time < STEP_DURATION_IN_SEC:
                    time.sleep(STEP_DURATION_IN_SEC - step_elapsed_time)

        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)
            else:
                log_message(f"\nCaught exception: {e}", log_file)

        episode_end_time = time.time()
        num_queries = max(1, t // cfg.num_open_loop_steps)
        avg_inference_time = total_model_query_time / num_queries

        user_input = input("Success? Enter 'y' or 'n': ")
        success = user_input.strip().lower() == "y"
        if success:
            total_successes += 1

        current_trial_count = len(trial_results) + 1
        success_rate = total_successes / current_trial_count
        trial_stats = {
            "trial_index": trial_idx + 1,
            "task_description": task_description,
            "success": success,
            "total_steps": t,
            "model_query_time": total_model_query_time,
            "episode_duration": episode_end_time - episode_start_time,
            "avg_inference_time": avg_inference_time,
            "cumulative_success_rate": success_rate,
        }
        trial_results.append(trial_stats)

        log_message("\nTrial summary:", log_file)
        log_message(f"Total steps: {t}", log_file)
        log_message(f"Total model query time: {total_model_query_time:.2f} sec", log_file)
        log_message(f"Episode duration: {trial_stats['episode_duration']:.2f} sec", log_file)
        log_message(f"Average inference time: {avg_inference_time:.3f} sec", log_file)
        log_message(f"Success: {success}", log_file)
        log_message(
            f"Current success rate: {total_successes}/{len(trial_results)} ({success_rate * 100:.1f}%)",
            log_file,
        )

        if rospy.is_shutdown():
            break

    if trial_results:
        final_success_rate = total_successes / len(trial_results)
        log_message("\nFinal multi-trial summary:", log_file)
        log_message(f"Total trials run: {len(trial_results)}", log_file)
        log_message(f"Total successes: {total_successes}", log_file)
        log_message(f"Overall success rate: {final_success_rate * 100:.1f}%", log_file)

        results_filename = f"{run_id or f'cobot_{DATE_TIME}'}_results.json"
        results_subdir = os.path.join(cfg.local_log_dir, cfg.unnorm_key or "default")
        os.makedirs(results_subdir, exist_ok=True)
        results_path = os.path.join(results_subdir, results_filename)
        with open(results_path, "w") as f:
            json.dump(trial_results, f, indent=2)
        log_message(f"Saved trial results to {results_path}", log_file)

    return trial_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="OpenVLA remote inference with local ROS control")
    
    # Model parameters
    parser.add_argument('--model_family', type=str, default='openvla', help='Model family')
    parser.add_argument('--center_crop', action='store_true', help='Center crop images')
    parser.add_argument('--num_open_loop_steps', type=int, default=25, help='Open loop steps')
    
    # Server parameters  
    parser.add_argument('--use_vla_server', action='store_true', default=True, help='Use VLA server')
    parser.add_argument('--vla_server_url', type=str, default='0.0.0.0', help='VLA server URL')
    parser.add_argument('--unnorm_key', type=str, default='', help='Dataset key for action un-normalization')
    
    # Robot control parameters
    parser.add_argument('--max_publish_step', type=int, default=10000, help='Max steps per episode')
    parser.add_argument('--publish_rate', type=int, default=40, help='Control frequency Hz')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of inference trials to record')
    parser.add_argument('--use_relative_actions', action='store_true', help='Use relative actions')
    parser.add_argument('--pos_lookahead_step', type=int, default=25, help='Lookahead steps')
    
    # ROS topics
    parser.add_argument('--img_front_topic', type=str, default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--puppet_arm_left_topic', type=str, default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', type=str, default='/puppet/joint_right')
    parser.add_argument('--puppet_arm_left_cmd_topic', type=str, default='/master/joint_left')
    parser.add_argument('--puppet_arm_right_cmd_topic', type=str, default='/master/joint_right')
    parser.add_argument('--robot_base_topic', type=str, default='/odom_raw')
    parser.add_argument('--robot_base_cmd_topic', type=str, default='/cmd_vel')
    
    # Robot base
    parser.add_argument('--use_robot_base', action='store_true', help='Use robot base')
    parser.add_argument('--use_actions_interpolation', action='store_true', help='Use action interpolation')
    
    # Utils
    parser.add_argument('--run_id_note', type=str, help='Run ID note')
    parser.add_argument('--local_log_dir', type=str, default='./experiments/logs', help='Log directory')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--task_label', type=str, default='', help='Default task label for get_next_task_label')
    
    args = parser.parse_args()
    
    # Create config from args
    cfg = OpenVLAConfig(
        model_family=args.model_family,
        center_crop=args.center_crop,
        num_open_loop_steps=args.num_open_loop_steps,
        use_vla_server=args.use_vla_server,
        vla_server_url=args.vla_server_url,
        unnorm_key=args.unnorm_key,
        max_publish_step=args.max_publish_step,
        publish_rate=args.publish_rate,
        num_trials=args.num_trials,
        use_relative_actions=args.use_relative_actions,
        pos_lookahead_step=args.pos_lookahead_step,
        img_front_topic=args.img_front_topic,
        img_left_topic=args.img_left_topic,
        img_right_topic=args.img_right_topic,
        puppet_arm_left_topic=args.puppet_arm_left_topic,
        puppet_arm_right_topic=args.puppet_arm_right_topic,
        puppet_arm_left_cmd_topic=args.puppet_arm_left_cmd_topic,
        puppet_arm_right_cmd_topic=args.puppet_arm_right_cmd_topic,
        robot_base_topic=args.robot_base_topic,
        robot_base_cmd_topic=args.robot_base_cmd_topic,
        use_robot_base=args.use_robot_base,
        use_actions_interpolation=args.use_actions_interpolation,
        run_id_note=args.run_id_note,
        local_log_dir=args.local_log_dir,
        seed=args.seed,
        task_label=args.task_label,
    )
    
    # Validate config
    validate_config(cfg)
    
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    
    # Initialize ROS operator
    ros_operator = RosOperator(cfg)
    
    log_message("OpenVLA remote inference initialized", log_file)
    log_message(f"Config: {cfg}", log_file)
    
    try:
        # Run inference loop
        run_inference_loop(cfg, ros_operator, log_file, run_id=run_id)
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main() 
