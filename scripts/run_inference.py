# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time

import numpy as np
from openpi_client import websocket_client_policy
from xarm.wrapper import XArmAPI

ACTION_HORIZON = 15


def franka_to_xarm(franka_joint_positions):
    offsets = np.array([0, 0, 0, 180, 0, 180, 0])
    return offsets - franka_joint_positions


def get_observation():
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": arm.get_servo_angle()[1],  # Your xArm joint positions
        "observation/gripper_position": arm.get_gripper_position(),  # Your gripper position
        "prompt": "move the arm slightly to the left",  # Your task description
    }


def run_inference():
    """
    Run inference loop until user interrupts
    """
    actions_from_chunk_completed = 0
    while True:
        if actions_from_chunk_completed == ACTION_HORIZON:
            actions_from_chunk_completed = 0
            observation = get_observation()
            result = policy.infer(observation)
            action_chunk = result["actions"]  # Shape: (15, 8) - these are VELOCITY COMMANDS
            dt = 1.0 / 15.0
            action_chunk = action_chunk.copy()
            action_chunk[:, :-1] *= dt
            action_chunk[:, :-1] = np.cumsum(
                action_chunk[:, :-1], axis=0
            )  # integrate to get delta position in radians
            current_joint_positions = arm.get_servo_angle()
            action_chunk[:, :-1] += current_joint_positions
            action_chunk[:, :-1] *= 360 / (2 * np.pi)  # convert to degrees
            actions_from_chunk_completed += 1

        actions_from_chunk_completed += 1


if __name__ == "__main__":
    # connect to policy server
    policy = websocket_client_policy.WebsocketClientPolicy(
        host="localhost",  # Docker host gateway (server running on host machine)
        port=8000,  # default port
    )

    # connect to xArm
    arm = XArmAPI("192.168.2.235")
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(1)

    print(f"arm.get_servo_angle(): {arm.get_servo_angle()}")

    arm.move_gohome(wait=True)
    print(f"arm.get_servo_angle(): {arm.get_servo_angle()}")

    # Create a DROID-format observation (DUMMY OBSERVATION)
    observation = {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),  # Your xArm joint positions
        "observation/gripper_position": np.random.rand(1),  # Your gripper position
        "prompt": "move the arm slightly to the left",  # Your task description
    }

    result = policy.infer(observation)
    action_chunk = result["actions"]  # Shape: (15, 8) - these are VELOCITY COMMANDS
    print(action_chunk[2])

    dt = 1.0 / 15.0
    action_chunk = action_chunk.copy()
    action_chunk[:, :-1] *= dt
    action_chunk[:, :-1] = np.cumsum(
        action_chunk[:, :-1], axis=0
    )  # integrate to get delta position in radians
    action_chunk[:, :-1] *= 360 / (2 * np.pi)  # convert to degrees

    gripper_value = action_chunk[:, 7]
    gripper_value = np.where(gripper_value > 0.5, 0.0, gripper_value)
    gripper_xarm = (1.0 - gripper_value) * 850

    # send commands to xArm
    for i in range(len(action_chunk)):
        print(f"Joint positions: {action_chunk[i, :7]} Gripper position: {gripper_xarm[i]}")
        arm.set_servo_angle(angle=action_chunk[i, :7], speed=50, wait=False)
        # arm.set_gripper_position(pos=gripper_xarm[i], speed=50, wait=False)

    arm.disconnect()
