#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Unitree G1 keyboard teleop via ControlCoordinator over DDS (LAN ethernet).

WASD keys → Twist → coordinator twist_command → UnitreeG1TwistAdapter (DDS).

Uses Unitree SDK2 DDS for wired ethernet control of the G1 humanoid robot.

Controls:
    W/S: Forward/backward (linear.x)
    Q/E: Strafe left/right (linear.y)
    A/D: Turn left/right (angular.z)
    Shift: 2x boost
    Ctrl: 0.5x slow
    Space: Emergency stop
    ESC: Quit

Usage:
    dimos --simulation run unitree-g1-dds-keyboard-teleop             # MuJoCo sim
    dimos run unitree-g1-dds-keyboard-teleop                          # real hardware (DDS)
    ROBOT_INTERFACE=enp60s0 dimos run unitree-g1-dds-keyboard-teleop  # specify interface
"""

from __future__ import annotations

from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.robot.unitree.keyboard_teleop import keyboard_teleop

if global_config.simulation:
    from dimos.robot.unitree.g1.connection import g1_connection

    _g1 = g1_connection()
    _teleop = keyboard_teleop()
else:
    import os

    from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
    from dimos.control.coordinator import TaskConfig, control_coordinator
    from dimos.core.transport import LCMTransport
    from dimos.msgs.geometry_msgs import Twist
    from dimos.msgs.sensor_msgs import JointState

    _g1_joints = make_twist_base_joints("g1")

    _g1 = control_coordinator(
        hardware=[
            HardwareComponent(
                hardware_id="g1",
                hardware_type=HardwareType.BASE,
                joints=_g1_joints,
                adapter_type="unitree_g1",
                address=os.getenv("ROBOT_INTERFACE", "enp60s0"),
            ),
        ],
        tasks=[
            TaskConfig(
                name="vel_g1",
                type="velocity",
                joint_names=_g1_joints,
                priority=10,
            ),
        ],
    ).transports(
        {
            ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
            ("twist_command", Twist): LCMTransport("/cmd_vel", Twist),
        }
    )

    _teleop = keyboard_teleop()

unitree_g1_dds_keyboard_teleop = autoconnect(_g1, _teleop)

__all__ = ["unitree_g1_dds_keyboard_teleop"]
