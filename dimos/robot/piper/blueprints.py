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

"""Piper arm blueprints: teleop, manipulation, and velocity control."""

from pathlib import Path
from typing import Any

from dimos.control.cartesian_jogger import cartesian_jogger
from dimos.control.components import HardwareComponent, HardwareType, make_joints
from dimos.control.coordinator import TaskConfig, control_coordinator
from dimos.core.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.manipulation.manipulation_module import manipulation_module
from dimos.manipulation.planning.spec import RobotModelConfig
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.msgs.sensor_msgs import JointState
from dimos.utils.data import get_data

# =============================================================================
# Piper Configuration
# =============================================================================


def _get_piper_urdf_path() -> Path:
    return get_data("piper_description") / "urdf/piper_description.urdf"


def _get_piper_package_paths() -> dict[str, Path]:
    return {"piper_description": get_data("piper_description")}


def _get_piper_model_path() -> str:
    piper_path = get_data("piper_description")
    return str(piper_path / "mujoco_model" / "piper_no_gripper_description.xml")


PIPER_GRIPPER_COLLISION_EXCLUSIONS: list[tuple[str, str]] = [
    ("gripper_base", "link7"),
    ("gripper_base", "link8"),
    ("link7", "link8"),
    ("link6", "gripper_base"),
]


def _piper(hardware_id: str = "arm", address: str = "can0") -> HardwareComponent:
    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=make_joints(hardware_id, 6),
        adapter_type="piper",
        address=address,
        auto_enable=True,
    )


def _mock_piper(hardware_id: str = "arm") -> HardwareComponent:
    return HardwareComponent(
        hardware_id=hardware_id,
        hardware_type=HardwareType.MANIPULATOR,
        joints=make_joints(hardware_id, 6),
        adapter_type="mock",
    )


def _joint_names(hardware_id: str, dof: int = 6) -> list[str]:
    return [f"{hardware_id}_joint{i + 1}" for i in range(dof)]


def _trajectory_task(name: str, hardware_id: str) -> TaskConfig:
    return TaskConfig(
        name=name,
        type="trajectory",
        joint_names=_joint_names(hardware_id),
        priority=10,
    )


def _velocity_task(name: str, hardware_id: str) -> TaskConfig:
    return TaskConfig(
        name=name,
        type="velocity",
        joint_names=_joint_names(hardware_id),
        priority=10,
    )


def _cartesian_ik_task(name: str, hardware_id: str) -> TaskConfig:
    return TaskConfig(
        name=name,
        type="cartesian_ik",
        joint_names=_joint_names(hardware_id),
        priority=10,
        model_path=_get_piper_model_path(),
        ee_joint_id=6,
    )


# =============================================================================
# Transports
# =============================================================================


def _standard_transports() -> dict[Any, Any]:
    return {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
    }


def _cartesian_transports() -> dict[Any, Any]:
    return {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
        ("cartesian_command", PoseStamped): LCMTransport(
            "/coordinator/cartesian_command", PoseStamped
        ),
    }


def _velocity_transports() -> dict[Any, Any]:
    return {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
        ("joint_command", JointState): LCMTransport("/velocity/joint_command", JointState),
    }


# =============================================================================
# Manipulation Module Config
# =============================================================================


def _piper_robot_config() -> RobotModelConfig:
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    joint_mapping = {f"arm_{j}": j for j in joint_names}

    return RobotModelConfig(
        name="arm",
        urdf_path=_get_piper_urdf_path(),
        base_pose=PoseStamped(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion.from_euler(Vector3(x=0.0, y=0.0, z=0.0)),
        ),
        joint_names=joint_names,
        end_effector_link="gripper_base",
        base_link="base_link",
        package_paths=_get_piper_package_paths(),
        xacro_args={},
        collision_exclusion_pairs=PIPER_GRIPPER_COLLISION_EXCLUSIONS,
        auto_convert_meshes=True,
        max_velocity=1.0,
        max_acceleration=2.0,
        joint_name_mapping=joint_mapping,
        coordinator_task_name="traj_piper",
    )


# =============================================================================
# Blueprints
# =============================================================================

# Teleop: CartesianIK + pygame jogger
_piper_cartesian_coordinator = control_coordinator(
    tick_rate=100.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[_piper("arm", "can0")],
    tasks=[_cartesian_ik_task("cartesian_ik_arm", "arm")],
).transports(_cartesian_transports())

piper_teleop = autoconnect(
    _piper_cartesian_coordinator,
    cartesian_jogger(),
)

# Manipulation: Trajectory + planning module
_piper_trajectory_coordinator = control_coordinator(
    tick_rate=100.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[_piper("arm", "can0")],
    tasks=[_trajectory_task("traj_piper", "arm")],
).transports(_standard_transports())

_piper_planner = manipulation_module(
    robots=[_piper_robot_config()],
    planning_timeout=10.0,
    enable_viz=True,
).transports(_standard_transports())

piper_manipulation = autoconnect(
    _piper_trajectory_coordinator,
    _piper_planner,
)

# Velocity: Streaming velocity control
piper_velocity = control_coordinator(
    tick_rate=100.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[_piper("arm", "can0")],
    tasks=[_velocity_task("velocity_arm", "arm")],
).transports(_velocity_transports())

# =============================================================================
# Mock Blueprints (for testing without hardware)
# =============================================================================

# Mock teleop: CartesianIK + pygame jogger (no hardware)
_piper_cartesian_coordinator_mock = control_coordinator(
    tick_rate=100.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[_mock_piper("arm")],
    tasks=[_cartesian_ik_task("cartesian_ik_arm", "arm")],
).transports(_cartesian_transports())

piper_teleop_mock = autoconnect(
    _piper_cartesian_coordinator_mock,
    cartesian_jogger(),
)

# Mock manipulation: Trajectory + planning module (no hardware)
_piper_trajectory_coordinator_mock = control_coordinator(
    tick_rate=100.0,
    publish_joint_state=True,
    joint_state_frame_id="coordinator",
    hardware=[_mock_piper("arm")],
    tasks=[_trajectory_task("traj_piper", "arm")],
).transports(_standard_transports())

piper_manipulation_mock = autoconnect(
    _piper_trajectory_coordinator_mock,
    _piper_planner,
)

__all__ = [
    "PIPER_GRIPPER_COLLISION_EXCLUSIONS",
    "piper_manipulation",
    "piper_manipulation_mock",
    "piper_teleop",
    "piper_teleop_mock",
    "piper_velocity",
]
