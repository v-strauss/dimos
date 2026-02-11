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

"""Livox-specific LiDAR module with IMU support."""

from dataclasses import dataclass

from dimos.core import Out, rpc
from dimos.hardware.sensors.lidar.module import LidarModule, LidarModuleConfig
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.spec import perception


@dataclass
class LivoxLidarModuleConfig(LidarModuleConfig):
    enable_imu: bool = True


class LivoxLidarModule(LidarModule, perception.IMU):
    """Livox LiDAR module — pointcloud + IMU.

    Extends LidarModule with IMU stream support for sensors like the Mid-360.
    """

    imu: Out[Imu]

    config: LivoxLidarModuleConfig  # type: ignore[assignment]
    default_config = LivoxLidarModuleConfig  # type: ignore[assignment]

    @rpc
    def start(self) -> None:
        super().start()

        if self.config.enable_imu:
            imu_stream = self.hardware.imu_stream()
            if imu_stream is not None:
                self._disposables.add(
                    imu_stream.subscribe(lambda imu_msg: self.imu.publish(imu_msg)),
                )


livox_lidar_module = LivoxLidarModule.blueprint

__all__ = [
    "LivoxLidarModule",
    "LivoxLidarModuleConfig",
    "livox_lidar_module",
]
