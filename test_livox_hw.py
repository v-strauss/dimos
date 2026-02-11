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

"""Deploy LivoxLidarModule with LCM transport for pointcloud and IMU."""

import time

from dimos.core import In, Module, start
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.lidar.livox.mid360 import LivoxMid360
from dimos.hardware.sensors.lidar.livox.module import LivoxLidarModule
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

pc_count = 0
imu_count = 0


class LidarListener(Module):
    pointcloud: In[PointCloud2]
    imu: In[Imu]

    def start(self):
        super().start()
        self.pointcloud.subscribe(self._on_pc)
        self.imu.subscribe(self._on_imu)

    def _on_pc(self, pc):
        global pc_count
        pc_count += 1
        n = len(pc.pointcloud.points) if hasattr(pc.pointcloud, "points") else "?"
        print(f"  [PC #{pc_count}] {n} points, ts={pc.ts:.3f}", flush=True)

    def _on_imu(self, imu_msg):
        global imu_count
        imu_count += 1
        if imu_count % 200 == 1:
            print(
                f"  [IMU #{imu_count}] acc=({imu_msg.linear_acceleration.x:.2f}, "
                f"{imu_msg.linear_acceleration.y:.2f}, {imu_msg.linear_acceleration.z:.2f})",
                flush=True,
            )


if __name__ == "__main__":
    dimos = start(2)

    lidar = dimos.deploy(
        LivoxLidarModule,
        hardware=lambda: LivoxMid360(host_ip="192.168.1.5", lidar_ips=["192.168.1.155"]),
    )
    listener = dimos.deploy(LidarListener)

    lidar.pointcloud.transport = LCMTransport("/lidar/pointcloud", PointCloud2)
    lidar.imu.transport = LCMTransport("/lidar/imu", Imu)
    listener.pointcloud.transport = LCMTransport("/lidar/pointcloud", PointCloud2)
    listener.imu.transport = LCMTransport("/lidar/imu", Imu)

    lidar.start()
    listener.start()

    print("LivoxLidarModule + listener running. Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    dimos.stop()
