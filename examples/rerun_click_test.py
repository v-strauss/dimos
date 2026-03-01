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

"""E2E test: Rerun viewer click → Go2 navigation goal.

Usage:
    1. Start the custom rerun viewer (with LCM click support):
       ./target/release/custom_callback_viewer

    2. Run this script (simulation mode):
       uv run python examples/rerun_click_test.py --simulation

    Or replay mode (no robot, headless):
       uv run python examples/rerun_click_test.py --replay

    3. Click on entities in the Rerun viewer → robot navigates to click position.

The custom viewer listens on gRPC port 9877 and publishes PointStamped
clicks to /clicked_point via LCM UDP multicast.
"""

import argparse
import signal
import sys
import time

import rerun as rr

from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.voxels import voxel_mapper
from dimos.msgs.geometry_msgs import PointStamped
from dimos.navigation.replanning_a_star.module import replanning_a_star_planner
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.robot.unitree.go2.connection import go2_connection
from dimos.visualization.rerun.bridge import rerun_bridge
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis

CUSTOM_VIEWER_URL = "rerun+http://127.0.0.1:9877/proxy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun click-to-navigate test")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation (mujoco)")
    parser.add_argument("--replay", action="store_true", help="Run in replay mode (no robot)")
    parser.add_argument("--robot-ip", type=str, default=None, help="Robot IP (for real hardware)")
    parser.add_argument(
        "--viewer-url",
        type=str,
        default=CUSTOM_VIEWER_URL,
        help="Custom viewer gRPC URL",
    )
    args = parser.parse_args()

    # Configure global config
    if args.simulation:
        global_config.update(simulation=True)
    if args.replay:
        global_config.update(replay=True)
    global_config.update(viewer_backend="rerun")
    if args.robot_ip:
        global_config.update(robot_ip=args.robot_ip)

    # Initialize rerun in the MAIN process and connect to the custom viewer.
    # The bridge runs in a worker subprocess via multiprocessing, so its
    # rr.init() only affects that subprocess. We need our own rr.init() +
    # connect_grpc() in the main process for rr.log() calls to reach the viewer.
    rr.init("dimos")
    rr.connect_grpc(args.viewer_url)
    print(f"Connected to custom Rerun viewer at {args.viewer_url}")

    print("Building blueprint...")
    rerun_config = {"pubsubs": [LCM(autoconf=True)], "viewer_mode": "none"}
    with_vis = autoconnect(rerun_bridge(**rerun_config))

    blueprint = (
        autoconnect(
            with_vis,
            go2_connection(),
            websocket_vis(),
            voxel_mapper(voxel_size=0.1),
            cost_mapper(),
            replanning_a_star_planner(),
        )
        .transports(
            {
                ("clicked_point", PointStamped): LCMTransport("/clicked_point", PointStamped),
            }
        )
        .global_config(n_workers=6, robot_model="unitree_go2")
    )

    coordinator = blueprint.build()

    print("Starting modules...")
    coordinator.start()
    print()
    print("Click on entities in the Rerun viewer to send navigation goals.")
    print("Press Ctrl+C to stop.")

    def shutdown(*_: object) -> None:
        print("\nShutting down...")
        coordinator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
