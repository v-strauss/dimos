#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""
Dask actor variant: dashboard runs in one actor, color and lidar loggers run in their own actors.

No futures for synchronization; actors encapsulate state and methods are invoked directly.
"""

import dataclasses
import os
from pathlib import Path
import sys
import time

from dask.distributed import Client, LocalCluster
import rerun as rr
import rerun.blueprint as rrb
import yaml

from dimos.dashboard.server import start_dashboard_server_thread

default_rerun_grpc_port = 19876
default_dashboard_port = 4400


@dataclasses.dataclass
class RerunInfo:
    logging_id: str = os.environ.get("RERUN_ID", "dimos_main_rerun")
    grpc_port: int = int(os.environ.get("RERUN_GRPC_PORT", default_rerun_grpc_port))
    server_memory_limit: str = os.environ.get("RERUN_SERVER_MEMORY_LIMIT", "25%")
    url: str = os.environ.get(
        "RERUN_URL",
        f"rerun+http://localhost:{os.environ.get('RERUN_GRPC_PORT', default_rerun_grpc_port)!s}/proxy",
    )


rerun_info = RerunInfo()


class RerunConnection:
    def __init__(self) -> None:
        self.stream = rr.RecordingStream(rerun_info.logging_id, recording_id=rerun_info.logging_id)
        self.stream.connect_grpc(rerun_info.url)

    def log(self, msg: str, value, **kwargs) -> None:
        self.stream.log(msg, value, **kwargs)


class DashboardActor:
    def __init__(self) -> None:
        print("[DashboardActor] init rerun")
        rr.init(rerun_info.logging_id, spawn=False, recording_id=rerun_info.logging_id)
        self.default_blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Spatial3DView(
                    name="Spatial3D",
                    origin="/",
                    line_grid=rrb.LineGrid3D(spacing=1.0, stroke_width=1.0),
                ),
            )
        )
        print("[DashboardActor] sending empty blueprint")
        rr.send_blueprint(self.default_blueprint)
        if not os.environ.get("RERUN_URL", None):
            print("[DashboardActor] starting rerun grpc")
            rr.serve_grpc(
                grpc_port=rerun_info.grpc_port,
                default_blueprint=self.default_blueprint,
                server_memory_limit=rerun_info.server_memory_limit,
            )
        self.thread = start_dashboard_server_thread(
            **{
                "auto_open": True,
                "terminal_commands": {"agent-spy": "htop", "lcm-spy": "dimos lcmspy"},
                "worker": 1,
                "port": int(os.environ.get("DASHBOARD_PORT", default_dashboard_port)),
            },
            rrd_url=rerun_info.url,
            keep_alive=True,
        )
        pass

    def ping(self) -> str:
        return "dashboard-alive"

    def close(self) -> None:
        print("[DashboardActor] closing")
        try:
            self.thread.join(timeout=1)
        except Exception as error:
            print(f"[DashboardActor.close] error: {error}")


class LoggerActor:
    def __init__(self, name: str, path_str: str) -> None:
        # self.name = name
        # self.path = Path(path_str)
        # self.rc = RerunConnection()
        # print(f"[LoggerActor-{self.name}] init complete")
        pass

    def replay(self) -> None:
        # if not self.path.exists():
        #     print(f"[LoggerActor-{self.name}] file {self.path} does not exist", file=sys.stderr)
        #     return

        # with self.path.open("r", encoding="utf-8") as f:
        #     for line_number, line in enumerate(f):
        #         if not line.strip():
        #             continue
        #         try:
        #             parsed = yaml.unsafe_load(line) or []
        #         except Exception as error:
        #             print(f"[LoggerActor-{self.name}] warning line:{line_number}: {error}")
        #             continue

        #         try:
        #             payload = parsed[0] if isinstance(parsed, list) else parsed
        #             print(f"[LoggerActor-{self.name}] logging message {line_number}")
        #             self.rc.log(f"/{self.name}", payload.to_rerun(), strict=True)
        #         except Exception as error:
        #             print(f"[LoggerActor-{self.name}] error: {error}")
        pass

    def close(self) -> None:
        print(f"[LoggerActor-{self.name}] close called")


def main() -> None:
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=4,
        processes=True,
        dashboard_address=None,
        services={},  # disable HTTP services that may require privileged ports
        scheduler_port=0,
        diagnostics_port=None,
        nanny=False,  # keep the sole worker from being restarted mid-actor-run
    )
    client = Client(cluster)

    # Pin actors to the lone worker so they don't move or disappear.
    worker_addrs = list(client.scheduler_info()["workers"].keys())
    target_worker = worker_addrs[0]
    print(f"[Main] Using worker {target_worker} for actors")

    dashboard = client.submit(DashboardActor, actor=True, workers=[target_worker]).result()
    # Give the dashboard/grpc server a moment to spin up before logging.
    time.sleep(1.5)

    color_actor = client.submit(
        LoggerActor,
        "color_image",
        "./dimos/dashboard/support/color_image.yaml",
        actor=True,
        workers=[target_worker],
    ).result()
    lidar_actor = client.submit(
        LoggerActor,
        "lidar",
        "./dimos/dashboard/support/lidar.yaml",
        actor=True,
        workers=[target_worker],
    ).result()

    # Kick off both logging actors.
    color_actor.replay()
    lidar_actor.replay()

    # Cleanup
    color_actor.close()
    lidar_actor.close()
    dashboard.close()

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
