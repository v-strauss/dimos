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
Run the p1 + p2 behavior on a single Dask worker to reproduce multiprocessing issues.

- One Dask task mirrors p1.py (starts rerun init/grpc and the dashboard server).
- A second Dask task mirrors p2.py but logs both color_image and lidar concurrently.

This avoids python's builtin multiprocessing; instead it uses Dask (LocalCluster +
distributed workers). Configure the worker count below (default: 1 worker).
"""

import dataclasses
import os
from pathlib import Path
import sys
import threading
import time

from dask.distributed import Client, Event, LocalCluster
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import yaml

from dimos.dashboard.server import start_dashboard_server_thread

default_rerun_grpc_port = 9876


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
        self.init_id = threading.get_ident()
        self.stream = rr.RecordingStream(rerun_info.logging_id, recording_id=rerun_info.logging_id)
        self.stream.connect_grpc(rerun_info.url)

    def log(self, msg: str, value, **kwargs) -> None:
        if self.init_id != threading.get_ident():
            raise Exception(
                """Looks like you are somehow using RerunConnection to log data to rerun. However, the thread where you init RerunConnection is different from where you are logging. A RerunConnection object needs to be created once per thread."""
            )

        self.stream.log(msg, value, **kwargs)


def dashboard_process(stop_event_name: str) -> None:
    """Mirrors p1.py: init rerun, send blueprint, start grpc and dashboard server."""
    stop_event = Event(stop_event_name)

    print("[Dask Dashboard] calling rr.init")
    rr.init(rerun_info.logging_id, spawn=False, recording_id=rerun_info.logging_id)
    default_blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(
                name="Spatial3D",
                origin="/",
                line_grid=rrb.LineGrid3D(spacing=1.0, stroke_width=1.0),
            ),
        )
    )
    print("[Dask Dashboard] sending empty blueprint")
    rr.send_blueprint(default_blueprint)
    if not os.environ.get("RERUN_URL", None):
        print("[Dask Dashboard] starting rerun grpc")
        rr.serve_grpc(
            grpc_port=rerun_info.grpc_port,
            default_blueprint=default_blueprint,
            server_memory_limit=rerun_info.server_memory_limit,
        )
    thread = start_dashboard_server_thread(
        **{
            "auto_open": True,
            "terminal_commands": {"agent-spy": "htop", "lcm-spy": "dimos lcmspy"},
            "worker": 1,
        },
        rrd_url=rerun_info.url,
        keep_alive=True,
    )
    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("[Dask Dashboard] interrupted, shutting down")
    finally:
        thread.join(timeout=1)


def _log_stream(name: str, path: Path, start_event: threading.Event) -> None:
    rc = RerunConnection()
    if not path.exists():
        print(f"[Dask Logging {name}] file {path} does not exist", file=sys.stderr)
        return

    start_event.wait()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            try:
                parsed = yaml.unsafe_load(line) or []
            except Exception as error:
                print(f"[Dask Logging {name}] warning line:{line_number}: {error}")
                continue

            try:
                payload = parsed[0] if isinstance(parsed, list) else parsed
                print(f"[Dask Logging {name}] logging message {line_number}")
                rc.log(f"/{name}", payload.to_rerun(), strict=True)
            except Exception as error:
                print(f"[Dask Logging {name}] error: {error}")


def logging_process() -> None:
    """Mirrors p2.py: replay color_image.yaml and lidar.yaml concurrently."""
    start_event = threading.Event()
    files = {
        "color_image": Path("./dimos/dashboard/support/color_image.yaml"),
        "lidar": Path("./dimos/dashboard/support/lidar.yaml"),
    }

    threads: list[threading.Thread] = []
    for name, path in files.items():
        t = threading.Thread(target=_log_stream, args=(name, path, start_event), name=f"log-{name}")
        threads.append(t)
        t.start()

    start_event.set()

    for t in threads:
        t.join()


def main() -> None:
    # Adjust worker/threads here if needed. The repro asks for 1 worker.
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=4,
        processes=True,
        dashboard_address=None,  # avoid port conflicts with rerun viewer
    )
    client = Client(cluster)
    stop_event_name = "dask-rerun-stop-event"
    stop_event = Event(stop_event_name, client=client)

    dashboard_future = client.submit(dashboard_process, stop_event_name, pure=False)
    # Give the dashboard/grpc server a moment to spin up before logging.
    time.sleep(1.5)
    logging_future = client.submit(logging_process, pure=False)

    logging_future.result()
    stop_event.set()
    # Give the dashboard process a moment to wind down.
    try:
        dashboard_future.result(timeout=5)
    except Exception as exc:
        print(f"[Main] dashboard future ended with: {exc}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
