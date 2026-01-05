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
Replay color_image and lidar streams as separate modules (vs a single DataReplay).

Useful for reproducing multi-process/worker issues by splitting the replay work.
"""

from collections.abc import Iterable
from pathlib import Path
import sys
import threading
import time

from reactivex.disposable import Disposable
import yaml

from dimos.core import Module, Out, pLCMTransport, pSHMTransport
from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.dashboard.module import Dashboard, RerunConnection
from dimos.msgs.sensor_msgs import Image
from dimos.robot.unitree_webrtc.type.lidar import LidarMessage


class Dashboard(Module):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        pass


def _iter_messages(path: str) -> Iterable:
    file_path = Path(path)
    if not file_path.exists():
        print(f"[DataReplaySplit] file {path} does not exist", file=sys.stderr)
        return []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            try:
                parsed = yaml.unsafe_load(line) or []
            except Exception as error:
                print(f"[DataReplaySplit] warning: line:{line_number} could not be parsed: {error}")
                continue

            if isinstance(parsed, list):
                yield from parsed
            else:
                yield parsed


class ColorReplay(Module):
    color_image: Out[Image] = None  # type: ignore[assignment]

    def __init__(
        self,
        path: str = "./dimos/dashboard/support/color_image.yaml",
        interval_sec: float = 0.05,
        loop: bool = True,
        **kwargs,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.path = path
        self.interval_sec = interval_sec
        self.loop = loop
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _publish_stream(self) -> None:
        rc = RerunConnection()
        output: Out = self.color_image
        while not self._stop_event.is_set():
            any_sent = False
            for i, msg in enumerate(_iter_messages(self.path)):
                if self._stop_event.is_set():
                    break
                if output and output.transport:
                    if i % 20 == 0:
                        print(f"[ColorReplay] publishing color_image message {i}")
                    rc.log("/color_image", msg.to_rerun(), strict=True)
                    output.publish(msg)  # type: ignore[no-untyped-call]
                any_sent = True
                # time.sleep(self.interval_sec)  # enable if pacing is needed
            if not self.loop or not any_sent:
                break

    @rpc
    def start(self) -> None:
        super().start()
        try:
            self._thread = threading.Thread(
                target=self._publish_stream, name="color-replay", daemon=True
            )
            self._thread.start()
            self._disposables.add(Disposable(self._stop_event.set))
            if self._thread:
                self._disposables.add(Disposable(self._thread.join))
        except Exception as error:
            print(f"[ColorReplay] error = {error}")


class LidarReplay(Module):
    lidar: Out[LidarMessage] = None  # type: ignore[assignment]

    def __init__(
        self,
        path: str = "./dimos/dashboard/support/lidar.yaml",
        interval_sec: float = 0.05,
        loop: bool = True,
        **kwargs,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.path = path
        self.interval_sec = interval_sec
        self.loop = loop
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _publish_stream(self) -> None:
        rc = RerunConnection()
        output: Out = self.lidar
        while not self._stop_event.is_set():
            any_sent = False
            for i, msg in enumerate(_iter_messages(self.path)):
                if self._stop_event.is_set():
                    break
                if output and output.transport:
                    # if i % 20 == 0:
                    print(f"[LidarReplay] publishing lidar message {i}")
                    rc.log("/lidar", msg.to_rerun(), strict=True)
                    output.publish(msg)  # type: ignore[no-untyped-call]
                any_sent = True
                # time.sleep(self.interval_sec)  # enable if pacing is needed
            if not self.loop or not any_sent:
                break

    @rpc
    def start(self) -> None:
        super().start()
        try:
            self._thread = threading.Thread(
                target=self._publish_stream, name="lidar-replay", daemon=True
            )
            self._thread.start()
            self._disposables.add(Disposable(self._stop_event.set))
            if self._thread:
                self._disposables.add(Disposable(self._thread.join))
        except Exception as error:
            print(f"[LidarReplay] error = {error}")


# NOTE: this data was recorded with `from dimos.dashboard.support.utils import record_message`
replay_paths = {
    "color_image": "./dimos/dashboard/support/color_image.yaml",
    "lidar": "./dimos/dashboard/support/lidar.yaml",
}

blueprint = (
    autoconnect(
        ColorReplay.blueprint(
            path=replay_paths["color_image"],
            interval_sec=0.05,
            loop=True,
        ),
        LidarReplay.blueprint(
            path=replay_paths["lidar"],
            interval_sec=0.05,
            loop=True,
        ),
        Dashboard.blueprint(
            auto_open=True,
            terminal_commands={
                "agent-spy": "htop",
                "lcm-spy": "dimos lcmspy",
            },
        ),
    )
    .transports(
        {
            ("color_image", Image): pSHMTransport("/replay/color_image"),
            ("lidar", LidarMessage): pLCMTransport("/replay/lidar"),
        }
    )
    .global_config(n_dask_workers=2, threads_per_worker=4)
)


def main() -> None:
    coordinator = blueprint.build()
    print("Split data replay running. Press Ctrl+C to stop.")
    coordinator.loop()


if __name__ == "__main__":
    main()
