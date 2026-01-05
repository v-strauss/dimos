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

"""Convert replay YAML with pickled objects into plain, serializable data."""

from __future__ import annotations

import base64
from pathlib import Path
import sys
import types
from typing import Any

import cv2
import yaml

# Stub out rxpy_backpressure and utils.* dependencies pulled in by Image imports.
if "rxpy_backpressure" not in sys.modules:
    rxbp_mod = types.ModuleType("rxpy_backpressure")

    class _BackPressure:
        """No-op stub."""

        def __init__(self, *args, **kwargs): ...

    rxbp_mod.BackPressure = _BackPressure
    sys.modules["rxpy_backpressure"] = rxbp_mod

    bp_mod = types.ModuleType("rxpy_backpressure.backpressure")
    bp_mod.BackPressure = _BackPressure
    sys.modules["rxpy_backpressure.backpressure"] = bp_mod

    sb_mod = types.ModuleType("rxpy_backpressure.sized_buffer")

    def wrap_observer_with_sized_buffer_strategy(*args, **kwargs):
        return lambda observer: observer

    class _CounterStub:
        def processed_event(self, fn=None):
            return fn

    sb_mod.wrap_observer_with_sized_buffer_strategy = wrap_observer_with_sized_buffer_strategy
    sb_mod.Counter = _CounterStub
    sys.modules["rxpy_backpressure.sized_buffer"] = sb_mod

# Stub out broken rxpy_backpressure dependency that imports `utils.logging`.
if "utils" not in sys.modules:
    utils_mod = types.ModuleType("utils")
    sys.modules["utils"] = utils_mod
logging_mod = types.ModuleType("utils.logging")
import logging as _py_logging

logging_mod.Logger = _py_logging.Logger
logging_mod.log = _py_logging.log
logging_mod.getLogger = _py_logging.getLogger
sys.modules["utils.logging"] = logging_mod

stats_mod = types.ModuleType("utils.stats")
from collections import Counter as _Counter

stats_mod.Counter = _Counter
sys.modules["utils.stats"] = stats_mod

from dimos.msgs.sensor_msgs import Image
from dimos.robot.unitree_webrtc.type.lidar import LidarMessage

SRC_PATHS = {
    "lidar": Path(__file__).with_name("lidar.yaml"),
    "color_image": Path(__file__).with_name("color_image.yaml"),
}
DEST_PATH = Path(__file__).with_name("replay_serialized.yaml")


def _iter_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            loaded = yaml.unsafe_load(line)
            if isinstance(loaded, list):
                yield from loaded
            else:
                yield loaded


def _serialize_lidar(msg: LidarMessage) -> dict[str, Any]:
    points = msg.as_numpy().tolist()
    radius = getattr(msg, "resolution", None)
    ts = getattr(msg, "ts", None)
    return {
        "stream": "lidar",
        "kind": "points3d",
        "positions": points,
        "radius": radius,
        "ts": ts,
    }


def _serialize_image(msg: Image) -> dict[str, Any]:
    ts = getattr(msg, "ts", None)
    jpeg_b64 = msg.to_base64()
    return {
        "stream": "color_image",
        "kind": "image",
        "encoding": "jpeg",
        "data_base64": jpeg_b64,
        "width": msg.width,
        "height": msg.height,
        "ts": ts,
    }


def main() -> int:
    output: list[dict[str, Any]] = []

    for _stream, path in SRC_PATHS.items():
        if not path.exists():
            print(f"[serialize_replay_data] missing source file: {path}", file=sys.stderr)
            continue
        for msg in _iter_yaml(path):
            if isinstance(msg, LidarMessage):
                output.append(_serialize_lidar(msg))
            elif isinstance(msg, Image):
                output.append(_serialize_image(msg))
            else:
                print(f"[serialize_replay_data] skipping unknown msg type {type(msg)} from {path}")

    if not output:
        print("[serialize_replay_data] no data serialized; nothing written", file=sys.stderr)
        return 1

    DEST_PATH.write_text(yaml.safe_dump(output), encoding="utf-8")
    print(f"[serialize_replay_data] wrote {len(output)} messages to {DEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
