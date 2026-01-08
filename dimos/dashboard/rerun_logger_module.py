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

"""Dashboard-owned Rerun logger module.

Strict separation:
- Producer modules only publish normal outputs (including metrics).
- This module subscribes and logs to Rerun using message-level `to_rerun()`.
"""

from __future__ import annotations

import time
from typing import Any

import rerun as rr

from dimos.core import In, Module, rpc
from dimos.core.global_config import GlobalConfig
from dimos.dashboard.rerun_init import connect_rerun

# These must be runtime imports for get_type_hints() to resolve In[...] annotations
from dimos.msgs.nav_msgs import OccupancyGrid  # noqa: TC001
from dimos.msgs.sensor_msgs import PointCloud2  # noqa: TC001
from dimos.msgs.std_msgs import Float32  # noqa: TC001
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class RerunLoggerModule(Module):
    """Log selected typed streams to Rerun with stable entity paths."""

    _global_config: GlobalConfig

    # Core visuals
    global_map: In[PointCloud2]
    global_costmap: In[OccupancyGrid]

    # Costmap metrics
    costmap_calc_ms: In[Float32]
    costmap_latency_ms: In[Float32]

    # Voxel metrics
    voxel_extract_ms: In[Float32]
    voxel_transport_ms: In[Float32]
    voxel_publish_ms: In[Float32]
    voxel_latency_ms: In[Float32]
    voxel_count: In[Float32]

    def __init__(
        self,
        global_config: GlobalConfig | None = None,
        *,
        # Voxel map render params (dashboard policy)
        voxel_box_size: float = 0.05,
        voxel_colormap: str | None = "turbo",
        # Costmap render params (dashboard policy)
        costmap_z_offset: float = 0.05,
        # Rate limits (dashboard policy)
        map_rate_limit_hz: float = 10.0,
        costmap_rate_limit_hz: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._global_config = global_config or GlobalConfig()

        self._voxel_box_size = float(voxel_box_size)
        self._voxel_colormap = voxel_colormap
        self._costmap_z_offset = float(costmap_z_offset)

        # Per-stream rate limiting (best-effort) — dashboard policy only.
        self._last_map_log = 0.0
        self._map_rate_limit_hz = float(map_rate_limit_hz)
        self._last_costmap_log = 0.0
        self._costmap_rate_limit_hz = float(costmap_rate_limit_hz)

    @rpc
    def start(self) -> None:
        super().start()
        if not self._global_config.rerun_enabled:
            return
        if not self._global_config.viewer_backend.startswith("rerun"):
            return

        connect_rerun(
            global_config=self._global_config, server_addr=self._global_config.rerun_server_addr
        )

        # Visuals
        self._disposables.add(self.global_map.observable().subscribe(self._on_global_map))  # type: ignore[no-untyped-call]
        self._disposables.add(self.global_costmap.observable().subscribe(self._on_global_costmap))  # type: ignore[no-untyped-call]

        # Metrics (Float32.to_rerun is trivial)
        self._disposables.add(
            self.costmap_calc_ms.observable().subscribe(
                lambda m: rr.log("metrics/costmap/calc_ms", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]
        self._disposables.add(
            self.costmap_latency_ms.observable().subscribe(
                lambda m: rr.log("metrics/costmap/latency_ms", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]

        self._disposables.add(
            self.voxel_extract_ms.observable().subscribe(
                lambda m: rr.log("metrics/voxel_map/extract_ms", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]
        self._disposables.add(
            self.voxel_transport_ms.observable().subscribe(
                lambda m: rr.log("metrics/voxel_map/transport_ms", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]
        self._disposables.add(
            self.voxel_publish_ms.observable().subscribe(
                lambda m: rr.log("metrics/voxel_map/publish_ms", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]
        self._disposables.add(
            self.voxel_latency_ms.observable().subscribe(
                lambda m: rr.log("metrics/voxel_map/latency_ms", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]
        self._disposables.add(
            self.voxel_count.observable().subscribe(
                lambda m: rr.log("metrics/voxel_map/voxel_count", m.to_rerun())
            )
        )  # type: ignore[no-untyped-call]

        logger.info("RerunLoggerModule started")

    def _rate_limit(self, last: float, hz: float) -> tuple[bool, float]:
        if hz <= 0:
            return False, last
        now = time.monotonic()
        if now - last < 1.0 / hz:
            return True, last
        return False, now

    def _on_global_map(self, msg: PointCloud2) -> None:
        limited, new_last = self._rate_limit(self._last_map_log, self._map_rate_limit_hz)
        if limited:
            return
        self._last_map_log = new_last
        rr.log(
            "world/map",
            msg.to_rerun(  # type: ignore[no-untyped-call]
                mode="boxes",
                size=self._voxel_box_size,
                colormap=self._voxel_colormap,
            ),
        )

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        limited, new_last = self._rate_limit(self._last_costmap_log, self._costmap_rate_limit_hz)
        if limited:
            return
        self._last_costmap_log = new_last
        rr.log(
            "world/nav/costmap/floor",
            msg.to_rerun(z_offset=self._costmap_z_offset),  # type: ignore[no-untyped-call]
        )


rerun_logger = RerunLoggerModule.blueprint
