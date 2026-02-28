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

"""QuadrupedAdapter registry with auto-discovery.

Mirrors the TwistBaseAdapterRegistry pattern: each subpackage provides a
``register(registry)`` function in its ``adapter.py`` module.

Usage:
    from dimos.hardware.quadrupeds.registry import quadruped_adapter_registry

    adapter = quadruped_adapter_registry.create("unitree_go2")
    print(quadruped_adapter_registry.available())  # ["unitree_go2"]
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dimos.hardware.quadrupeds.spec import QuadrupedAdapter

logger = logging.getLogger(__name__)


class QuadrupedAdapterRegistry:
    """Registry for quadruped adapters with auto-discovery."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[QuadrupedAdapter]] = {}

    def register(self, name: str, cls: type[QuadrupedAdapter]) -> None:
        """Register an adapter class."""
        self._adapters[name.lower()] = cls

    def create(self, name: str, **kwargs: Any) -> QuadrupedAdapter:
        """Create an adapter instance by name."""
        key = name.lower()
        if key not in self._adapters:
            raise KeyError(f"Unknown quadruped adapter: {name}. Available: {self.available()}")
        return self._adapters[key](**kwargs)

    def available(self) -> list[str]:
        """List available adapter names."""
        return sorted(self._adapters.keys())

    def discover(self) -> None:
        """Discover and register adapters from subpackages."""
        import dimos.hardware.quadrupeds as pkg

        for _, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            if not ispkg:
                continue
            # Walk one level deeper: dimos.hardware.quadrupeds.unitree.go2.adapter
            try:
                sub = importlib.import_module(f"dimos.hardware.quadrupeds.{name}")
                for _, sub_name, sub_ispkg in pkgutil.iter_modules(sub.__path__):
                    if not sub_ispkg:
                        continue
                    try:
                        mod = importlib.import_module(
                            f"dimos.hardware.quadrupeds.{name}.{sub_name}.adapter"
                        )
                        if hasattr(mod, "register"):
                            mod.register(self)
                    except ImportError as e:
                        logger.warning(
                            f"Skipping quadruped adapter {name}.{sub_name}: {e}"
                        )
            except ImportError as e:
                logger.warning(f"Skipping quadruped package {name}: {e}")


quadruped_adapter_registry = QuadrupedAdapterRegistry()
quadruped_adapter_registry.discover()

__all__ = ["QuadrupedAdapterRegistry", "quadruped_adapter_registry"]
