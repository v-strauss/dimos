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
Shared memory transport that mimics LCMTransport interface.
"""

from typing import Callable, Optional
from dimos.core.transport import PubSubTransport
from dimos.utils.shared_memory_transport import SharedMemoryTransport


class SharedMemoryImageTransport(PubSubTransport):
    """Drop-in replacement for LCMTransport using shared memory"""

    def __init__(self, topic: str, shape: tuple, dtype):
        """
        Initialize shared memory image transport.

        Args:
            topic: Topic name (used for shared memory name)
            shape: Shape of the image data
            dtype: Data type of the image
        """
        super().__init__(topic)
        self.transport = SharedMemoryTransport(topic, shape, dtype)

    def broadcast(self, _, msg):
        """Broadcast message to shared memory"""
        self.transport.publish(msg)

    def subscribe(self, callback: Callable, selfstream=None):
        """Subscribe to shared memory updates"""
        return self.transport.subscribe(callback, selfstream)

    def close(self):
        """Close the transport"""
        self.transport.close()
