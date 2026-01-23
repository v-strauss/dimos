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

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Any

from dimos.utils.cli.lcmspy.lcmspy import Topic as BaseTopic


@dataclass
class DDSSpyConfig:
    domain_id: int | None = None
    poll_interval: float = 0.2
    cyclonedds_uri: str | None = None


class DDSTopic(BaseTopic):
    def __init__(self, name: str, history_window: float = 60.0) -> None:
        super().__init__(name=name, history_window=history_window)
        self.type_name: str | None = None
        self.writers: set[str] = set()
        self.readers: set[str] = set()
        self.last_seen: float = 0.0

    def msg_size(self, size: int) -> None:
        self.message_history.append((time.time(), size))
        self.total_traffic_bytes += size
        self._cleanup_old_messages()


class DDSSpy:
    def __init__(self, config: DDSSpyConfig) -> None:
        self.config = config
        self.topic: dict[str, DDSTopic] = {}
        self._participant = None
        self._builtin_pub_reader = None
        self._builtin_sub_reader = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._imports = None

    def start(self) -> None:
        if self.config.cyclonedds_uri:
            os.environ["CYCLONEDDS_URI"] = self.config.cyclonedds_uri

        self._imports = _import_cyclonedds()
        domain_id = self.config.domain_id if self.config.domain_id is not None else 0
        self._participant = self._imports.DomainParticipant(domain_id)

        builtin_names = self._imports.BuiltinTopicNames
        publication_name = _get_builtin_topic_name(
            builtin_names,
            "DCPSPublication",
            "DCPS_PUBLICATION",
            "DCPSPublicationBuiltinTopicData",
        )
        subscription_name = _get_builtin_topic_name(
            builtin_names,
            "DCPSSubscription",
            "DCPS_SUBSCRIPTION",
            "DCPSSubscriptionBuiltinTopicData",
        )

        self._builtin_pub_reader = self._imports.BuiltinDataReader(
            self._participant, publication_name
        )
        self._builtin_sub_reader = self._imports.BuiltinDataReader(
            self._participant, subscription_name
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_discovery()
            except Exception:
                pass
            time.sleep(self.config.poll_interval)

    def _poll_discovery(self) -> None:
        pub_samples = _take_samples(self._builtin_pub_reader)
        if pub_samples:
            for sample in pub_samples:
                self._handle_discovery_sample(sample, is_writer=True)

        sub_samples = _take_samples(self._builtin_sub_reader)
        if sub_samples:
            for sample in sub_samples:
                self._handle_discovery_sample(sample, is_writer=False)

    def _handle_discovery_sample(self, sample: Any, is_writer: bool) -> None:
        data, info = _extract_sample(sample)
        if info is not None and hasattr(info, "valid_data") and not info.valid_data:
            self._handle_dispose(info, data, is_writer)
            return

        topic_name = _get_attr(data, ("topic_name", "topicName", "topic"))
        if not topic_name:
            return
        type_name = _get_attr(data, ("type_name", "typeName", "type"))
        instance_key = _get_attr(data, ("key", "instance_key", "instance_handle", "handle"))

        with self._lock:
            topic = self.topic.get(topic_name)
            if not topic:
                topic = DDSTopic(topic_name)
                self.topic[topic_name] = topic
            if type_name and not topic.type_name:
                topic.type_name = str(type_name)

            now = time.time()
            topic.last_seen = now
            topic.msg_size(0)

            if instance_key is not None:
                key = _format_key(instance_key)
                if is_writer:
                    topic.writers.add(key)
                else:
                    topic.readers.add(key)

    def _handle_dispose(self, info: Any, data: Any, is_writer: bool) -> None:
        instance_key = _get_attr(data, ("key", "instance_key", "instance_handle", "handle"))
        topic_name = _get_attr(data, ("topic_name", "topicName", "topic"))
        if not topic_name or instance_key is None:
            return
        with self._lock:
            topic = self.topic.get(topic_name)
            if not topic:
                return
            key = _format_key(instance_key)
            if is_writer:
                topic.writers.discard(key)
            else:
                topic.readers.discard(key)


@dataclass
class CycloneDDSImports:
    DomainParticipant: Any
    BuiltinDataReader: Any
    BuiltinTopicNames: Any


def _import_cyclonedds() -> CycloneDDSImports:
    try:
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicNames
        from cyclonedds.domain import DomainParticipant
    except Exception as exc:
        raise ImportError(
            "Cyclone DDS Python bindings not found. Install 'cyclonedds' to use ddsspy."
        ) from exc

    return CycloneDDSImports(
        DomainParticipant=DomainParticipant,
        BuiltinDataReader=BuiltinDataReader,
        BuiltinTopicNames=BuiltinTopicNames,
    )


def _get_builtin_topic_name(names: Any, *candidates: str) -> Any:
    for name in candidates:
        if hasattr(names, name):
            return getattr(names, name)
    return candidates[0]


def _take_samples(reader: Any) -> list[Any]:
    if reader is None:
        return []
    if hasattr(reader, "take"):
        return list(reader.take())
    if hasattr(reader, "read"):
        return list(reader.read())
    return []


def _extract_sample(sample: Any) -> tuple[Any, Any | None]:
    if hasattr(sample, "data"):
        return sample.data, getattr(sample, "info", None)
    return sample, getattr(sample, "info", None)


def _get_attr(obj: Any, names: tuple[str, ...]) -> Any | None:
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    return None


def _format_key(key: Any) -> str:
    if hasattr(key, "value"):
        return str(key.value)
    return str(key)
