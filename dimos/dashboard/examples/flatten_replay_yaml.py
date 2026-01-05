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

"""Flatten replay YAML files to one list entry per line for streaming use."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


# Custom dumper that forces single-line scalars (no folded binary blocks).
class _OneLineDumper(yaml.SafeDumper):
    pass


def _represent_binary(dumper: yaml.Dumper, data: bytes):
    import base64

    b64 = base64.b64encode(data).decode("ascii")
    # Plain scalar with binary tag; long width handled by dumper width setting.
    return dumper.represent_scalar("tag:yaml.org,2002:binary", b64, style="")


_OneLineDumper.add_representer(bytes, _represent_binary)
_OneLineDumper.add_representer(bytearray, lambda d, data: _represent_binary(d, bytes(data)))


def serialize_entry(item: object) -> str:
    """Serialize one entry to a single-line YAML list item."""
    return yaml.dump(
        item,
        Dumper=_OneLineDumper,
        default_flow_style=True,
        width=10**9,
        allow_unicode=False,
    ).strip()


def deserialize_entry(line: str) -> object:
    """Deserialize one entry from a single-line YAML list item (no leading '- ')."""
    return yaml.safe_load(line)


def yaml_replay_read(path: str | Path) -> Iterator[object]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.startswith("- "):
                continue
            try:
                loaded = deserialize_entry(line[2:])
            except Exception as exc:
                print(f"[yaml_replay_read] line {i} parse error: {exc}", file=sys.stderr)
                continue
            if loaded is None:
                continue
            yield loaded


def _flatten_one(src: Path, dst: Path) -> int:
    """Flatten a replay YAML file to one list entry per line."""
    if not src.exists():
        print(f"[flatten_replay_yaml] missing source: {src}", file=sys.stderr)
        return 0

    count = 0
    import pickle

    with src.open("r", encoding="utf-8") as infile, dst.open("w", encoding="utf-8") as outfile:
        data = yaml.safe_load(infile)
        items: Iterable[object] = data if isinstance(data, list) else [data]
        for item in items:
            dumped = serialize_entry(item)
            print(f"""len(item) = {len(item)}""")
            if item != deserialize_entry(dumped):
                raise AssertionError(f"Round-trip mismatch for {src} -> {dst}")
            outfile.write(f"- {dumped}\n")
            count += 1
    print(f"[flatten_replay_yaml] wrote {count} entries -> {dst}")
    return count


def main() -> int:
    base_dir = Path(__file__).parent
    sources = [
        base_dir / "example_data_lidar.yaml",
        base_dir / "example_data_color_image.yaml",
    ]
    for src in sources:
        dst = src.with_suffix(".flat.yaml")
        _flatten_one(src, dst)
        # Quick round-trip check
        original = yaml.safe_load(src.read_text(encoding="utf-8"))
        original_list = original if isinstance(original, list) else [original]
        flattened_list = list(yaml_replay_read(dst))
        if original_list != flattened_list:
            raise AssertionError(f"Round-trip mismatch for {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
