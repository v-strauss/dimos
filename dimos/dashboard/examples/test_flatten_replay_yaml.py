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

from pathlib import Path

import pytest
import yaml

from dimos.dashboard.examples.flatten_replay_yaml import _flatten_one, yaml_replay_read


def test_flatten_round_trip(tmp_path):
    mod_dir = Path(__file__).parent
    sources = [
        mod_dir / "example_data_lidar.yaml",
        mod_dir / "example_data_color_image.yaml",
    ]

    for src in sources:
        if not src.exists():
            pytest.skip(f"source file missing: {src}")

    for src in sources:
        dst = tmp_path / f"{src.name}.flat"
        _flatten_one(src, dst)

        original = yaml.safe_load(src.read_text(encoding="utf-8"))
        original_list = original if isinstance(original, list) else [original]

        flattened_list = list(yaml_replay_read(dst))
        assert original_list == flattened_list
