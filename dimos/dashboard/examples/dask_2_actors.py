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

"""Minimal Dask actors check: two simple actors, 1 worker / 4 threads."""

from dask.distributed import Client, LocalCluster


class SimpleActorA:
    def __init__(self, name: str) -> None:
        self.name = name
        print(f"[SimpleActorA-{name}] init")

    def ping(self) -> str:
        msg = f"pongA-{self.name}"
        print(f"[SimpleActorA-{self.name}] ping -> {msg}")
        return msg


class SimpleActorB:
    def __init__(self, name: str) -> None:
        self.name = name
        print(f"[SimpleActorB-{name}] init")

    def ping(self) -> str:
        msg = f"pongB-{self.name}"
        print(f"[SimpleActorB-{self.name}] ping -> {msg}")
        return msg


def main() -> None:
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=4,
        processes=True,
        dashboard_address=None,
        services={},
        scheduler_port=0,
        diagnostics_port=None,
        nanny=False,
    )
    client = Client(cluster)

    worker = next(iter(client.scheduler_info()["workers"].keys()))
    print(f"[Main] worker: {worker}")

    a = client.submit(SimpleActorA, "alpha", actor=True, workers=[worker]).result()
    b = client.submit(SimpleActorB, "beta", actor=True, workers=[worker]).result()

    print(a.ping())
    print(b.ping())

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
