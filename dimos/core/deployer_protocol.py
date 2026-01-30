# Copyright 2026 Dimensional Inc.
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

from typing import Any, Protocol

from dimos.core.module import Module, ModuleT
from dimos.core.rpc_client import RPCClient


# the class below is only ever used for type hinting
# why? because the RPCClient instance is going to have all the methods of a Module
# but those methods/attributes are super dynamic, so the type hints can't figure that out
class ModuleProxy(RPCClient, Module):  # type: ignore[misc]
    def start(self) -> None: ...
    def stop(self) -> None: ...


# all our different module types (DaskDeployer, WorkerDeployer, DockerDeployer, etc.) need to implement this
class DeployerProtocol(Protocol):
    def deploy(self, module_class: type[ModuleT], *args: Any, **kwargs: Any) -> ModuleProxy: ...
    def close_all(self) -> None: ...
