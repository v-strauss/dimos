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

# shared_state.py
from collections.abc import MutableMapping
import copy

# state_server.py
from multiprocessing import Queue


def _state_server(_req_q: Queue, _resp_q: Queue):
    state = {}

    while True:
        msg = _req_q.get()

        op = msg["op"]
        if op == "stop":
            return

        if op == "get":
            # full snapshot
            _resp_q.put(copy.deepcopy(state))

        elif op == "set":
            state = msg["value"]
            _resp_q.put(True)

        elif op == "patch":
            state.update(msg["value"])
            _resp_q.put(True)

        else:
            raise ValueError(f"Unknown op {op}")


class _SharedStateDict(MutableMapping):
    """
    Dictionary-like wrapper.
    ALWAYS refreshes the full state before any read.
    """

    def __init__(self, _req_q, _resp_q):
        self.__req_q = _req_q
        self._resp_q = _resp_q
        self._cache = {}

    # ---- internal ----

    def _refresh(self):
        self.__req_q.put({"op": "get"})
        self._cache = self._resp_q.get()

    # ---- reads (always refresh first) ----

    def __getitem__(self, key):
        self._refresh()
        return self._cache[key]

    def get(self, key, default=None):
        self._refresh()
        return self._cache.get(key, default)

    def __contains__(self, key):
        self._refresh()
        return key in self._cache

    def keys(self):
        self._refresh()
        return self._cache.keys()

    def values(self):
        self._refresh()
        return self._cache.values()

    def items(self):
        self._refresh()
        return self._cache.items()

    def __len__(self):
        self._refresh()
        return len(self._cache)

    def __iter__(self):
        self._refresh()
        return iter(self._cache)

    # ---- writes (explicit) ----

    def __setitem__(self, key, value):
        # patch single key
        self.__req_q.put(
            {
                "op": "patch",
                "value": {key: value},
            }
        )
        self._resp_q.get()

    def update(self, other):
        self.__req_q.put(
            {
                "op": "patch",
                "value": dict(other),
            }
        )
        self._resp_q.get()

    def replace(self, new_dict):
        """Replace entire JSON document"""
        self.__req_q.put(
            {
                "op": "set",
                "value": dict(new_dict),
            }
        )
        self._resp_q.get()

    def __delitem__(self, key):
        self._refresh()
        new_state = dict(self._cache)
        del new_state[key]
        self.replace(new_state)


from multiprocessing import Process, Queue

_req_q = Queue()
_resp_q = Queue()


def start():
    _state_manager = Process(target=_state_server, args=(_req_q, _resp_q))
    _state_manager.start()


state = _SharedStateDict(_req_q, _resp_q)
