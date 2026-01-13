#!/usr/bin/env python3

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

import time

from dimos import core
from dimos.msgs.sensor_msgs import JointCommand, JointState

latest_state: JointState | None = None


def on_joint_state(msg: JointState) -> None:
    global latest_state
    latest_state = msg


def wait_for_state(timeout_s: float = 5.0) -> JointState | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if latest_state is not None:
            return latest_state
        time.sleep(0.05)
    return None


def main() -> None:
    state_sub = core.LCMTransport("/xarm/joint_states", JointState)
    cmd_pub = core.LCMTransport("/xarm/joint_position_command", JointCommand)
    state_sub.subscribe(on_joint_state)

    print("Waiting for /xarm/joint_states...")
    state = wait_for_state()
    if state is None:
        print("No joint state received; exiting.")
        state_sub.stop()
        return

    target = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if len(state.position) != len(target):
        print(
            f"Expected {len(target)} joints, got {len(state.position)}. "
            "Update target size to match your arm."
        )
        state_sub.stop()
        cmd_pub.stop()
        return

    cmd_pub.broadcast(None, JointCommand(positions=target))
    print(f"sent /xarm/joint_position_command: {target}")

    timeout_s = 10.0
    threshold = 0.02
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while time.monotonic() < deadline:
        state = latest_state
        if state is None:
            time.sleep(0.05)
            continue
        errors = [abs(a - b) for a, b in zip(state.position, target, strict=False)]
        max_err = max(errors) if errors else float("inf")
        now = time.monotonic()
        if now - last_print >= 0.5:
            positions = ", ".join(f"{v:.4f}" for v in state.position)
            print(f"joint_states positions=[{positions}] max_err={max_err:.4f}")
            last_print = now
        if max_err <= threshold:
            print(f"target reached (max_err={max_err:.4f} <= {threshold:.2f})")
            break
        time.sleep(0.05)
    else:
        print("timeout waiting for target; check the robot and try again.")

    cmd_pub.stop()
    state_sub.stop()


if __name__ == "__main__":
    main()
