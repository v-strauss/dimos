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

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
import os
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# ----------------------------- sudo helpers -----------------------------


@cache
def _is_root_user() -> bool:
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False


def sudo_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    if _is_root_user():
        return subprocess.run(list(args), **kwargs)
    return subprocess.run(["sudo", *args], **kwargs)


def _read_sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(["sysctl", name], capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"[sysctl] ERROR: `sysctl {name}` rc={result.returncode} stderr={result.stderr!r}"
            )
            return None

        text = result.stdout.strip().replace(":", "=")
        if "=" not in text:
            print(f"[sysctl] ERROR: unexpected output for {name}: {text!r}")
            return None

        return int(text.split("=", 1)[1].strip())
    except Exception as error:
        print(f"[sysctl] ERROR: reading {name}: {error}")
        return None


def _write_sysctl_int(name: str, value: int) -> None:
    sudo_run("sysctl", "-w", f"{name}={value}", check=True, text=True, capture_output=False)


# -------------------------- base class for system config checks/requirements --------------------------


class SystemConfigurator(ABC):
    critical: bool = False

    @abstractmethod
    def check(self) -> bool:
        """Return True if configured. Log errors and return False on uncertainty."""
        raise NotImplementedError

    @abstractmethod
    def explanation(self) -> str | None:
        """
        Return a human-readable summary of what would be done (sudo commands) if not configured.
        Return None when no changes are needed.
        """
        raise NotImplementedError

    @abstractmethod
    def fix(self) -> None:
        """Apply fixes (may attempt sudo, catch, and apply fallback measures if needed)."""
        raise NotImplementedError


# ----------------------------- generic enforcement of system configs -----------------------------


def configure_system(checks: list[SystemConfigurator], check_only: bool = False) -> None:
    if os.environ.get("CI"):
        print("CI environment detected: skipping system configuration.")
        return

    # run checks
    failing = [check for check in checks if not check.check()]
    if not failing:
        return

    # ask for permission to modify system
    explanations: list[str] = [msg for check in failing if (msg := check.explanation()) is not None]

    if explanations:
        print("System configuration changes are recommended/required:\n")
        print("\n\n".join(explanations))
        print()

    if check_only:
        return

    try:
        answer = input("Apply these changes now? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer not in ("y", "yes"):
        if any(check.critical for check in failing):
            raise SystemExit(1)
        return

    for check in failing:
        try:
            check.fix()
        except subprocess.CalledProcessError as error:
            if check.critical:
                print(f"Critical fix failed rc={error.returncode}")
                print(f"stdout: {error.stdout}")
                print(f"stderr: {error.stderr}")
                raise
            print(f"Optional improvement failed: rc={error.returncode}")
            print(f"stdout: {error.stdout}")
            print(f"stderr: {error.stderr}")

    print("System configuration completed.")


# ----------------------------- bridge: SystemConfigurator → Blueprint.requirements() -----------------------------


def system_checks(*configurators: SystemConfigurator) -> Callable[[], str | None]:
    """Wrap SystemConfigurator instances into a Blueprint.requirements()-compatible callable.

    Returns a function that runs configure_system() and converts SystemExit
    (raised when a critical check is declined) into an error string.
    Non-critical declines return None (proceed with degraded state).
    """

    def _check() -> str | None:
        try:
            configure_system(list(configurators))
        except SystemExit:
            labels = [type(c).__name__ for c in configurators]
            return f"Required system configuration was declined: {', '.join(labels)}"
        return None

    return _check
