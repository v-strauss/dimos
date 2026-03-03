#!/usr/bin/env python3
"""
run_go2.py — Interactive launcher for Dimos Go2 blueprints.

Usage:
    python run_go2.py

Requirements:
    - Python 3.8+ (stdlib only, no extra packages)
    - 'dimos' CLI installed (activate your venv first: source .venv/bin/activate)

Reads/writes .env in the repo root, validates API keys, and constructs the
correct `dimos [global-flags] run <blueprint>` command for you.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Environment bootstrap
# Runs before everything else. If 'dimos' is not in PATH:
#   1. Creates .venv with uv (falls back to python -m venv)
#   2. Installs dimos[unitree] which includes agents, web, perception, sim
#   3. Re-execs this script inside the new venv
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent
_VENV_DIR = _REPO_ROOT / ".venv"
_VENV_PYTHON = _VENV_DIR / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
)
_VENV_DIMOS = _VENV_DIR / (
    "Scripts/dimos.exe" if sys.platform == "win32" else "bin/dimos"
)
# unitree pulls in base → agents + web + perception + visualization + sim
_INSTALL_EXTRA = ".[unitree]"


def _bootstrap_env() -> None:
    """Ensure dimos is installed; create .venv and install if needed, then re-exec."""
    # Already available in the current Python / PATH — nothing to do.
    if shutil.which("dimos"):
        return

    # .venv exists with dimos installed: re-exec using venv python if not already in it.
    if _VENV_PYTHON.exists() and _VENV_DIMOS.exists():
        if Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
            os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
        return

    print("\n  dimos not found in PATH. Setting up the environment automatically.\n")

    # ── Step 1: create .venv ─────────────────────────────────────────────────
    if not _VENV_PYTHON.exists():
        use_uv = bool(shutil.which("uv"))
        print("  [1/2] Creating virtual environment (.venv) …")
        if use_uv:
            cmd = ["uv", "venv", "--python", "3.12", str(_VENV_DIR)]
        else:
            print("        (uv not found — using python -m venv, this is slower)")
            cmd = [sys.executable, "-m", "venv", str(_VENV_DIR)]
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            print("\n  ERROR: Failed to create virtual environment.")
            print("  Install uv for faster setup:  pip install uv")
            sys.exit(1)
        print("  Virtual environment created.\n")

    # ── Step 2: install dimos ────────────────────────────────────────────────
    use_uv = bool(shutil.which("uv"))
    print(f"  [2/2] Installing {_INSTALL_EXTRA}")
    print("        (includes agents, web, perception, simulation, Unitree WebRTC)")
    print("        This may take a few minutes on first run …\n")
    if use_uv:
        install_cmd = [
            "uv", "pip", "install",
            "--python", str(_VENV_PYTHON),
            "-e", _INSTALL_EXTRA,
        ]
    else:
        install_cmd = [str(_VENV_PYTHON), "-m", "pip", "install", "-e", _INSTALL_EXTRA]
    result = subprocess.run(install_cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        print("\n  ERROR: Installation failed. Check the output above.")
        print("  To fix manually:")
        print(f"    source {_VENV_DIR}/bin/activate")
        print(f"    pip install -e '{_INSTALL_EXTRA}'")
        sys.exit(1)

    print(f"\n  Installation complete. Restarting launcher …\n")
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Blueprint catalog
# Tuple: (cli_name, display_name, category, description, required_keys, needs_robot_ip)
# required_keys: list of env var names that must be set. Special value
#   "OLLAMA_RUNNING" checks for the ollama binary in PATH instead.
# ─────────────────────────────────────────────────────────────────────────────

BLUEPRINTS = [
    # ── Basic ────────────────────────────────────────────────────────────────
    (
        "unitree-go2-basic",
        "Go2 Basic",
        "Basic",
        "Minimal WebRTC connection + Rerun visualization. No navigation.",
        [],
        True,
    ),
    # ── Smart (navigation stack, no LLM) ─────────────────────────────────────
    (
        "unitree-go2",
        "Go2 Smart",
        "Smart",
        "Full navigation: 3D voxel map, costmap, A* planner, frontier exploration.",
        [],
        True,
    ),
    (
        "unitree-go2-spatial",
        "Go2 Spatial",
        "Smart",
        "Smart + CLIP spatial memory (ChromaDB). Remembers visited places.",
        [],
        True,
    ),
    (
        "unitree-go2-detection",
        "Go2 Detection",
        "Smart",
        "Smart + YOLO 2D/3D object detection pipeline.",
        [],
        True,
    ),
    (
        "unitree-go2-ros",
        "Go2 ROS Bridge",
        "Smart",
        "Smart + ROS2 transport bridge (PointCloud2, Image, PoseStamped).",
        [],
        True,
    ),
    (
        "unitree-go2-vlm-stream-test",
        "Go2 VLM Stream Test",
        "Smart",
        "Basic + VLM vision stream test using GPT-4o.",
        ["OPENAI_API_KEY"],
        True,
    ),
    # ── Agentic (full LLM agent) ──────────────────────────────────────────────
    (
        "unitree-go2-agentic",
        "Go2 Agentic  (GPT-4o)",
        "Agentic",
        "Full agent: GPT-4o, spatial nav, person follow, TTS speak, web UI (port 5555).",
        ["OPENAI_API_KEY"],
        True,
    ),
    (
        "unitree-go2-agentic-mcp",
        "Go2 Agentic + MCP",
        "Agentic",
        "Agentic + MCP TCP server on port 9990 (for Claude Code / external AI).",
        ["OPENAI_API_KEY"],
        True,
    ),
    (
        "unitree-go2-agentic-ollama",
        "Go2 Agentic  (Ollama local)",
        "Agentic",
        "Full agent via local Ollama qwen3:8b. No API key needed.",
        ["OLLAMA_RUNNING"],
        True,
    ),
    (
        "unitree-go2-agentic-huggingface",
        "Go2 Agentic  (HuggingFace)",
        "Agentic",
        "Full agent via HuggingFace Qwen2.5-1.5B-Instruct.",
        ["HUGGINGFACE_API_KEY"],
        True,
    ),
    (
        "unitree-go2-temporal-memory",
        "Go2 Temporal Memory",
        "Agentic",
        "Agentic + long-term episodic/temporal memory module.",
        ["OPENAI_API_KEY"],
        True,
    ),
    # ── Demo (no robot hardware needed) ──────────────────────────────────────
    (
        "demo-agent",
        "Demo Agent",
        "Demo",
        "LLM agent only — no robot hardware. Good for testing the ReAct loop.",
        ["OPENAI_API_KEY"],
        False,
    ),
    (
        "demo-agent-camera",
        "Demo Agent + Camera",
        "Demo",
        "LLM agent + webcam stream — no robot hardware.",
        ["OPENAI_API_KEY"],
        False,
    ),
]

# Descriptions shown when offering to set a missing key
KNOWN_KEYS = {
    "OPENAI_API_KEY": "OpenAI API key  (gpt-4o, Whisper TTS, speech output)",
    "HUGGINGFACE_API_KEY": "HuggingFace API key  (Hub models)",
    "GOOGLE_MAPS_API_KEY": "Google Maps API key  (GPS nav / where_am_i skill)",
    "ANTHROPIC_API_KEY": "Anthropic API key  (Claude models)",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — .env reader / writer
# ─────────────────────────────────────────────────────────────────────────────

ENV_PATH = Path(__file__).parent / ".env"


def read_env() -> dict[str, str]:
    """Parse .env file into a dict. Returns {} if file does not exist."""
    result: dict[str, str] = {}
    if not ENV_PATH.exists():
        return result
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        result[key] = val
    return result


def write_env_key(key: str, value: str) -> None:
    """Append or update a single KEY=VALUE in .env (preserves all other lines)."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    new_lines: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            # Quote value if it contains spaces
            safe_val = f'"{value}"' if " " in value else value
            new_lines.append(f"{key}={safe_val}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        safe_val = f'"{value}"' if " " in value else value
        new_lines.append(f"{key}={safe_val}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def get_robot_ip_from_env(env: dict[str, str]) -> str:
    """Check both ROBOT_IP and DIMOS_ROBOT_IP (pydantic-settings strips prefix)."""
    return env.get("ROBOT_IP") or env.get("DIMOS_ROBOT_IP") or ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Display helpers
# ─────────────────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def cyan(t: str) -> str:
    return _c(t, "36")


def _width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def print_header() -> None:
    w = _width()
    print("\n" + "═" * w)
    print(bold(cyan("  Dimos Go2 Launcher")))
    print(dim("  Interactive blueprint runner — reads/writes .env in repo root"))
    print("═" * w + "\n")


def print_menu(blueprints: list, env: dict) -> None:
    """Render the grouped blueprint menu with key-status indicators."""
    current_category: str | None = None
    idx = 1
    for cli_name, display_name, category, desc, req_keys, needs_ip in blueprints:
        if category != current_category:
            print(f"\n  {bold(cyan(f'── {category} ──'))}")
            current_category = category

        # Determine key status
        missing = _missing_keys(req_keys, env, {})
        if not req_keys:
            key_badge = dim("  [no keys needed]")
        elif missing:
            key_badge = red(f"  [missing: {', '.join(missing)}]")
        else:
            key_badge = green("  [keys OK]")

        ip_note = dim("  [needs IP]") if needs_ip else dim("  [no robot]")

        print(f"  {bold(str(idx)):>5}.  {display_name}{ip_note}")
        print(f"          {dim(desc)}{key_badge}")
        idx += 1
    print()


def print_separator() -> None:
    print(dim("─" * _width()))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Validation logic
# ─────────────────────────────────────────────────────────────────────────────


def _missing_keys(
    req_keys: list[str], env: dict[str, str], session_env: dict[str, str]
) -> list[str]:
    """Return list of required keys that are not yet set."""
    combined = {**env, **session_env}
    missing = []
    for key in req_keys:
        if key == "OLLAMA_RUNNING":
            if not shutil.which("ollama"):
                missing.append(key)
        elif not combined.get(key):
            missing.append(key)
    return missing


def validate_ip(ip: str) -> bool:
    """Accept dotted-quad IPv4 or any non-empty hostname."""
    if not ip:
        return False
    parts = ip.split(".")
    if len(parts) == 4:
        return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)
    return True  # hostname or other format


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Interactive prompt functions
# ─────────────────────────────────────────────────────────────────────────────


def _ask(prompt: str) -> str:
    """Wrapper around input() that handles KeyboardInterrupt gracefully."""
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{yellow('  Cancelled.')}")
        sys.exit(0)


def prompt_blueprint_choice(blueprints: list) -> tuple:
    """Show numbered menu and return the selected blueprint tuple."""
    while True:
        choice = _ask(bold("  Select blueprint number  [q to quit]: "))
        if choice.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(blueprints):
                return blueprints[n - 1]
        print(red(f"  Please enter a number between 1 and {len(blueprints)}."))


def prompt_robot_ip(env: dict, needs_ip: bool) -> str | None:
    """Prompt for the robot's IP address.

    Returns the IP string, or None if not needed / user skipped.
    """
    if not needs_ip:
        return None

    current = get_robot_ip_from_env(env)
    hint = f" {dim(f'[current: {current}]')}" if current else ""
    print(f"\n  {bold('Robot IP address')}{hint}")
    print(
        f"  {dim('Press Enter to keep current value, enter a new IP, or type skip to run without.')}"
    )

    raw = _ask(bold("  Robot IP: "))

    if raw.lower() == "skip":
        print(yellow("  Skipping IP — you can use --simulation or --replay mode."))
        return None
    if not raw:
        if current:
            return current
        print(yellow("  No IP set — proceeding without --robot-ip."))
        return None
    if validate_ip(raw):
        return raw
    print(red(f"  '{raw}' does not look like a valid IP. Ignoring."))
    return current or None


def prompt_missing_keys(
    missing: list[str], session_env: dict[str, str]
) -> dict[str, str]:
    """For each missing required key, offer to enter it (session or .env)."""
    additions: dict[str, str] = {}
    for key in missing:
        if key == "OLLAMA_RUNNING":
            print(f"\n  {red('Ollama not found in PATH.')}")
            print(f"  Install from  https://ollama.com/download")
            print(f"  Then run:  {dim('ollama serve')}  and  {dim('ollama pull qwen3:8b')}")
            _ask("  Press Enter to continue anyway (the blueprint may fail)...")
            continue

        desc = KNOWN_KEYS.get(key, key)
        print(f"\n  {red('Missing:')}  {bold(key)}")
        print(f"  {dim(desc)}")
        print(f"  {bold('[1]')} Enter value for this session only")
        print(f"  {bold('[2]')} Enter value and save to .env")
        print(f"  {bold('[3]')} Skip  (blueprint may fail at startup)")
        choice = _ask(bold("  Choice [1/2/3]: "))

        if choice in ("1", "2"):
            value = _ask(f"  Value for {bold(key)}: ")
            if value:
                additions[key] = value
                if choice == "2":
                    write_env_key(key, value)
                    print(green(f"  Saved {key} to .env"))
                else:
                    print(green(f"  Using {key} for this session only (not saved to .env)"))
    return additions


def prompt_mode() -> tuple[bool, bool]:
    """Ask whether to use real robot, simulation, or replay mode."""
    print(f"\n  {bold('Run mode')}")
    print(f"  {bold('[1]')} Real robot  (default — uses robot IP)")
    print(f"  {bold('[2]')} MuJoCo simulation  (--simulation, no hardware needed)")
    print(f"  {bold('[3]')} Replay recorded data  (--replay, downloads ~2.4 GB first time)")
    choice = _ask(bold("  Mode [1/2/3, Enter=1]: ")) or "1"
    simulation = choice == "2"
    replay = choice == "3"
    return simulation, replay


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Command builder
# ─────────────────────────────────────────────────────────────────────────────


def build_command(
    blueprint_name: str,
    robot_ip: str | None,
    simulation: bool,
    replay: bool,
    session_env: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Construct the `dimos [global-flags] run <blueprint>` command.

    IMPORTANT: Global flags (--robot-ip, --simulation, --replay) MUST come
    before the 'run' subcommand. This is because the dimos CLI uses a typer
    callback that sets GlobalConfig before subcommands execute.
    See: dimos/robot/cli/dimos.py lines 36-128
    """
    cmd = ["dimos"]

    if robot_ip:
        cmd += ["--robot-ip", robot_ip]
    if simulation:
        cmd.append("--simulation")
    if replay:
        cmd.append("--replay")

    cmd += ["run", blueprint_name]

    # Environment: inherit current shell env, add session-only keys on top.
    # The .env file is already loaded by dimos itself (load_dotenv() at CLI startup).
    extra_env = {**os.environ}
    extra_env.update(session_env)

    return cmd, extra_env


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — main()
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print_header()

    # 1. Load current .env
    env = read_env()
    session_env: dict[str, str] = {}

    current_ip = get_robot_ip_from_env(env)
    if current_ip:
        print(f"  {dim('Current robot IP from .env:')} {bold(current_ip)}")
    else:
        print(f"  {yellow('No robot IP set in .env.')} You can enter one below or use simulation mode.")

    # 2. Show menu and get blueprint selection
    print_menu(BLUEPRINTS, env)
    bp = prompt_blueprint_choice(BLUEPRINTS)
    cli_name, display_name, category, desc, req_keys, needs_ip = bp

    print(f"\n  {bold('Selected:')} {green(display_name)}")
    print(f"  {dim(desc)}")

    # 3. Robot IP
    robot_ip = prompt_robot_ip(env, needs_ip)

    # 4. API key validation and remediation
    missing = _missing_keys(req_keys, env, session_env)
    if req_keys and not missing:
        present_keys = [k for k in req_keys if k != "OLLAMA_RUNNING"]
        if present_keys:
            print(green(f"\n  Required keys present: {', '.join(present_keys)}"))
        if "OLLAMA_RUNNING" in req_keys and not missing:
            print(green("\n  Ollama found in PATH."))

    if missing:
        print(yellow(f"\n  Some required keys are missing: {', '.join(missing)}"))
        new_keys = prompt_missing_keys(missing, session_env)
        session_env.update(new_keys)

    # 5. Run mode (only for blueprints that talk to a robot)
    simulation, replay = False, False
    if needs_ip:
        simulation, replay = prompt_mode()

    # 6. Build command
    cmd, extra_env = build_command(cli_name, robot_ip, simulation, replay, session_env)
    cmd_str = " ".join(cmd)

    # Show summary
    print()
    print_separator()
    print(f"  {bold('Blueprint:')}  {display_name}")
    if robot_ip:
        print(f"  {bold('Robot IP:')}   {robot_ip}")
    elif needs_ip:
        print(f"  {bold('Robot IP:')}   {dim('(not set)')}")
    if simulation:
        print(f"  {bold('Mode:')}       MuJoCo simulation")
    elif replay:
        print(f"  {bold('Mode:')}       Replay recorded data")
    else:
        print(f"  {bold('Mode:')}       Real robot")
    print()
    print(f"  {bold('Command:')}")
    print(f"  {cyan(cmd_str)}")

    # Show session-only keys (masked)
    session_display = {k: v for k, v in session_env.items() if k not in os.environ}
    if session_display:
        print()
        print(f"  {bold('Session env vars (not saved to .env):')}")
        for k, v in session_display.items():
            masked = (v[:4] + "..." + v[-2:]) if len(v) > 8 else "***"
            print(f"    {dim(k)}={masked}")

    print_separator()
    print()

    # 7. Confirm and run
    confirm = _ask(bold("  Run now? [Y/n]: "))
    if confirm.lower() in ("n", "no"):
        print(f"\n  {yellow('Cancelled.')}  To run manually:\n  {cyan(cmd_str)}\n")
        return

    print(f"\n  {green(f'Starting {display_name}...')}")
    print(f"  {dim('Press Ctrl+C to stop.')}\n")

    try:
        subprocess.run(cmd, env=extra_env)
    except KeyboardInterrupt:
        print(f"\n  {yellow('Stopped by user.')}")
    except FileNotFoundError:
        print(red("\n  Error: 'dimos' command not found in PATH."))
        print("  Make sure your virtual environment is activated:")
        print(f"  {cyan('source .venv/bin/activate')}")
        print(f"\n  Then try again, or run directly:")
        print(f"  {cyan(cmd_str)}")
        sys.exit(1)


if __name__ == "__main__":
    _bootstrap_env()
    main()
