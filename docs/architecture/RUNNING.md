# Running Dimos from the Repo

> This guide is for running Dimos directly from a cloned copy of this repository — not from a pip-installed package.

---

## Table of Contents

1. [One-time Setup](#1-one-time-setup)
2. [API Keys and Environment Variables](#2-api-keys-and-environment-variables)
3. [Running Blueprints with the CLI](#3-running-blueprints-with-the-cli)
4. [Running Blueprints Directly in Python](#4-running-blueprints-directly-in-python)
5. [Running Without a Robot (Simulation & Demo)](#5-running-without-a-robot-simulation--demo)
6. [Talking to the Agent (humancli)](#6-talking-to-the-agent-humancli)
7. [Debugging Tools](#7-debugging-tools)
8. [Common Errors](#8-common-errors)

---

## 1. One-time Setup

### Prerequisites (macOS)

```bash
brew install gnu-sed gcc portaudio git-lfs libjpeg-turbo python pre-commit
```

### Prerequisites (Ubuntu 22.04 / 24.04)

```bash
sudo apt-get update
sudo apt-get install -y curl g++ portaudio19-dev git-lfs libturbojpeg python3-dev pre-commit
```

### Install `uv` (Python package manager)

Dimos uses `uv` instead of `pip` because it handles complex dependency groups correctly.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

### Clone and install (if you haven't already)

```bash
# Skip downloading large LFS files immediately (they'll download on-demand)
export GIT_LFS_SKIP_SMUDGE=1
git clone -b dev https://github.com/dimensionalOS/dimos.git
cd dimos
```

### Install all dependencies from the repo

This installs dimos in **editable mode** (`-e`) — meaning your source code changes take effect immediately without reinstalling.

```bash
# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install everything (agents, perception, simulation, hardware support, dev tools)
uv pip install -e '.[base,dev,manipulation,misc,unitree,drone]'
```

**What each extra includes:**
| Extra | What it adds |
|-------|-------------|
| `base` | agents + web + perception + visualization + sim |
| `agents` | LangChain, LangGraph, OpenAI, Ollama, Anthropic, MCP, Whisper |
| `perception` | YOLO11 (ultralytics), Moondream, HuggingFace Transformers |
| `web` | FastAPI server, SSE streaming, uvicorn |
| `sim` | MuJoCo simulator + Playground |
| `visualization` | Rerun SDK |
| `unitree` | WebRTC connection to Unitree Go2/G1/B1 |
| `manipulation` | Drake planning, xArm, Piper |
| `misc` | TensorZero, Cerebras, sentence-transformers, scikit-learn |
| `dev` | ruff, mypy, pytest, pre-commit |
| `cuda` | CUDA-accelerated inference (Linux x86 only) |

### Install pre-commit hooks (optional, recommended)

```bash
pre-commit install
```

---

## 2. API Keys and Environment Variables

Create a `.env` file in the repo root. Dimos automatically loads it at startup ([`dimos/robot/cli/dimos.py:33`](../../dimos/robot/cli/dimos.py)).

```bash
# .env — place this in the repo root (same directory as pyproject.toml)

# ── LLM Keys (pick at least one) ───────────────────────────────────────
OPENAI_API_KEY=sk-...          # Required for default gpt-4o agent + Whisper STT + TTS
ANTHROPIC_API_KEY=sk-ant-...   # Required if you use a Claude model
HUGGINGFACE_API_KEY=hf_...     # Required for HuggingFace Hub models

# ── Robot Connection ────────────────────────────────────────────────────
DIMOS_ROBOT_IP=192.168.123.161  # Your Unitree robot's IP address

# ── Optional: run in simulation instead of connecting to a robot ────────
DIMOS_SIMULATION=true

# ── Optional: maps / GPS skills ─────────────────────────────────────────
GOOGLE_MAPS_API_KEY=AIza...

# ── Optional: debug logging ─────────────────────────────────────────────
DIMOS_LOG_LEVEL=DEBUG
RERUN_SAVE=1    # saves rerun visualization data to rerun.json
```

**Important:** All `DIMOS_` prefixed variables map directly to fields in [`GlobalConfig`](../../dimos/core/global_config.py). CLI flags override `.env` values, which override defaults.

For Ollama (local models), **no API key is needed** — just install and run Ollama:
```bash
# Install Ollama: https://ollama.com/download
ollama serve         # start the daemon
ollama pull qwen3:8b # download the model (first time only)
```

---

## 3. Running Blueprints with the CLI

The `dimos` CLI is the main way to launch the system. It is installed as a script by `pip install -e .` and maps to [`dimos/robot/cli/dimos.py`](../../dimos/robot/cli/dimos.py).

### List all available blueprints

```bash
dimos list
```

### Run a blueprint

```bash
dimos run <blueprint-name>
```

### Common blueprint examples

```bash
# Full Go2 navigation stack (requires DIMOS_ROBOT_IP)
dimos run unitree-go2

# LLM agent with GPT-4o (requires OPENAI_API_KEY + DIMOS_ROBOT_IP)
dimos run unitree-go2-agentic

# LLM agent with local Ollama (requires Ollama running with qwen3:8b)
dimos run unitree-go2-agentic-ollama

# LLM agent + MCP server on port 9990 (for Claude Code integration)
dimos run unitree-go2-agentic-mcp

# Standalone agent demo (no robot hardware needed, just needs OPENAI_API_KEY)
dimos run demo-agent

# Standalone agent demo with webcam
dimos run demo-agent-camera
```

### Pass config via CLI flags

CLI flags override the `.env` file and defaults:

```bash
# Connect to a specific robot IP
dimos --robot-ip 192.168.123.161 run unitree-go2

# Run in MuJoCo simulation (no real robot)
dimos --simulation run unitree-go2

# Replay a recorded session (downloads ~2.4 GB of data on first run via LFS)
dimos --replay run unitree-go2

# Show current config without running
dimos show-config
```

### All GlobalConfig flags

These map to fields in [`dimos/core/global_config.py`](../../dimos/core/global_config.py):

| Flag | Default | Description |
|------|---------|-------------|
| `--robot-ip` | `None` | IP of the physical robot |
| `--simulation / --no-simulation` | `false` | Use MuJoCo instead of real robot |
| `--replay / --no-replay` | `false` | Replay a recorded session |
| `--viewer-backend` | `rerun-web` | Visualization: `rerun`, `rerun-web`, `foxglove`, `none` |
| `--n-dask-workers` | `2` | Number of Dask parallel workers |

---

## 4. Running Blueprints Directly in Python

You don't need the CLI. Any blueprint can be run as a plain Python script:

```python
# my_run.py — run this with: python my_run.py

from dimos.robot.unitree.go2.blueprints.agentic.unitree_go2_agentic import unitree_go2_agentic

if __name__ == "__main__":
    unitree_go2_agentic.build().loop()
```

```bash
python my_run.py
```

You can also override config at build time:

```python
from dimos.agents.agent import Agent
from dimos.agents.demo_agent import demo_agent

# Run the demo agent with Claude instead of GPT-4o
demo_with_claude = demo_agent  # agent defaults to gpt-4o in demo_agent.py

# Or build a custom agent with explicit model
from dimos.core.blueprints import autoconnect
from dimos.agents.agent import Agent

my_agent = autoconnect(Agent.blueprint(model="ollama:qwen3:8b"))

if __name__ == "__main__":
    my_agent.build().loop()
```

---

## 5. Running Without a Robot (Simulation & Demo)

### Option A: MuJoCo simulation (full navigation stack, simulated robot)

```bash
# Needs dimos[sim] extra
dimos --simulation run unitree-go2
```

The simulated Go2 runs in MuJoCo with the full navigation stack. Opens Rerun visualization at `localhost:9090` (web viewer).

### Option B: Replay a recorded session

```bash
dimos --replay run unitree-go2
```

On first run, Git LFS downloads ~2.4 GB of recorded LiDAR + video data. Good for exploring the perception and navigation stack without hardware.

### Option C: Demo agent (LLM only, no robot at all)

```bash
dimos run demo-agent
```

This runs just the `Agent` module ([`dimos/agents/demo_agent.py`](../../dimos/agents/demo_agent.py)) — you can send messages to it and see the ReAct loop in action. Needs `OPENAI_API_KEY` (or another model).

```bash
# With a webcam attached:
dimos run demo-agent-camera
```

### Option D: Agentic simulation

```bash
export OPENAI_API_KEY=sk-...
dimos --simulation run unitree-go2-agentic
```

Full agent + MuJoCo simulated robot. You can give natural language commands and see the robot move in simulation.

---

## 6. Talking to the Agent (humancli)

Once a blueprint with an agent is running, use `humancli` to send text commands:

```bash
# In a second terminal
humancli
```

The CLI connects to the `WebInput` module which is listening on port **5555**. Type your command and press Enter:

```
> explore the space
> follow the person in the red shirt
> go to the kitchen
> what do you see in front of you?
```

The agent's responses are also printed here (and spoken via TTS if configured).

**Source:** [`dimos/utils/cli/human/humanclianim.py`](../../dimos/utils/cli/human/humanclianim.py)

---

## 7. Debugging Tools

### Enable debug logging

```bash
DIMOS_LOG_LEVEL=DEBUG dimos run unitree-go2
```

### Monitor LCM messages (inter-module bus)

```bash
# In a second terminal — shows all messages on the LCM bus in real time
dimos lcmspy

# Or via the standalone command:
lcmspy
```

### Monitor agent reasoning

```bash
# Shows the agent's ReAct chain (which tools it is calling and why)
dimos agentspy

# Or:
agentspy
```

### Rerun visualization

Rerun is the 3D visualizer. By default, it opens as a web viewer at `http://localhost:9090`:

```bash
dimos run unitree-go2
# then open http://localhost:9090 in your browser
```

To save a recording:
```bash
RERUN_SAVE=1 dimos run unitree-go2
# saves rerun.json in the current directory
```

To launch Rerun separately:
```bash
dimos rerun-bridge --viewer-mode web
# or: native (desktop app), none (headless)
```

---

## 8. Common Errors

### `OPENAI_API_KEY not set` / `AuthenticationError`

Add your key to `.env`:
```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

Or export it directly:
```bash
export OPENAI_API_KEY=sk-...
dimos run unitree-go2-agentic
```

### `Cannot connect to Ollama daemon`

Ollama is not running. Start it:
```bash
ollama serve
```

### `ModuleNotFoundError: No module named 'langchain'`

The `agents` extra is not installed. Run:
```bash
uv pip install -e '.[base,dev,manipulation,misc,unitree,drone]'
```

### `Cannot connect to robot at 192.168.x.x`

- Make sure your computer is on the same network as the robot
- Export the correct IP: `export DIMOS_ROBOT_IP=<actual_ip>`
- Or use simulation: `dimos --simulation run unitree-go2`

### `dimos: command not found`

The virtual environment is not activated. Run:
```bash
source .venv/bin/activate
```

### `uv.lock` conflict / dependency errors

```bash
uv sync --all-extras --no-extra dds
```

This syncs all extras (except DDS which has extra system requirements) from the lockfile.
