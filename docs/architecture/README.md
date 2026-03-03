# Dimos Architecture Guide

> **For software engineering students.** This document explains the `dimos` (Dimensional OS) codebase — an AI-native robotics operating system. The goal is to help you understand how the major subsystems fit together before you start reading the source code.
>
> File paths are relative to the repo root (`/dimos/`). Click any path to navigate directly to the source.

---

## Table of Contents

1. [What is Dimos?](#1-what-is-dimos)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [LLM, Agent & MCP Integration](#3-llm-agent--mcp-integration)
4. [Object Detection & Perception Pipeline](#4-object-detection--perception-pipeline)
5. [Key Concepts Glossary](#5-key-concepts-glossary)
6. [File Map](#6-file-map)

---

## 1. What is Dimos?

**Dimos** is a Python SDK and runtime environment for controlling generalist robots (humanoids, quadrupeds, drones, arms) **without requiring ROS** (Robot Operating System). The design philosophy is:

- **Agent-native**: robots are controlled by LLM agents that reason about goals, not just execute pre-programmed behaviors.
- **Modular streams**: every robot subsystem is a `Module` that publishes and subscribes to typed reactive data streams.
- **No-ROS by default**: communication uses LCM (Low-Latency Communications) instead of ROS2, so installation is simpler — but ROS2 is supported as an optional transport.
- **Python-first**: most logic is pure Python (with optional C++ extensions via pybind11 for performance-critical code like the A\* replanner).

**Supported hardware** (as of v0.0.10): Unitree Go2 / G1 / B1 robots, xArm and Piper manipulators, DJI drones, Mavlink vehicles, ZED / RealSense / Livox sensors.

---

## 2. System Architecture Overview

The system is organized into **5 horizontal layers**, each building on the one below it:

| Layer | Role | Key source directory |
|-------|------|---------------------|
| **L1 — Human Input** | Text, voice, teleoperation, or external AI clients | [`dimos/agents/web_human_input.py`](../../dimos/agents/web_human_input.py), [`dimos/protocol/mcp/`](../../dimos/protocol/mcp/) |
| **L2 — Agent / Brain** | LLM-driven decision loop + vision-language queries | [`dimos/agents/agent.py`](../../dimos/agents/agent.py), [`dimos/agents/vlm_agent.py`](../../dimos/agents/vlm_agent.py) |
| **L3 — Skills, Perception & Memory** | Robot skills (callable by agent), sensor processing, spatial memory | [`dimos/agents/skills/`](../../dimos/agents/skills/), [`dimos/perception/`](../../dimos/perception/), [`dimos/navigation/`](../../dimos/navigation/) |
| **L4 — Stream / Transport** | Message-passing between modules | [`dimos/core/transport.py`](../../dimos/core/transport.py), [`dimos/rxpy_backpressure/`](../../dimos/rxpy_backpressure/) |
| **L5 — Hardware & Sensors** | Physical robot connections | [`dimos/robot/unitree/`](../../dimos/robot/unitree/), [`dimos/hardware/sensors/`](../../dimos/hardware/sensors/) |

![System Architecture Overview](./dimos_system_overview.excalidraw)

### How modules are wired together

The core concept is the **Blueprint**. A Blueprint is a recipe that says "create these Module instances and connect their streams automatically". The `autoconnect()` function ([`dimos/core/blueprints.py`](../../dimos/core/blueprints.py)) wires modules together by matching stream **types** and **names**: if module A has `Out[Image]` named `"color_image"` and module B has `In[Image]` named `"color_image"`, they get connected automatically.

```python
# Example Blueprint — from dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic.py
unitree_go2_agentic = autoconnect(
    unitree_go2_spatial,    # navigation + spatial memory
    agent(),                # LLM agent (defaults to gpt-4o)
    _common_agentic,        # skills: navigate, follow person, speak, web_input
)
```

Every `Module` ([`dimos/core/module.py`](../../dimos/core/module.py)) communicates using `In[T]` (input stream) and `Out[T]` (output stream) annotations, which are RxPy Observables under the hood.

---

## 3. LLM, Agent & MCP Integration

This is the heart of Dimos — how a language model controls a physical robot.

![LLM, Agent & MCP Integration](./dimos_llm_mcp_integration.excalidraw)

---

### 3.1 Supported LLM Models

Dimos uses **LangChain** as the model abstraction layer. The `Agent` and `VLMAgent` modules each take a `model` string in their config. The model string format determines which backend is used:

| Format | Backend | Example | Requires |
|--------|---------|---------|----------|
| `"gpt-4o"` | OpenAI API (**default**) | `agent()` | `OPENAI_API_KEY` |
| `"gpt-4o-mini"` | OpenAI API | `agent(model="gpt-4o-mini")` | `OPENAI_API_KEY` |
| `"ollama:<model>"` | Local Ollama | `agent(model="ollama:qwen3:8b")` | Ollama daemon running |
| `"huggingface:<model>"` | HuggingFace Hub | `agent(model="huggingface:Qwen/Qwen2.5-1.5B-Instruct")` | `HUGGINGFACE_API_KEY` |
| Any LangChain model ID | via `init_chat_model()` | `agent(model="anthropic/claude-opus-4-6")` | Provider API key |

**Installed LLM packages** (from [`pyproject.toml`](../../pyproject.toml) `[agents]` extra):
- `langchain-openai` — OpenAI / Azure OpenAI
- `langchain-ollama` — local models via Ollama (Llama, Qwen, Mistral, etc.)
- `langchain-huggingface` — HuggingFace Hub models
- `anthropic` — Anthropic Claude models (via LangChain)
- `openai` — also used for Whisper STT and TTS
- `cerebras-cloud-sdk` — Cerebras inference (fast for smaller models, in `[misc]` extra)
- `tensorzero` — routing/optimization layer (`[misc]` extra)

**Vision-Language Models (VLMAgent):**
The `VLMAgent` ([`dimos/agents/vlm_agent.py`](../../dimos/agents/vlm_agent.py)) uses `init_chat_model()` so it supports the same model strings. The camera frame is encoded as a base64 image and sent alongside the text query. Examples in use:
- `"gpt-4o"` (default) — best vision quality
- `"ollama:llava"` — fully local vision
- `"moondream"` — tiny, fast, runs on-device (installed via `moondream` package in `[perception]` extra)

**Where model strings are resolved** — [`dimos/agents/agent.py:96-112`](../../dimos/agents/agent.py):
```python
# agent.py — on_system_modules()
if self.config.model.startswith("ollama:"):
    from dimos.agents.ollama_agent import ensure_ollama_model
    ensure_ollama_model(self.config.model.removeprefix("ollama:"))

model: str | BaseChatModel = self.config.model
# ... LangGraph create_agent() is then called with this model string
self._state_graph = create_agent(
    model=model,
    tools=_get_tools_from_modules(self, modules, self.rpc),
    system_prompt=self.config.system_prompt,
)
```

**Default model config** — [`dimos/agents/agent.py:43-46`](../../dimos/agents/agent.py):
```python
@dataclass
class AgentConfig(ModuleConfig):
    system_prompt: str | None = SYSTEM_PROMPT
    model: str = "gpt-4o"      # ← change this to switch models
    model_fixture: str | None = None   # for testing (mock model)
```

---

### 3.2 The Agent Module — ReAct Loop

**File:** [`dimos/agents/agent.py`](../../dimos/agents/agent.py)

The `Agent` class is the central brain. It runs a **ReAct loop** (Reason + Act) implemented with **LangGraph**:

```
Human text input
       ↓
   [human_input: In[str]]  ← stream from WebInput, humancli, or MCP
       ↓
  _message_queue (Thread-safe Queue)
       ↓
  _thread_loop()  ← runs in a background thread
       ↓
  LangGraph state_graph.stream({"messages": history})
       ├── LLM node: model reasons about which tool to call
       ├── Tool node: tool is executed via RPC (calls @skill method on robot)
       └── LLM node: model observes result and decides next step
       ↓
  [agent: Out[BaseMessage]]  ← response published to other modules
```

The full lifecycle of a command:

```
User: "follow the person in the red shirt"
    ↓ WebInput (port 5555) → human_input stream
    ↓ Agent._message_queue.put(HumanMessage(...))
    ↓ LangGraph: LLM calls person_follow_skill(color="red")
    ↓ _skill_to_tool() → RpcCall → person_follow_skill.run()
    ↓ Robot starts following the detected person
    ↓ Agent publishes AIMessage("Now following the person in the red shirt.")
```

Key methods in `agent.py`:

| Method | Line | Purpose |
|--------|------|---------|
| `on_system_modules()` | ~93 | Called at startup — creates the LangGraph, discovers all @skill tools |
| `_get_tools_from_modules()` | ~154 | Scans all connected modules for @skill methods, wraps them as LangChain StructuredTools |
| `_skill_to_tool()` | ~161 | Converts a SkillInfo into a LangChain tool backed by an RPC call |
| `_thread_loop()` | ~119 | Background thread that processes the message queue |
| `_process_message()` | ~131 | Runs one message through the LangGraph and publishes results |

---

### 3.3 The `@skill` Decorator

**File:** [`dimos/agents/annotation.py`](../../dimos/agents/annotation.py)

```python
@skill
def navigate_with_text(self, text: str) -> str:
    """Navigate to a location described in natural language."""
    ...
```

The `@skill` decorator sets `__skill__ = True` and `__rpc__ = True` on the function. This makes the method:
1. **Discoverable**: `_get_tools_from_modules()` finds it at startup via attribute inspection
2. **LLM-callable**: wrapped as a LangChain `StructuredTool` — the tool's JSON schema is derived from the function's Python type hints and docstring
3. **RPC-callable**: invocable remotely from other processes/modules (MCP server uses this too)

**Where skills live:**
| Skill | File |
|-------|------|
| Navigation (go to place, explore) | [`dimos/agents/skills/navigation.py`](../../dimos/agents/skills/navigation.py) |
| Person follow | [`dimos/agents/skills/person_follow.py`](../../dimos/agents/skills/person_follow.py) |
| Speak (TTS) | [`dimos/agents/skills/speak_skill.py`](../../dimos/agents/skills/speak_skill.py) |
| GPS navigation | [`dimos/agents/skills/gps_nav_skill.py`](../../dimos/agents/skills/gps_nav_skill.py) |
| Google Maps | [`dimos/agents/skills/google_maps_skill_container.py`](../../dimos/agents/skills/google_maps_skill_container.py) |
| Unitree robot actions (flip, sit, etc.) | [`dimos/robot/unitree/unitree_skill_container.py`](../../dimos/robot/unitree/unitree_skill_container.py) |
| G1 humanoid skills | [`dimos/robot/unitree/g1/skill_container.py`](../../dimos/robot/unitree/g1/skill_container.py) |

---

### 3.4 The System Prompt ("Daneel")

**File:** [`dimos/agents/system_prompt.py`](../../dimos/agents/system_prompt.py)

Every agent is loaded with a system prompt that defines the **"Daneel"** persona — a set of instructions that define:
- Safety rules (never harm humans or damage the robot)
- Identity (it is "Daneel", an AI agent by Dimensional)
- Communication style (concise, speaks via `speak` skill since users hear via speakers)
- Navigation flow (use `navigate_with_text` for most movement, tag important locations)
- Behavior rules (proactive inference, delivery/pickup protocols)

You can override the system prompt by passing `system_prompt=...` when constructing the agent.

---

### 3.5 The MCP Protocol

**File:** [`dimos/protocol/mcp/mcp.py`](../../dimos/protocol/mcp/mcp.py)

MCP (Model Context Protocol) is a standardized JSON-RPC protocol for **external AI clients** (like Claude Code) to discover and call tools on a running Dimos system. Dimos starts an MCP TCP server on port **9990**.

```
External AI client (e.g. Claude Code)
       ↓ connects to TCP :9990
MCPModule (dimos/protocol/mcp/mcp.py)
       ↓ tools/list → JSON schemas of all @skill methods
       ↓ tools/call → executes skill via RPC, returns result
```

Components:
| File | Role |
|------|------|
| [`dimos/protocol/mcp/mcp.py`](../../dimos/protocol/mcp/mcp.py) | asyncio TCP server, MCP protocol handler |
| [`dimos/protocol/mcp/bridge.py`](../../dimos/protocol/mcp/bridge.py) | stdin/stdout ↔ TCP adapter (lets Claude Code use it as a subprocess MCP server) |
| [`dimos/protocol/mcp/__main__.py`](../../dimos/protocol/mcp/__main__.py) | Entry point for `python -m dimos.protocol.mcp` |

**Blueprint that combines Agent + MCP:**
[`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py`](../../dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py)

---

### 3.6 The VLMAgent (Vision Queries)

**File:** [`dimos/agents/vlm_agent.py`](../../dimos/agents/vlm_agent.py)

For visual queries, there is a separate `VLMAgent` module:

```
[color_image: In[Image]]  ← camera stream (latest frame buffered)
[query_stream: In[HumanMessage]]  ← text question
       ↓
  _invoke_image():
      content = [{"type": "text", "text": query}, *image.agent_encode()]
      self._llm.invoke([system_message, HumanMessage(content)])
       ↓
[answer_stream: Out[AIMessage]]  ← model's answer
```

The `image.agent_encode()` method encodes the numpy image as a base64 JPEG and formats it as a LangChain multimodal content block.

Config defaults to `model="gpt-4o"` — any model that supports image inputs works (GPT-4o, Claude, Ollama llava, etc.).

---

### 3.7 Audio Pipeline (STT / TTS)

**WebInput** module ([`dimos/agents/web_human_input.py`](../../dimos/agents/web_human_input.py)) serves a web UI on port **5555**. It accepts both text and audio input.

| Component | File | Model |
|-----------|------|-------|
| Speech-to-Text | [`dimos/stream/audio/stt/node_whisper.py`](../../dimos/stream/audio/stt/node_whisper.py) | **OpenAI Whisper** (local, via `openai-whisper` package) |
| Text-to-Speech | [`dimos/agents/skills/speak_skill.py`](../../dimos/agents/skills/speak_skill.py) | **OpenAI TTS** API or `pyttsx3` (local fallback) |

The audio flow: microphone → WebInput → Whisper STT → text → Agent → response text → TTS → speaker.

---

### 3.8 LangChain / LangGraph Dependency Summary

All agent dependencies are in the `[agents]` extra in [`pyproject.toml`](../../pyproject.toml):

```
langchain==1.2.3
langchain-core==1.2.3
langchain-openai>=1,<2      ← OpenAI / Azure
langchain-ollama>=1,<2      ← local models
langchain-huggingface>=1,<2 ← HuggingFace Hub
langchain-chroma>=1,<2      ← vector store for memory
anthropic>=0.19.0           ← Claude API (used via langchain)
ollama>=0.6.0               ← local Ollama client
mcp>=1.0.0                  ← MCP server SDK
```

LangGraph (the ReAct state machine) is pulled in transitively by `langchain`. The `create_agent()` call in `agent.py:108` uses LangGraph's prebuilt `react_agent` under the hood.

---

## 4. Object Detection & Perception Pipeline

This section explains **how the robot "sees"** — how raw camera frames become actionable robot commands.

![Object Detection & Perception Pipeline](./dimos_perception_pipeline.excalidraw)

### Stage-by-stage walkthrough

**Stage 1 — Camera Input**
[`dimos/hardware/sensors/camera/module.py`](../../dimos/hardware/sensors/camera/module.py) wraps any camera (ZED stereo, RealSense depth, Webcam, GStreamer, or the built-in Unitree camera via WebRTC). It emits `Out[Image]` — a continuous stream of BGR frames.

**Stage 2 — Stream Processing**
Raw frames flow through `VideoOperators.with_fps_sampling(fps=10)` ([`dimos/stream/video_operators.py`](../../dimos/stream/video_operators.py)) which limits the frame rate. The **BackPressure** system ([`dimos/rxpy_backpressure/backpressure.py`](../../dimos/rxpy_backpressure/backpressure.py)) drops stale frames if the GPU falls behind, preventing queue buildup.

**Stage 3 — 2D Object Detection (YOLO)**
[`dimos/perception/detection/module2D.py`](../../dimos/perception/detection/module2D.py) runs each frame through a `Yolo2DDetector` ([`dimos/perception/detection/detectors/yolo.py`](../../dimos/perception/detection/detectors/yolo.py)) — **YOLO11** via the `ultralytics` package. Outputs `ImageDetections2D` — bounding boxes, class names, confidence scores. GPU (CUDA) is used if available; falls back to CPU.

**Stage 4 — 3D Projection & Fusion**
[`dimos/perception/detection/module3D.py`](../../dimos/perception/detection/module3D.py) takes the 2D bounding boxes **and** a `PointCloud2` from LiDAR. Projects each box onto the 3D point cloud to estimate the object's real-world position (x, y, z). Output: `ImageDetections3DPC` with 3D bounding volumes.

**Stage 5 — Tracking & Re-Identification**
`PersonTracker` ([`dimos/perception/detection/person_tracker.py`](../../dimos/perception/detection/person_tracker.py)) assigns consistent IDs across frames. The `ReidModule` ([`dimos/perception/detection/reid/module.py`](../../dimos/perception/detection/reid/module.py)) uses **TorchReid** to generate visual embedding vectors — if a person leaves the frame and returns, they are re-identified by comparing embeddings, not just bounding box overlap.

**Stage 6 — Spatial Reasoning & Memory**
`ObjectDB` ([`dimos/perception/detection/moduleDB.py`](../../dimos/perception/detection/moduleDB.py)) maintains a persistent database of seen objects with world-frame positions. `SpatioTemporalRAG` allows querying like "where did I last see the blue mug?" — the agent can use this as a `@skill`.

**Stage 7 — Navigation Response**
`DetectionNavigation.compute_twist_for_detection_3d()` ([`dimos/navigation/visual_servoing/detection_navigation.py`](../../dimos/navigation/visual_servoing/detection_navigation.py)) converts a 3D object position into a `Twist` command:
- `linear.x` = forward speed (approach target, default distance 1.5 m)
- `angular.z` = turn rate (steer toward the object's horizontal offset from center)

**Stage 8 — Control Arbitration**
`ControlCoordinator` ([`dimos/control/coordinator.py`](../../dimos/control/coordinator.py)) runs a deterministic tick loop: reads all pending motion commands, **arbitrates** conflicts (e.g., "follow person" vs. "avoid obstacle"), and routes the winning command to hardware.

**Stage 9 — Robot Actuation**
`GO2Connection.move(twist, duration)` ([`dimos/robot/unitree/go2/connection.py`](../../dimos/robot/unitree/go2/connection.py)) sends the final velocity command to the Unitree Go2 over WebRTC.

---

## 5. Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **Module** | A reusable robot subsystem. Defined by `class MyModule(Module)`. Communicates via `In[T]` / `Out[T]` stream annotations. Source: [`dimos/core/module.py`](../../dimos/core/module.py) |
| **Blueprint** | A function or composition that creates and wires multiple Modules together. `autoconnect()` matches streams by name+type. Source: [`dimos/core/blueprints.py`](../../dimos/core/blueprints.py) |
| **In[T] / Out[T]** | Typed stream I/O. `In[Image]` is a subscriber; `Out[Image]` is a publisher. Backed by RxPy Observables. Source: [`dimos/core/stream.py`](../../dimos/core/stream.py) |
| **@skill** | Decorator that marks a method as LLM-callable and RPC-callable. Source: [`dimos/agents/annotation.py`](../../dimos/agents/annotation.py) |
| **LCM** | Lightweight Communications and Marshalling — fast message bus, default transport between modules. |
| **RxPy Observable** | An asynchronous event stream (ReactiveX). Modules subscribe to input streams and react to incoming data without polling. |
| **BackPressure** | Mechanism to handle overloaded queues — drop stale items or keep only the latest. Source: [`dimos/rxpy_backpressure/backpressure.py`](../../dimos/rxpy_backpressure/backpressure.py) |
| **ReAct Loop** | Reason + Act — LLM reasons, calls a tool (action), observes result, reasons again. Implemented via LangGraph in [`dimos/agents/agent.py`](../../dimos/agents/agent.py) |
| **MCP** | Model Context Protocol — JSON-RPC protocol for external AI clients to invoke tools. TCP port 9990. Source: [`dimos/protocol/mcp/mcp.py`](../../dimos/protocol/mcp/mcp.py) |
| **VLM** | Vision-Language Model — accepts images + text (e.g., GPT-4o Vision). Source: [`dimos/agents/vlm_agent.py`](../../dimos/agents/vlm_agent.py) |
| **SLAM** | Simultaneous Localization and Mapping — robot builds a map while tracking its position. Implemented via FastLIO2. |
| **Twist** | Velocity command: `linear.x` (forward), `linear.y` (sideways), `angular.z` (rotation). |
| **PointCloud2** | 3D sensor message where each point is an x,y,z coordinate, from LiDAR or depth camera. |
| **TorchReid** | Deep learning library for person re-identification across camera frames using visual embeddings. |
| **RPC** | Remote Procedure Call — calling a function on another module. Used to invoke skills from the agent. Source: [`dimos/core/core.py`](../../dimos/core/core.py) |
| **GlobalConfig** | Shared runtime config (robot IP, simulation mode, etc.). Loaded from `.env`, env vars, or CLI flags. Source: [`dimos/core/global_config.py`](../../dimos/core/global_config.py) |

---

## 6. File Map

Quick reference for the most important files:

### Core Framework
| Component | File |
|-----------|------|
| Module base class | [`dimos/core/module.py`](../../dimos/core/module.py) |
| Blueprint / autoconnect | [`dimos/core/blueprints.py`](../../dimos/core/blueprints.py) |
| Transport layer (LCM, ROS2, SHM) | [`dimos/core/transport.py`](../../dimos/core/transport.py) |
| In[T] / Out[T] stream types | [`dimos/core/stream.py`](../../dimos/core/stream.py) |
| Global config (robot_ip, simulation, etc.) | [`dimos/core/global_config.py`](../../dimos/core/global_config.py) |
| RPC decorator + registry | [`dimos/core/core.py`](../../dimos/core/core.py) |

### Agent & LLM
| Component | File |
|-----------|------|
| Agent Module (ReAct loop, LangGraph) | [`dimos/agents/agent.py`](../../dimos/agents/agent.py) |
| VLMAgent (vision + text queries) | [`dimos/agents/vlm_agent.py`](../../dimos/agents/vlm_agent.py) |
| @skill decorator | [`dimos/agents/annotation.py`](../../dimos/agents/annotation.py) |
| System prompt ("Daneel" persona) | [`dimos/agents/system_prompt.py`](../../dimos/agents/system_prompt.py) |
| Web input UI (port 5555) | [`dimos/agents/web_human_input.py`](../../dimos/agents/web_human_input.py) |
| Ollama local model helper | [`dimos/agents/ollama_agent.py`](../../dimos/agents/ollama_agent.py) |
| Standalone demo agent (no robot) | [`dimos/agents/demo_agent.py`](../../dimos/agents/demo_agent.py) |

### Skills (callable by LLM)
| Skill | File |
|-------|------|
| Navigation (go to place, explore, tag) | [`dimos/agents/skills/navigation.py`](../../dimos/agents/skills/navigation.py) |
| Person follow | [`dimos/agents/skills/person_follow.py`](../../dimos/agents/skills/person_follow.py) |
| Speak (TTS output) | [`dimos/agents/skills/speak_skill.py`](../../dimos/agents/skills/speak_skill.py) |
| GPS navigation | [`dimos/agents/skills/gps_nav_skill.py`](../../dimos/agents/skills/gps_nav_skill.py) |
| Google Maps integration | [`dimos/agents/skills/google_maps_skill_container.py`](../../dimos/agents/skills/google_maps_skill_container.py) |
| Unitree robot actions (flip, sit, etc.) | [`dimos/robot/unitree/unitree_skill_container.py`](../../dimos/robot/unitree/unitree_skill_container.py) |

### MCP Protocol
| Component | File |
|-----------|------|
| MCPModule (asyncio TCP server :9990) | [`dimos/protocol/mcp/mcp.py`](../../dimos/protocol/mcp/mcp.py) |
| MCP Bridge (stdin/stdout ↔ TCP) | [`dimos/protocol/mcp/bridge.py`](../../dimos/protocol/mcp/bridge.py) |
| MCP module entry point | [`dimos/protocol/mcp/__main__.py`](../../dimos/protocol/mcp/__main__.py) |
| Go2 agentic + MCP blueprint | [`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py`](../../dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py) |

### Perception & Detection
| Component | File |
|-----------|------|
| Detection2DModule (YOLO11) | [`dimos/perception/detection/module2D.py`](../../dimos/perception/detection/module2D.py) |
| Detection3DModule (YOLO + LiDAR fusion) | [`dimos/perception/detection/module3D.py`](../../dimos/perception/detection/module3D.py) |
| YOLO11 detector (ultralytics) | [`dimos/perception/detection/detectors/yolo.py`](../../dimos/perception/detection/detectors/yolo.py) |
| Person re-identification (TorchReid) | [`dimos/perception/detection/reid/module.py`](../../dimos/perception/detection/reid/module.py) |
| Object database (world-frame positions) | [`dimos/perception/detection/moduleDB.py`](../../dimos/perception/detection/moduleDB.py) |
| Person tracker (multi-frame IDs) | [`dimos/perception/detection/person_tracker.py`](../../dimos/perception/detection/person_tracker.py) |
| Spatial perception (semantic map) | [`dimos/perception/spatial_perception.py`](../../dimos/perception/spatial_perception.py) |

### Navigation & Control
| Component | File |
|-----------|------|
| Detection → velocity command | [`dimos/navigation/visual_servoing/detection_navigation.py`](../../dimos/navigation/visual_servoing/detection_navigation.py) |
| Control coordinator (arbitration tick loop) | [`dimos/control/coordinator.py`](../../dimos/control/coordinator.py) |
| Velocity task | [`dimos/control/tasks/velocity_task.py`](../../dimos/control/tasks/velocity_task.py) |
| A* replanner | [`dimos/navigation/replanning_a_star/module.py`](../../dimos/navigation/replanning_a_star/module.py) |
| Wavefront frontier explorer | [`dimos/navigation/frontier_exploration/wavefront_frontier_goal_selector.py`](../../dimos/navigation/frontier_exploration/wavefront_frontier_goal_selector.py) |

### Robot Hardware
| Component | File |
|-----------|------|
| Unitree base connection | [`dimos/robot/unitree/connection.py`](../../dimos/robot/unitree/connection.py) |
| Go2 connection (WebRTC) | [`dimos/robot/unitree/go2/connection.py`](../../dimos/robot/unitree/go2/connection.py) |
| G1 humanoid connection | [`dimos/robot/unitree/g1/connection.py`](../../dimos/robot/unitree/g1/connection.py) |
| Camera module (generic) | [`dimos/hardware/sensors/camera/module.py`](../../dimos/hardware/sensors/camera/module.py) |
| ZED stereo camera | [`dimos/hardware/sensors/camera/zed/camera.py`](../../dimos/hardware/sensors/camera/zed/camera.py) |
| LiDAR (Livox Mid-360) | [`dimos/hardware/sensors/lidar/livox/module.py`](../../dimos/hardware/sensors/lidar/livox/module.py) |

### Stream Processing
| Component | File |
|-----------|------|
| Video operators (FPS, encoding) | [`dimos/stream/video_operators.py`](../../dimos/stream/video_operators.py) |
| BackPressure system | [`dimos/rxpy_backpressure/backpressure.py`](../../dimos/rxpy_backpressure/backpressure.py) |
| Audio / Whisper STT | [`dimos/stream/audio/stt/node_whisper.py`](../../dimos/stream/audio/stt/node_whisper.py) |

### CLI & Configuration
| Component | File |
|-----------|------|
| Main CLI entry point | [`dimos/robot/cli/dimos.py`](../../dimos/robot/cli/dimos.py) |
| Global config definition | [`dimos/core/global_config.py`](../../dimos/core/global_config.py) |
| All available blueprints (auto-generated) | [`dimos/robot/all_blueprints.py`](../../dimos/robot/all_blueprints.py) |
| Project dependencies | [`pyproject.toml`](../../pyproject.toml) |

### Complete Robot Blueprints (good entry points to read)
| Blueprint | CLI name | File |
|-----------|----------|------|
| Go2 basic (connection + visualization only) | `unitree-go2-basic` | [`dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py`](../../dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py) |
| Go2 full navigation stack | `unitree-go2` | [`dimos/robot/unitree/go2/blueprints/smart/unitree_go2.py`](../../dimos/robot/unitree/go2/blueprints/smart/unitree_go2.py) |
| Go2 with object detection | `unitree-go2-detection` | [`dimos/robot/unitree/go2/blueprints/smart/unitree_go2_detection.py`](../../dimos/robot/unitree/go2/blueprints/smart/unitree_go2_detection.py) |
| Go2 + LLM agent (GPT-4o default) | `unitree-go2-agentic` | [`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic.py`](../../dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic.py) |
| Go2 + local Ollama agent (qwen3:8b) | `unitree-go2-agentic-ollama` | [`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_ollama.py`](../../dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_ollama.py) |
| Go2 + HuggingFace agent | `unitree-go2-agentic-huggingface` | [`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_huggingface.py`](../../dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_huggingface.py) |
| Go2 + agent + MCP server | `unitree-go2-agentic-mcp` | [`dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py`](../../dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py) |
| Common agentic skills (navigation, follow, speak) | — | [`dimos/robot/unitree/go2/blueprints/agentic/_common_agentic.py`](../../dimos/robot/unitree/go2/blueprints/agentic/_common_agentic.py) |
| Standalone demo agent (no robot hardware) | `demo-agent` | [`dimos/agents/demo_agent.py`](../../dimos/agents/demo_agent.py) |

---

## How to Read the Codebase

**Recommended reading order for a beginner:**

1. [`dimos/core/module.py`](../../dimos/core/module.py) — understand Module, In[T], Out[T]
2. [`dimos/core/blueprints.py`](../../dimos/core/blueprints.py) — understand autoconnect()
3. [`dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py`](../../dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py) — simplest real blueprint
4. [`dimos/agents/agent.py`](../../dimos/agents/agent.py) — the LLM agent (ReAct loop)
5. [`dimos/agents/system_prompt.py`](../../dimos/agents/system_prompt.py) — what the agent "knows"
6. [`dimos/agents/annotation.py`](../../dimos/agents/annotation.py) — the @skill decorator
7. [`dimos/protocol/mcp/mcp.py`](../../dimos/protocol/mcp/mcp.py) — the MCP server
8. [`dimos/perception/detection/module2D.py`](../../dimos/perception/detection/module2D.py) — YOLO detection module
9. [`dimos/navigation/visual_servoing/detection_navigation.py`](../../dimos/navigation/visual_servoing/detection_navigation.py) — how detection drives navigation

**Also see:** [`docs/architecture/RUNNING.md`](./RUNNING.md) — how to run Dimos from this repo.
