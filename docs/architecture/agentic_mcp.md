# Agentic System and MCP Protocol — Technical Reference

> This document explains how Dimos makes a quadruped robot language-model-controlled.
> We cover the ReAct loop, the `@skill` decorator mechanism, all available skills,
> and the MCP TCP protocol that lets external tools like Claude Code control the robot.
> All code references are relative to the repository root.

---

## Table of Contents

1. [What "Agentic" Means in Dimos](#1-what-agentic-means-in-dimos)
2. [The ReAct Loop — Code Walkthrough](#2-the-react-loop--code-walkthrough)
3. [The @skill Decorator Mechanism](#3-the-skill-decorator-mechanism)
4. [Complete Skill Inventory](#4-complete-skill-inventory)
5. [MCP Server — Protocol Details](#5-mcp-server--protocol-details)
6. [Data Flow: User Command to Robot Action](#6-data-flow-user-command-to-robot-action)
7. [Blueprint Inheritance for Agentic Variants](#7-blueprint-inheritance-for-agentic-variants)
8. [System Prompt: Daneel](#8-system-prompt-daneel)

---

## 1. What "Agentic" Means in Dimos

There are three levels of autonomy in the Go2 blueprints:

| Level | Example Blueprint | How it works | Who decides? |
|---|---|---|---|
| **Basic** | `unitree-go2-basic` | Raw WebRTC connection. Sensor data in, velocity commands out. | Human (keyboard/joystick) |
| **Smart** | `unitree-go2` | Navigation stack. Given a goal pose, robot plans a path and executes it. | Path planner algorithm |
| **Agentic** | `unitree-go2-agentic` | LLM receives natural language, reasons about actions, calls skill functions that move the robot. | LLM (GPT-4o or other) |

The key difference in the agentic level: **the LLM has real authority over the
robot's motors through the skill interface**. When the agent decides to call
`relative_move(forward=2.0)`, that is not a simulation — it dispatches a real
RPC to `UnitreeSkillContainer.relative_move()` which blocks until the robot
has physically moved 2 metres.

---

## 2. The ReAct Loop — Code Walkthrough

**File:** `dimos/agents/agent.py`

### 2a. The Agent Module

```python
@dataclass
class AgentConfig(ModuleConfig):       # agent.py:42
    system_prompt: str | None = SYSTEM_PROMPT
    model: str = "gpt-4o"             # default LLM
    model_fixture: str | None = None  # for testing with mock responses

class Agent(Module):                   # agent.py:49
    agent:       Out[BaseMessage]     # publishes each LLM step to LCM bus
    human_input: In[str]              # receives text commands (from WebInput)
    agent_idle:  Out[bool]            # True when queue is empty
```

The `Agent` is a standard Dimos `Module`. It subscribes to `human_input` (a text
stream from the `WebInput` module listening on TCP port 5555) and publishes its
response messages to `agent`.

### 2b. Startup — Tool Discovery

When the system starts, `on_system_modules()` (line 93) is called with all peer
modules as RPC client handles:

```python
@rpc
def on_system_modules(self, modules: list[RPCClient]) -> None:
    # For Ollama: ensure model is pulled before starting
    if self.config.model.startswith("ollama:"):
        ensure_ollama_model(self.config.model.removeprefix("ollama:"))

    # Create LangGraph agent with tools from all modules
    self._state_graph = create_agent(          # agent.py:108
        model=model,                           # "gpt-4o" or BaseChatModel
        tools=_get_tools_from_modules(self, modules, self.rpc),
        system_prompt=self.config.system_prompt,
    )
    self._thread.start()  # start the background message-processing thread
```

`_get_tools_from_modules()` (line 154) iterates over every peer module and calls
`module.get_skills()` via RPC. Each returned `SkillInfo` is converted to a
LangChain `StructuredTool`. The complete tool list is then frozen into the
LangGraph state graph.

### 2c. The Background Thread Loop

```python
def _thread_loop(self) -> None:       # agent.py:119
    while not self._stop_event.is_set():
        try:
            message = self._message_queue.get(timeout=0.5)
        except Empty:
            continue                  # keep polling for new messages
        self._process_message(self._state_graph, message)

def _process_message(self, state_graph, message):   # agent.py:131
    self.agent_idle.publish(False)
    self._history.append(message)
    self.agent.publish(message)       # immediately publish user message

    # Stream every step of the ReAct loop
    for update in state_graph.stream(
        {"messages": self._history},
        stream_mode="updates",        # receive each node's output as it runs
    ):
        for node_output in update.values():
            for msg in node_output.get("messages", []):
                self._history.append(msg)
                self.agent.publish(msg)   # each reasoning step → LCM bus

    if self._message_queue.empty():
        self.agent_idle.publish(True)
```

### 2d. ReAct Loop — Step by Step

The LangGraph `create_agent()` creates a **Re**ason + **Act** loop with two
alternating node types:

```
User: "Go to the kitchen and bring back a report"
         │
         ▼
  ┌─────────────────────┐
  │  LLM Node (GPT-4o)  │  ← receives full conversation history
  │                      │
  │  Thinks: "I need to navigate to the kitchen.
  │  I'll call navigate_with_text('kitchen')"
  │                      │
  │  Output: AIMessage with tool_calls=[
  │    {name: "navigate_with_text", args: {query: "kitchen"}}
  │  ]
  └──────────┬──────────┘
             │ tool_call
             ▼
  ┌─────────────────────┐
  │  Tools Node         │  ← executes the RPC call
  │                      │
  │  → RpcCall("navigate_with_text")(query="kitchen")
  │  → NavigationSkillContainer.navigate_with_text("kitchen")
  │  → A* planner finds path → robot walks to kitchen
  │  → returns "Navigation started"
  │                      │
  │  Output: ToolMessage("Navigation started")
  └──────────┬──────────┘
             │ tool result
             ▼
  ┌─────────────────────┐
  │  LLM Node (GPT-4o)  │  ← now knows navigation started
  │                      │
  │  Thinks: "Navigation is underway. I can now
  │  report back to the user."
  │                      │
  │  Output: AIMessage("I'm heading to the kitchen now!")
  └─────────────────────┘
         │
         ▼
  [loop ends — queue empty → agent_idle=True]
```

`self._history` accumulates the full conversation — every user message, every
AI response, every tool call and its result. The LLM always sees the complete
context when reasoning.

---

## 3. The @skill Decorator Mechanism

### 3a. The Decorator

**File:** `dimos/agents/annotation.py`

```python
# The entire file (24 lines):
def skill(func: F) -> F:
    func.__rpc__ = True    # marks method as callable via dimos RPC system
    func.__skill__ = True  # marks method for LLM tool discovery
    return func
```

That's all `@skill` does — it sets two boolean attributes on the function object.
These are later read during discovery.

### 3b. Skill Discovery — `get_skills()`

**File:** `dimos/core/module.py:383`

Every `Module` subclass inherits this method:

```python
@rpc
def get_skills(self) -> list[SkillInfo]:
    skills: list[SkillInfo] = []
    for name in dir(self):
        attr = getattr(self, name)
        if callable(attr) and hasattr(attr, "__skill__"):
            # tool() is LangChain's decorator — it reads the method's
            # type hints and docstring to build a Pydantic JSON schema
            schema = json.dumps(tool(attr).args_schema.model_json_schema())
            skills.append(
                SkillInfo(
                    class_name=self.__class__.__name__,
                    func_name=name,
                    args_schema=schema,  # JSON string with parameter types + docs
                )
            )
    return skills
```

`tool(attr).args_schema` automatically creates a Pydantic model from the
method's Python type hints and docstring. This is how:

```python
@skill
def relative_move(self, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0) -> str:
    """Move the robot relative to its current position."""
    ...
```

...becomes the JSON schema:
```json
{
  "description": "Move the robot relative to its current position.",
  "properties": {
    "forward": {"type": "number", "default": 0.0},
    "left":    {"type": "number", "default": 0.0},
    "degrees": {"type": "number", "default": 0.0}
  }
}
```

This is what the LLM receives when it queries `tools/list` — the schema tells
the LLM exactly what parameters to pass.

### 3c. Converting Skills to LangChain Tools

**File:** `dimos/agents/agent.py:161`

```python
def _skill_to_tool(agent: Agent, skill: SkillInfo, rpc: RPCSpec) -> StructuredTool:
    # Create an RPC callable for this specific skill
    rpc_call = RpcCall(None, rpc, skill.func_name, skill.class_name, [])

    def wrapped_func(*args, **kwargs) -> str:
        result = rpc_call(*args, **kwargs)   # actual RPC to skill worker

        if result is None:
            return "It has started. You will be updated later."

        # Special case: if result has agent_encode(), it's a visual artifact
        # (e.g. an Image or detection result) — add it to conversation as multimodal
        if hasattr(result, "agent_encode"):
            uuid_ = str(uuid.uuid4())
            _append_image_to_history(agent, skill, uuid_, result)
            return f"Tool call started with UUID: {uuid_}"

        return str(result)   # plain text result

    return StructuredTool(
        name=skill.func_name,
        func=wrapped_func,
        args_schema=json.loads(skill.args_schema),  # Pydantic model class
    )
```

The key insight: `wrapped_func` is a **closure** that captures the RPC call.
When the LangGraph Tools node executes the tool, it calls `wrapped_func(**args)`
which dispatches the actual RPC to the skill's Dask worker on a background
thread.

---

## 4. Complete Skill Inventory

All `@skill` decorated methods available in the agentic blueprints:

### 4a. Navigation Skills

**File:** `dimos/agents/skills/navigation.py`

| Method | Line | Description |
|---|---|---|
| `tag_location(name)` | ~82 | Save current robot position with a name. Later: `navigate_with_text("kitchen")` finds it. |
| `navigate_with_text(query)` | ~116 | Navigate to a description. Tries: (1) tagged location, (2) visible object via detection, (3) semantic map CLIP query. |
| `stop_navigation()` | ~279 | Cancel current navigation goal immediately. |

### 4b. Robot Control Skills

**File:** `dimos/robot/unitree/unitree_skill_container.py`

| Method | Line | Description |
|---|---|---|
| `relative_move(forward, left, degrees)` | 211 | Move robot by offset from current position. All values in metres / degrees. Blocks until arrival. |
| `wait(seconds)` | ~282 | Pause execution for N seconds. Used in delivery/pickup workflows. |
| `current_time()` | ~292 | Return current wall-clock datetime string. |
| `execute_sport_command(command_name)` | ~297 | Execute one of 38 named Unitree sport commands (see full list below). Uses fuzzy matching for misspellings. |

**Full list of sport commands** (from `UNITREE_WEBRTC_CONTROLS`, line 34):

| Command | Description |
|---|---|
| `BalanceStand` | Activates balanced standing mode |
| `StandUp` | Transition from sitting/prone to standing |
| `StandDown` | Transition from standing to sitting/prone |
| `RecoveryStand` | **Run after flips/jumps** — recovers to command-ready state |
| `Sit` | Sit down |
| `RiseSit` | Rise from sitting to standing |
| `SwitchGait` | Switch walking pattern for different terrain |
| `Trigger` | Trigger a custom routine |
| `BodyHeight` | Adjust body height from ground |
| `FootRaiseHeight` | Control foot lift height during walking |
| `SpeedLevel` | Set movement speed level |
| `Hello` | Greeting gesture (wave) |
| `Stretch` | Stretching routine |
| `TrajectoryFollow` | Follow a predefined trajectory |
| `ContinuousGait` | Continuous walking mode |
| `Content` | Happy display action |
| `Wallow` | Falls onto back and rolls around |
| `Dance1` | Predefined dance routine 1 |
| `Dance2` | Predefined dance routine 2 |
| `GetBodyHeight` | Query current body height |
| `GetFootRaiseHeight` | Query current foot raise height |
| `GetSpeedLevel` | Query current speed level |
| `SwitchJoystick` | Switch to joystick control mode |
| `Pose` | Assume a predefined pose |
| `Scrape` | Scraping motion |
| `FrontFlip` | Front flip (requires `RecoveryStand` after) |
| `FrontJump` | Forward jump |
| `FrontPounce` | Forward pounce |
| `WiggleHips` | Hip wiggling motion |
| `GetState` | Query current robot state |
| `EconomicGait` | Energy-efficient movement mode |
| `FingerHeart` | Finger heart gesture on hind legs |
| `Handstand` | Balance on front legs |
| `CrossStep` | Cross-step movement pattern |
| `OnesidedStep` | One-sided step movement |
| `Bound` | Bounding movement |
| `MoonWalk` | Moonwalk motion |
| `LeftFlip` | Left-side flip |
| `RightFlip` | Right-side flip |
| `Backflip` | Backflip |

> **Important:** After any dynamic command (FrontFlip, FrontJump, Sit, Dance,
> Wallow), always call `execute_sport_command("RecoveryStand")` before issuing
> navigation commands. This is explicitly stated in the system prompt.

### 4c. Person Following Skills

**File:** `dimos/agents/skills/person_follow.py`

| Method | Description |
|---|---|
| `follow_person(query)` | Follow a person matching the description (e.g. "man with blue shirt"). Uses Qwen-VL for initial identification and EdgeTAM for multi-frame tracking. Combines visual servoing with 3D navigation. |
| `stop_following()` | Stop following, publish zero velocity. |

### 4d. Speech Skill

**File:** `dimos/agents/skills/speak_skill.py:52`

| Method | Description |
|---|---|
| `speak(text)` | Synthesise text to speech via OpenAI TTS (Onyx voice, 1.2× speed). Blocks until audio playback completes. Returns the spoken text. |

> **Why blocking?** The system prompt says "users hear you through speakers but
> cannot see text." The robot must finish speaking before moving on so it
> doesn't narrate over itself.

### 4e. GPS Navigation Skills

**File:** `dimos/agents/skills/gps_nav_skill.py`

| Method | Description |
|---|---|
| `set_gps_travel_points(points)` | Set outdoor GPS waypoints as a list of `{"lat": float, "lon": float}` dicts. Validates max 50km from current location. |

### 4f. Google Maps Skills

**File:** `dimos/agents/skills/google_maps_skill_container.py`
**Requires:** `GOOGLE_MAPS_API_KEY`

| Method | Description |
|---|---|
| `where_am_i(context_radius=200)` | Reverse geocode current GPS position → street name, nearby landmarks, neighbourhood context. |
| `get_gps_position_for_queries(queries)` | Look up GPS coordinates for place names or intersections. Used before `set_gps_travel_points`. |

### 4g. OpenStreetMap Skill

**File:** `dimos/agents/skills/osm.py`

| Method | Description |
|---|---|
| `map_query(query_sentence)` | Query OpenStreetMap via VL model. E.g. "Where can I find a coffee shop?" Returns location name, coordinates, and distance. |

### 4h. Vision Skills

**File:** `dimos/perception/detection/module3D.py:114`
**File:** `dimos/perception/object_scene_registration.py:224`

| Method | Description |
|---|---|
| `ask_vlm(question)` | Ask Qwen-VL a question about the current camera view. Returns text answer. |
| `detect(*prompts)` | Detect named objects using YOLO-E. Returns object_id UUIDs. |
| `select(track_id)` | Promote a detected object to permanent memory by its tracking ID. |

---

## 5. MCP Server — Protocol Details

The **Model Context Protocol (MCP)** allows external AI tools (Claude Code, any
MCP client) to call all `@skill` methods on the robot via TCP.

**File:** `dimos/protocol/mcp/mcp.py`

### 5a. Server Basics

```
Protocol:   JSON-RPC 2.0
Transport:  TCP, newline-delimited JSON (one JSON object per line)
Port:       9990  (bound to 0.0.0.0 — accessible from any network interface)
```

The server starts in `MCPModule.start()` → `_start_server(port=9990)`. It runs
inside an asyncio event loop on a background thread (the same thread used by
all async Dimos modules via Dask).

### 5b. Supported Methods

#### `initialize` — Handshake

```json
// Request
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

// Response
{"jsonrpc": "2.0", "id": 1, "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "dimensional", "version": "1.0.0"}
}}
```

#### `tools/list` — Discover all skills

```json
// Request
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

// Response
{"jsonrpc": "2.0", "id": 2, "result": {"tools": [
    {
        "name": "relative_move",
        "description": "Move the robot relative to its current position.",
        "inputSchema": {
            "properties": {
                "forward": {"default": 0.0, "type": "number"},
                "left":    {"default": 0.0, "type": "number"},
                "degrees": {"default": 0.0, "type": "number"}
            }
        }
    },
    ...
]}}
```

#### `tools/call` — Execute a skill

```json
// Request
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {"name": "relative_move", "arguments": {"forward": 2.0}}}

// Response (after robot completes the movement)
{"jsonrpc": "2.0", "id": 3, "result": {
    "content": [{"type": "text", "text": "Navigation goal reached"}]
}}
```

Note: `tools/call` **blocks** until the skill returns. For navigation skills,
this can take 10–30 seconds while the robot physically moves.

### 5c. Error Codes

| Scenario | JSON-RPC error code |
|---|---|
| Missing or invalid tool name | `-32602` |
| Unknown method | `-32601` |
| Skill not found in registry | `result.content = "Skill not found"` (not an error code) |
| Skill throws an exception | `result.content = "Error: <exception message>"` |

### 5d. Skill Discovery (same as Agent)

`on_system_modules()` in `MCPModule` uses the **exact same** discovery pattern
as `Agent`:

```python
@rpc
def on_system_modules(self, modules: list[RPCClient]) -> None:
    self._skills = [skill for module in modules for skill in (module.get_skills() or [])]
    self._rpc_calls = {
        skill.func_name: RpcCall(None, self.rpc, skill.func_name, skill.class_name, [])
        for skill in self._skills
    }
```

All skills from all connected modules are collected into `_rpc_calls`, a dict
mapping `func_name → RpcCall`. When a `tools/call` request arrives, the skill
is dispatched via `run_in_executor` (to avoid blocking the asyncio loop) calling
`rpc_call(**args)`.

### 5e. Quick Test from Command Line

While `unitree-go2-agentic-mcp` is running, you can test the MCP server:

```bash
# In a second terminal:

# List all available skills
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | nc localhost 9990

# Make the robot say hello
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"speak","arguments":{"text":"Hello from the terminal!"}}}' | nc localhost 9990

# Make the robot do a dance
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"execute_sport_command","arguments":{"command_name":"Dance1"}}}' | nc localhost 9990
```

---

## 6. Data Flow: User Command to Robot Action

### Path A — humancli → Agent

```
[User types in humancli terminal]
    │  text string via TCP port 5555
    ▼
WebInput module (dimos/agents/web_human_input.py)
    │  Out[str] published to LCM bus
    ▼
Agent.human_input: In[str]
    │  _on_human_input() puts HumanMessage on queue
    ▼
Agent._thread_loop()
    │  dequeues message → _process_message()
    ▼
LangGraph state_graph.stream({"messages": history})

    ┌──── LLM Node (GPT-4o via OpenAI API) ────┐
    │  receives: system_prompt + full history   │
    │  produces: tool_call to "speak"           │
    └──────────────────┬────────────────────────┘
                       │ AIMessage(tool_calls=[...])
                       ▼
    ┌──── Tools Node ──────────────────────────────┐
    │  _skill_to_tool wrapped_func(**args)          │
    │  → RpcCall("speak")("Hello from Daneel!")     │
    │  → SpeakSkill.speak() on Dask worker          │
    │  → OpenAI TTS → audio bytes → speakers        │
    │  → returns "Spoke: Hello from Daneel!"        │
    └──────────────────┬───────────────────────────┘
                       │ ToolMessage(result)
                       ▼
    ┌──── LLM Node ────────────────────────────────┐
    │  sees tool result, generates final response   │
    │  produces: AIMessage("Hi! I am Daneel.")      │
    └──────────────────────────────────────────────┘
         │
         ▼
agent.publish(msg) → Out[BaseMessage] → LCM bus
    (agentspy shows this in a second terminal)
```

### Path B — Claude Code → MCP → Robot

```
[Claude Code (or any MCP client)]
    │  TCP JSON-RPC to port 9990
    ▼
MCPModule._handle_request()
    │  method="tools/call", name="navigate_with_text"
    │  args={"query": "go to the sofa"}
    ▼
run_in_executor(rpc_call(**args))
    ▼
RpcCall("navigate_with_text")(query="go to the sofa")
    │  dispatched via internal RPC to NavigationSkillContainer worker
    ▼
NavigationSkillContainer.navigate_with_text("go to the sofa")
    │  1. check tagged locations
    │  2. visual detection in camera
    │  3. SpatialMemory.query_by_text("go to the sofa")
    │     → CLIP embedding → ChromaDB → nearest stored pose
    ▼
NavigationInterface.set_goal(target_pose)
    │  RPC to ReplanningAStarPlanner
    ▼
A* Planner generates path
    │  Out[Twist] cmd_vel → LCM bus
    ▼
GO2Connection.cmd_vel: In[Twist]
    │  WebRTC publish to robot
    ▼
Unitree Go2 motors move
    │
    ▼ (after arrival)
returns "Navigation started" to MCP client
```

---

## 7. Blueprint Inheritance for Agentic Variants

The agentic blueprints build on each other using composition and inheritance:

```
unitree_go2_basic  (blueprints/basic/unitree_go2_basic.py)
│
│  Modules:
│  ├── GO2Connection      ← WebRTC link to physical robot
│  │     Out[Image]  color_image  (1280×720 RGB @ 30 Hz)
│  │     Out[PointCloud2] lidar
│  │     Out[PoseStamped] odom
│  │     In[Twist]   cmd_vel      ← receives velocity commands
│  ├── DepthModule        ← front depth camera
│  └── KeyboardTeleop     ← WASD manual override
│
└── unitree_go2  (blueprints/smart/unitree_go2.py)
    │
    │  Adds:
    │  ├── VoxelMapper            ← 3D occupancy grid (0.1m voxels)
    │  ├── CostMapper             ← 2D inflation layer for planning
    │  ├── ReplanningAStarPlanner ← path planning + dynamic replanning
    │  └── WavefrontFrontierExplorer ← autonomous area exploration
    │
    └── unitree_go2_spatial  (blueprints/smart/unitree_go2_spatial.py)
        │
        │  Adds:
        │  ├── SpatialMemory   ← CLIP + ChromaDB semantic map
        │  └── Utilization     ← CPU/GPU/memory monitoring
        │
        └── unitree_go2_agentic  (blueprints/agentic/unitree_go2_agentic.py)
            │
            │  Adds (from _common_agentic.py):
            │  ├── Agent(model="gpt-4o")          ← LangGraph ReAct loop
            │  ├── NavigationSkillContainer        ← tag/navigate/stop skills
            │  ├── PersonFollowSkillContainer      ← follow_person skill
            │  ├── UnitreeSkillContainer           ← relative_move, sport cmds
            │  ├── WebInput (port 5555)            ← humancli text input
            │  └── SpeakSkill                      ← TTS output
            │
            ├── unitree_go2_agentic_mcp  (adds MCPModule TCP:9990)
            │     → External AI tools can call all skills via JSON-RPC
            │
            ├── unitree_go2_agentic_ollama  (changes model="ollama:qwen3:8b")
            │     → No API key. Needs local `ollama serve` + pulled model.
            │
            ├── unitree_go2_agentic_huggingface  (model="huggingface:Qwen/Qwen2.5-1.5B")
            │     → Needs HUGGINGFACE_API_KEY
            │
            └── unitree_go2_temporal_memory  (adds TemporalMemory module)
                  → Keeps episodic log of past events + reasoning history
```

**How `autoconnect()` wires this:** Each module declares typed ports:
```python
color_image: Out[Image]   # GO2Connection produces this
color_image: In[Image]    # SpatialMemory consumes this (same name + type)
```
`autoconnect()` matches `Out[Image] color_image` → `In[Image] color_image`
automatically. No manual wiring code needed. The entire stack builds from
one call to `blueprint.build()`.

---

## 8. System Prompt: Daneel

**File:** `dimos/agents/system_prompt.py`

The robot's name is **Daneel** (a reference to Isaac Asimov's robot Daneel
Olivaw — a robot that serves humanity while operating under strict safety laws).

```python
SYSTEM_PROMPT = """
You are Daneel, an AI agent created by Dimensional to control a Unitree Go2 quadruped robot.
```

Key sections of the prompt and what they enforce:

**Safety (highest priority):**
> "Prioritize human safety above all else. Respect personal boundaries.
> Never take actions that could harm humans, damage property, or damage the robot."

This constraint is absolute — it overrides any user instruction.

**Identity handling:**
> "If someone says 'daniel' or similar, ignore it (speech-to-text error)."

The robot's name is close to common human names, and the system uses Whisper
for speech-to-text. This prevents the agent from being accidentally addressed
by misheard speech.

**Communication via `speak`:**
> "Users hear you through speakers but cannot see text. Use `speak` to communicate."

The agent must call `speak(text)` to send any output the user needs to hear.
Returning text only from the ReAct loop is invisible to the user.

**Navigation discipline:**
> "Use `navigate_with_text` for most navigation."
> "Always run `execute_sport_command('RecoveryStand')` after dynamic movements."

This prevents the agent from using `relative_move` for semantic navigation
(always prefer the higher-level `navigate_with_text`) and from trying to walk
immediately after a flip (which would fail because the robot is still on the ground).

**GPS workflow:**
> "1. Use `get_gps_position_for_queries` to look up coordinates.
> 2. Then use `set_gps_travel_points` with those coordinates."

The GPS skill requires coordinates, not place names. The prompt enforces the
two-step pattern: first look up coordinates, then send waypoints.

**Proactive behaviour:**
> "Infer reasonable actions from ambiguous requests."

The agent is designed to make decisions rather than ask for clarification.
It should act on reasonable interpretations and inform the user of its
assumption (`speak("Heading to the front door — let me know if I should go elsewhere.")`).

---

**Excalidraw diagram:** `docs/architecture/go2_agentic_mcp.excalidraw`

---

*Sources:*
- `dimos/agents/agent.py`
- `dimos/agents/annotation.py`
- `dimos/agents/system_prompt.py`
- `dimos/core/module.py:383-395`
- `dimos/protocol/mcp/mcp.py`
- `dimos/robot/unitree/unitree_skill_container.py`
- `dimos/agents/skills/navigation.py`
- `dimos/agents/skills/person_follow.py`
- `dimos/agents/skills/speak_skill.py`
- `dimos/robot/unitree/go2/blueprints/agentic/_common_agentic.py`
- `dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic_mcp.py`
