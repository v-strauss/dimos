# Perception Pipeline — Technical Reference

> This document traces the full perception stack in Dimos: from raw camera
> pixels to 3D object positions that agents and navigation can consume.
> All code references are relative to the repository root.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Input Sources — Camera and LiDAR](#2-input-sources--camera-and-lidar)
3. [YOLO 2D Detection](#3-yolo-2d-detection)
4. [3D Projection — Pixel to World](#4-3d-projection--pixel-to-world)
5. [ObjectDB — Deduplication and Persistence](#5-objectdb--deduplication-and-persistence)
6. [SpatialMemory — CLIP + ChromaDB Semantic Map](#6-spatialmemory--clip--chromadb-semantic-map)
7. [ObjectSceneRegistration — YOLO-E with Prompting](#7-objectsceneregistration--yolo-e-with-prompting)
8. [Complete Data Flow Diagram](#8-complete-data-flow-diagram)
9. [How Agents Consume Perception Outputs](#9-how-agents-consume-perception-outputs)
10. [Blueprint Wiring — How It All Connects](#10-blueprint-wiring--how-it-all-connects)

---

## 1. Architecture Overview

The perception stack has **three parallel pipelines** that run simultaneously:

| Pipeline | Speed | Output | Used for |
|---|---|---|---|
| **2D fast path** | ~10 Hz | `Detection2DArray` (bounding boxes) | Visualization, quick object awareness |
| **3D fused path** | ~5 Hz | `ImageDetections3DPC` (3D pointclouds) | Navigation goals, spatial reasoning |
| **Semantic memory** | ~1 frame/s | ChromaDB embeddings | "Where is the kitchen?"-type queries |

All three read from the same camera stream. The 3D path additionally uses the LiDAR pointcloud to compute real metric distances.

```
Camera ──► 2D fast path ──────────────────► Detection2DArray → LCM bus
       │
       ├──► 3D fused path (+ LiDAR) ──────► ImageDetections3DPC → ObjectDB
       │                                                              │
       │                                                         agent_encode()
       │
       └──► Semantic memory (+ odometry) ──► CLIP embedding → ChromaDB
                                                              │
                                                          query_by_text()
```

---

## 2. Input Sources — Camera and LiDAR

All sensor streams originate from the `GO2Connection` module.

**File:** `dimos/robot/unitree/go2/connection.py`

```python
# Output ports (RxPy Observables carrying typed messages)
color_image: Out[Image]      # 1280×720 RGB from front depth camera @ ~30 Hz
lidar:       Out[PointCloud2] # 3D LiDAR scan (mid360 or L1) @ ~10 Hz
odom:        Out[PoseStamped] # wheel odometry + TF transform @ ~50 Hz
camera_info: Out[CameraInfo] # static calibration, published once at startup
```

The camera intrinsics matrix K (used for 3D projection) is hardcoded for the
Go2's built-in front camera:

```python
# dimos/robot/unitree/go2/connection.py
fx, fy = 819.553492, 820.646595  # focal lengths (pixels)
cx, cy = 625.284099, 336.808987  # principal point (image centre)
width, height = 1280, 720
```

These values encode how the physical lens maps world angles to pixel positions.
Knowing `fx`, `fy`, `cx`, `cy` is what makes reversing the projection possible
(see Section 4).

The `In[T]` / `Out[T]` types are not just type hints — they are real RxPy
Observable subscriptions under the hood. `autoconnect()` in the blueprint
wires modules together by matching output and input ports by **name + type**.

---

## 3. YOLO 2D Detection

### 3a. Detector — `Yolo2DDetector`

**File:** `dimos/perception/detection/detectors/yolo.py`

```python
class Yolo2DDetector(Detector):
    def __init__(self, model_path="models_yolo", model_name="yolo11n.pt", ...):
        self.model = YOLO(get_data(model_path) / model_name, task="detect")
        self.device = "cuda" if is_cuda_available() else "cpu"   # lines 43-48

    def process_image(self, image: Image) -> ImageDetections2D:  # line 50
        results = self.model.track(
            source=image.to_opencv(),
            device=self.device,
            conf=0.5,    # discard detections below 50% confidence
            iou=0.6,     # non-max suppression threshold
            persist=True, # ← IMPORTANT: keeps tracker state between frames
            verbose=False,
        )
        return ImageDetections2D.from_ultralytics_result(image, results)
```

Key points:
- **Model:** YOLO11n ("nano") — fastest variant, ~3ms per frame on CPU, ~1ms on CUDA
- **`persist=True`**: the tracker maintains a Kalman filter across frames. Each
  object gets a stable `track_id` integer that persists as long as the object
  is visible. This is the primary deduplication key in `ObjectDB` (Section 5).
- **`conf=0.5`**: only detections with ≥50% confidence are returned
- **Output:** `ImageDetections2D` — a list of `Detection2DBBox` objects, each with
  `bbox`, `track_id`, `class_id`, `confidence`, `name`, `ts`, `image`

### 3b. Module — `Detection2DModule`

**File:** `dimos/perception/detection/module2D.py:54`

```python
class Detection2DModule(Module):
    color_image: In[Image]           # subscribes to camera stream

    detections:       Out[Detection2DArray]   # published detections
    annotations:      Out[ImageAnnotations]   # Foxglove viz overlay
    detected_image_0: Out[Image]              # cropped top-3 detections
    detected_image_1: Out[Image]
    detected_image_2: Out[Image]
```

**Config** (line 39):
```python
@dataclass
class Config(ModuleConfig):
    max_freq: float = 10              # Hz cap via sharpness_barrier
    detector: Callable = Yolo2DDetector
    publish_detection_images: bool = True
    camera_info: CameraInfo = None
    filter: list[Filter2D] | None = None
```

**Pipeline** (lines 82-92):
```python
def sharp_image_stream(self) -> Observable[Image]:
    # 1. Apply sharpness filter (blur metric) to avoid running YOLO on blurry frames
    # 2. Apply backpressure: if YOLO is slow, drop intermediate frames
    #    rather than building a queue
    return backpressure(
        self.color_image.pure_observable().pipe(
            sharpness_barrier(self.config.max_freq),  # max 10 Hz
        )
    )

def detection_stream_2d(self) -> Observable[ImageDetections2D]:
    # Map each sharp frame through YOLO detector
    return backpressure(
        self.sharp_image_stream().pipe(ops.map(self.process_image_frame))
    )
```

Why `sharpness_barrier`? Fast motion causes motion blur. YOLO's accuracy drops
sharply on blurred frames. The barrier computes a Laplacian variance on each
incoming frame and drops frames whose score falls below a threshold.

Why `backpressure`? If the GPU is saturated, instead of accumulating a 10-second
backlog of frames, the system drops intermediate frames and always processes the
*most recent* one. This keeps latency low at the cost of throughput.

---

## 4. 3D Projection — Pixel to World

### 4a. How 2D becomes 3D

**File:** `dimos/perception/detection/module3D.py`

`Detection3DModule` extends `Detection2DModule` with a second input:

```python
class Detection3DModule(Detection2DModule):
    color_image: In[Image]      # camera frame
    pointcloud:  In[PointCloud2] # LiDAR scan (same coordinate frame as camera)
```

The core operation is `process_frame()` (line 66):
```python
def process_frame(self, detections: ImageDetections2D,
                  pointcloud: PointCloud2, transform: Transform) -> ImageDetections3DPC:
    for detection in detections:
        detection3d = Detection3DPC.from_2d(
            detection,
            world_pointcloud=pointcloud,
            camera_info=self.config.camera_info,
            world_to_optical_transform=transform,  # TF: lidar frame → camera frame
        )
        ...
```

`Detection3DPC.from_2d()` works like this:
1. Take the 2D bounding box pixels from YOLO
2. Transform the LiDAR pointcloud into camera optical frame (using the TF transform)
3. Select only LiDAR points that fall inside the 2D bounding box in the image
4. The 3D centre of those points becomes the object's 3D position

This is more accurate than using an assumed depth because it uses the actual
LiDAR returns from the object's surface.

### 4b. `pixel_to_3d()` — The Math

**File:** `dimos/perception/detection/module3D.py:88`

For cases where LiDAR is unavailable or for quick estimates:

```python
def pixel_to_3d(self, pixel: tuple[int, int], assumed_depth: float = 1.0) -> Vector3:
    # Camera intrinsics from the 3x3 matrix K stored row-major as 9 floats:
    # K = [fx,  0, cx,
    #       0, fy, cy,
    #       0,  0,  1]
    fx, fy = self.config.camera_info.K[0], self.config.camera_info.K[4]
    cx, cy = self.config.camera_info.K[2], self.config.camera_info.K[5]

    # Reverse the projection: undo how the camera maps world → pixels
    x_norm = (pixel[0] - cx) / fx   # normalised X in image plane
    y_norm = (pixel[1] - cy) / fy   # normalised Y in image plane

    # Scale by assumed depth (meters) to get 3D position
    # Camera optical frame convention: X right, Y down, Z forward
    return Vector3(x_norm * assumed_depth, y_norm * assumed_depth, assumed_depth)
```

Intuition: if `fx = 820` and the image is 1280 wide, then one pixel of
horizontal offset at `z=1m` corresponds to `1/820 ≈ 1.2mm` in the real world.

### 4c. Temporal Alignment

LiDAR (10 Hz) and camera (30 Hz) emit at different rates. Before 3D projection,
the module calls `align_timestamped()` to match each detection batch with the
closest pointcloud within a 250ms tolerance window.

```python
# dimos/types/timestamped.py
# Used in Detection3DModule to synchronise 10 Hz LiDAR with 30 Hz camera
align_timestamped(detections_observable, pointcloud_observable, tolerance_ms=250)
```

If no matching pointcloud is found within the window, the frame is skipped.

---

## 5. ObjectDB — Deduplication and Persistence

Every time the 3D detection pipeline processes a frame, it produces 5–20 new
`Object` instances. Without deduplication, "the chair" would appear dozens of
times in the database. `ObjectDB` solves this.

**File:** `dimos/perception/detection/objectDB.py:33`

### 5a. Two-tier Storage

```python
class ObjectDB:
    def __init__(
        self,
        distance_threshold: float = 0.2,           # metres
        min_detections_for_permanent: int = 6,
        pending_ttl_s: float = 5.0,
        track_id_ttl_s: float = 5.0,
    ):
        self._pending_objects: dict[str, Object] = {}  # not yet confirmed
        self._objects:         dict[str, Object] = {}  # permanent (6+ detections)
        self._track_id_map:    dict[int, str]    = {}  # tracker ID → object UUID
        self._lock = threading.RLock()                  # thread-safe (GPU + CPU threads)
```

An `Object` starts in `_pending_objects`. Once it has been seen at least
`min_detections_for_permanent` (default 6) times, it is promoted to `_objects`.
Only `_objects` are returned to the agent via `agent_encode()`.

Why require 6 detections? This filters out one-time false positives (a shadow
that briefly looks like a person, a specular highlight that YOLO misclassifies).
An object seen from 6 different camera angles with stable 3D position is
reliably real.

### 5b. Deduplication Algorithm

**File:** `dimos/perception/detection/objectDB.py` — the `_match()` method

When `add_objects()` is called with a new batch, each incoming object goes
through this match priority:

```
New incoming Object
        │
        ▼
1. Track ID match?
   ─ YOLO's tracker assigns track_id across frames
   ─ look up track_id in _track_id_map (dict, O(1))
   ─ if found and TTL not expired: UPDATE that object
        │
        ▼  (no track_id match)
2. Distance match?
   ─ compute euclidean distance: |new.center - existing.center|
   ─ search _pending_objects then _objects
   ─ if any object within 0.2m: UPDATE that object
        │
        ▼  (no match at all)
3. Create new pending object
   ─ assign fresh UUID, insert into _pending_objects
```

The `track_id` TTL (5 seconds) handles the case where an object temporarily
leaves the camera frame. If it reappears within 5 seconds with the same YOLO
`track_id`, it is recognised as the same object.

The distance threshold (0.2m) handles the case where `track_id` was reset (e.g.
after a camera cut) but the object is clearly the same physical thing.

### 5c. Promotion Logic

**File:** `dimos/perception/detection/objectDB.py` — `_check_promotion()`

After each update, `_check_promotion()` checks if `object.detections_count >= 6`.
If so, the object is moved from `_pending_objects` to `_objects`.

```python
def _check_promotion(self, obj: Object) -> bool:
    if obj.detections_count >= self._min_detections:
        self._objects[obj.object_id] = obj
        del self._pending_objects[obj.object_id]
        return True
    return False
```

### 5d. Stale Object Pruning

Pending objects not seen for 5 seconds are removed:

```python
def _prune_stale_pending(self, now: float) -> None:
    stale = [
        obj_id for obj_id, obj in self._pending_objects.items()
        if (now - obj.last_seen_ts) > self._pending_ttl_s
    ]
    for obj_id in stale:
        del self._pending_objects[obj_id]
```

### 5e. Agent Interface

```python
def agent_encode(self) -> list[dict]:
    """Return permanent objects in a format suitable for LLM consumption."""
    with self._lock:
        return [obj.agent_encode() for obj in self._objects.values()]
```

Each `Object.agent_encode()` returns a dict with `object_id`, `name`, `center`,
`size`, `detections_count` — compact enough to fit many objects in an LLM context.

| Field | Type | Meaning |
|---|---|---|
| `object_id` | UUID string | Stable identifier (use for navigation goals) |
| `name` | str | YOLO class name ("chair", "person", "bottle") |
| `center` | `{x, y, z}` | 3D centroid in world frame (metres) |
| `size` | `{x, y, z}` | Bounding box dimensions (metres) |
| `detections_count` | int | How many frames this object has been seen |

---

## 6. SpatialMemory — CLIP + ChromaDB Semantic Map

While `ObjectDB` tracks *specific objects* the YOLO detector has seen,
`SpatialMemory` builds a *visual map of everywhere the robot has been* — and
allows querying it with natural language.

**File:** `dimos/perception/spatial_perception.py`

### 6a. What It Stores

Every time the robot moves ≥10cm or 1 second elapses, `SpatialMemory` captures
the current camera frame and:

1. Computes a **512-dimensional CLIP embedding** of the image (visual content)
2. Records the robot's **3D pose** (from odometry)
3. Stores both in **ChromaDB** (a local vector database on disk)

```python
# Persistence path
_SPATIAL_MEMORY_DIR = "assets/output/memory/spatial_memory/chromadb_data/"
```

### 6b. CLIP Embeddings — What They Are

CLIP (Contrastive Language-Image Pre-training) maps both images and text into the
same 512-dimensional vector space. This means:

- Image of a kitchen → vector ≈ text "kitchen" vector
- Image of a sofa → vector ≈ text "living room" vector

When you call `query_by_text("find the kitchen")`, CLIP encodes the text query
into the same 512-dim space, then ChromaDB finds the stored image embeddings
that are most geometrically close (cosine similarity). The result is the stored
camera frame (and its associated pose) that most closely matches your query.

### 6c. Query Methods

```python
# dimos/perception/spatial_perception.py

def query_by_text(self, text: str) -> list[SpatialMemoryResult]:
    """Find stored frames whose visual content matches a text description."""
    # CLIP encodes text → 512-dim vector
    # ChromaDB finds closest stored image embeddings (cosine similarity)
    # Returns frames + their 3D robot poses

def query_by_image(self, image: Image) -> list[SpatialMemoryResult]:
    """Find stored frames visually similar to a given image."""

def query_by_location(self, x: float, y: float, radius: float) -> list[SpatialMemoryResult]:
    """Find frames captured near a specific 3D position."""

def add_named_location(self, name: str) -> None:
    """Tag the current position with a human-readable name."""

def query_tagged_location(self, query: str) -> SpatialMemoryResult | None:
    """Find a tagged location by name or semantic similarity."""
```

The `navigate_with_text()` agent skill calls `query_by_text()` as its third
fallback when direct object tracking and tagged-location search both fail.

---

## 7. ObjectSceneRegistration — YOLO-E with Prompting

While `ObjectDB` + `Detection3DModule` run *passively* in the background,
`ObjectSceneRegistrationModule` provides an **agent-callable skill** for
on-demand detection of specific objects.

**File:** `dimos/perception/object_scene_registration.py`

### 7a. YOLO-E vs Standard YOLO

Standard YOLO (Section 3) uses a fixed vocabulary of 80 COCO classes. You
cannot ask it to detect "the red coffee mug on the left table".

YOLO-E (YOLO with Embeddings) supports two modes:

| Mode | Model | How to use | When |
|---|---|---|---|
| `LRPC` (label-free) | `yoloe-11s-seg-pf.pt` | No prompts — detects any object | Background scanning |
| `PROMPT` | `yoloe-11s-seg.pt` | Text or visual prompts | Agent requests specific object |

### 7b. The `detect()` Skill

```python
@skill  # dimos/perception/object_scene_registration.py:224
def detect(self, *prompts: str) -> str:
    """Detect objects in the current camera view.

    Args:
        *prompts: Object descriptions to search for.
                  E.g. detect("person", "coffee mug", "red chair")
                  If no prompts given, detects all visible objects.

    Returns:
        String listing detected objects with their object_id UUIDs.
    """
```

Example agent usage:
```
Agent calls: detect("person", "laptop")
→ YOLO-E runs with PROMPT mode, text prompts = ["person", "laptop"]
→ Returns: "Found: person (id: abc123-...), laptop (id: def456-...)"
```

The `object_id` UUID is stable within the `ObjectDB` session — the agent can
later call `navigate_with_text("go to abc123-...")` to navigate to that object.

---

## 8. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GO2Connection (dimos/robot/unitree/go2/connection.py)                  │
│                                                                          │
│  color_image: Out[Image]  ──────────────────────────────────────────┐   │
│  lidar:  Out[PointCloud2] ──────────────────────────────────────┐    │   │
│  odom:   Out[PoseStamped] ───────────────────────────────────┐   │    │   │
└─────────────────────────────────────────────────────────────────────────┘
                                                               │   │    │
                                ┌──────────────────────────────┘   │    │
                                │  SPATIAL MEMORY                   │    │
                                │  (spatial_perception.py)          │    │
                                │  • every 1s or 10cm movement      │    │
                                │  • CLIP 512-dim embedding          │    │
                                │  • ChromaDB storage               │    │
                                │  • query_by_text() → nav skill    │    │
                                └───────────────────────────────────┘    │
                                                                         │
          ┌──────────────────────────────────────────────────────────────┘
          │  DETECTION 2D (module2D.py:54)
          │
          │  sharp_image_stream()         ← 10 Hz cap + sharpness filter
          │       ↓
          │  Yolo2DDetector.process_image()
          │  model.track(conf=0.5, iou=0.6, persist=True)
          │       ↓
          │  ImageDetections2D (bboxes + track_ids)
          │       ↓
          ├──► Out[Detection2DArray]  ──► LCM bus → Foxglove / Rerun viz
          │
          │  DETECTION 3D (module3D.py:45)
          │  (inherits Detection2DModule, adds PointCloud2 input)
          │
          │  align_timestamped(detections, pointcloud, tol=250ms)
          │       ↓
          │  process_frame()
          │    for each detection:
          │      Detection3DPC.from_2d(
          │        detection,
          │        world_pointcloud,    ← LiDAR scan
          │        camera_info,        ← intrinsics K matrix
          │        world_to_optical_transform)
          │       ↓
          │  ImageDetections3DPC (detections with 3D pointclouds)
          │       ↓
          │  ObjectDB.add_objects()
          │    ├── _match() → track_id → distance → create new
          │    └── _check_promotion() → pending (0-5) → permanent (6+)
          │       ↓
          │  ObjectDB._objects (permanent objects only)
          │       ↓
          └──► agent_encode() → LLM tool result
               find_nearest() → Navigation goal pose
```

---

## 9. How Agents Consume Perception Outputs

There are three routes by which the LLM agent interacts with the perception
pipeline:

### Route 1 — Direct visual query: `ask_vlm()`

**File:** `dimos/perception/detection/module3D.py:114`

```python
@skill
def ask_vlm(self, question: str) -> str:
    """Ask a visual model about the current robot camera view.
    Example: 'Is there a banana in the trunk?'
    """
    model = QwenVlModel()
    image = self.color_image.get_next()
    return model.query(image, question)
```

Use case: the agent wants a yes/no or descriptive answer about what is currently
visible. Bypasses YOLO entirely — sends raw image to Qwen-VL.

### Route 2 — Navigation: `navigate_with_text()`

**File:** `dimos/agents/skills/navigation.py:116`

This skill tries three strategies in order:
1. **Tagged location lookup** — did the user previously call `tag_location("kitchen")`?
   If yes, navigate there directly.
2. **Visual detection** — use `ObjectSceneRegistrationModule.detect()` to find the
   object in the current camera view, then use visual servoing or 3D navigation.
3. **Semantic map fallback** — call `SpatialMemory.query_by_text(query)` to find
   the stored frame (and thus 3D pose) that matches the query best, then navigate
   to that pose.

### Route 3 — Explicit detection: `detect(*prompts)`

**File:** `dimos/perception/object_scene_registration.py:224`

The agent explicitly asks for detection of named objects. Returns `object_id`
UUIDs that can be used as navigation targets or referenced in later tool calls.

---

## 10. Blueprint Wiring — How It All Connects

The perception modules are not manually wired. Instead, `autoconnect()` matches
module input and output ports by name and type automatically.

**Inheritance chain for `unitree-go2-spatial`:**

```
unitree_go2_basic  (dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py)
  GO2Connection (WebRTC → color_image, lidar, odom Out streams)
  DepthModule   (front depth camera)
  KeyboardTeleop

    └─ unitree_go2  (blueprints/smart/unitree_go2.py)
         VoxelMapper    In[PointCloud2] ← lidar
         CostMapper     In[occupancy_grid]
         AStarPlanner   In[costmap]  Out[cmd_vel]
         FrontierExplorer

             └─ unitree_go2_spatial  (blueprints/smart/unitree_go2_spatial.py)
                  SpatialMemory  In[Image] ← color_image, In[PoseStamped] ← odom
                  Utilization    (CPU/GPU/memory monitoring)

                      └─ unitree_go2_agentic  (blueprints/agentic/unitree_go2_agentic.py)
                           Agent(model="gpt-4o")
                           NavigationSkillContainer
                           PersonFollowSkillContainer
                           UnitreeSkillContainer
                           WebInput (port 5555)
                           SpeakSkill
```

**Excalidraw diagram:** `docs/architecture/go2_perception_pipeline.excalidraw`

---

*Sources:*
- `dimos/perception/detection/detectors/yolo.py`
- `dimos/perception/detection/module2D.py`
- `dimos/perception/detection/module3D.py`
- `dimos/perception/detection/objectDB.py`
- `dimos/perception/spatial_perception.py`
- `dimos/perception/object_scene_registration.py`
- `dimos/robot/unitree/go2/connection.py`
