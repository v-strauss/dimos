# Piper Arm

6-DOF manipulator arm from Agilex Robotics, connected via CAN bus.

## Prerequisites

### Dependencies

```bash
# System packages
sudo apt update && sudo apt install ethtool can-utils

# Python SDK
pip install piper-sdk
```

### CAN Bus Setup

Before running any Piper code, you must activate the CAN interface:

```bash
# Basic usage (defaults: can0, 1000000 bitrate)
bash dimos/hardware/manipulators/piper/can_activate.sh

# With custom CAN name and bitrate
bash dimos/hardware/manipulators/piper/can_activate.sh can0 1000000

# With USB hardware address (for multiple CAN adapters)
bash dimos/hardware/manipulators/piper/can_activate.sh can0 1000000 1-2:1.0
```

## Blueprints

All blueprints can be run via `dimos run <blueprint-name>`:

### Real Hardware

| Blueprint | Description | Use Case |
|-----------|-------------|----------|
| `piper-teleop` | CartesianIK + pygame jogger | Interactive teleoperation with keyboard/mouse |
| `piper-manipulation` | Trajectory control + planning module | Motion planning with Meshcat visualization |
| `piper-velocity` | Streaming velocity control | External velocity commands |

### Mock (No Hardware)

| Blueprint | Description |
|-----------|-------------|
| `piper-teleop-mock` | Simulated teleop for testing |
| `piper-manipulation-mock` | Simulated manipulation planning |

## Quick Start

### 1. Teleop (Keyboard Control)

```bash
# Activate CAN first
bash dimos/hardware/manipulators/piper/can_activate.sh

# Run teleop
dimos run piper-teleop
```

Use arrow keys and WASD for cartesian control in the pygame window.

### 2. Motion Planning

```bash
# Terminal 1: Start manipulation stack
dimos run piper-manipulation

# Terminal 2: Interactive client
python -m dimos.manipulation.planning.examples.manipulation_client
```

In the IPython shell:
```python
url()                    # Get Meshcat visualization URL
joints()                 # Get current joint positions
ee()                     # Get end-effector pose
plan_pose(0.3, 0, 0.2)   # Plan to cartesian pose
preview()                # Preview in Meshcat
execute()                # Execute trajectory
```

### 3. Test Without Hardware

```bash
# Mock teleop - no CAN needed
dimos run piper-teleop-mock

# Mock manipulation
dimos run piper-manipulation-mock
```

## Troubleshooting

### CAN Interface Not Found
```bash
# Check if CAN interface exists
ip link show type can

# List USB devices to find CAN adapter
lsusb
```

### Permission Denied
```bash
# Add user to dialout group
sudo usermod -aG dialout $USER
# Log out and back in
```

### Multiple CAN Adapters
```bash
# Find USB addresses
for iface in $(ip -br link show type can | awk '{print $1}'); do
    echo "$iface: $(sudo ethtool -i $iface | grep bus-info | awk '{print $2}')"
done

# Activate specific adapter
bash can_activate.sh can0 1000000 1-2:1.0
```
