# Enhanced Logging Guide for Real-Time Drone Simulation

## Overview

Your drone simulation project **already has comprehensive logging infrastructure** that captures detailed information during real-time GUI simulation. This guide shows you how to utilize all the existing logging capabilities to monitor everything during simulation.

## 🔍 Current Logging Infrastructure

### 1. **Background Validation Logging** ✅ ACTIVE

- **Location**: `logs/background_validator_*/`
- **Captures**: Real-time physics validation every 0.1 seconds
- **Data**: Position, velocity, control inputs, physics violations, corrections
- **Format**: Detailed timestamped logs + JSON analysis data

### 2. **Real-Time Simulation Logging** ✅ ACTIVE

- **Location**: `logs/realtime_simulation_*/`
- **Captures**: Session information, performance metrics, errors
- **Data**: Duration, status, warnings, system info
- **Format**: Structured logs with session summaries

### 3. **Enhanced Console Logging** ✅ ACTIVE

- **Output**: Real-time console display
- **Captures**: Status, controls, AI progress, system metrics
- **Features**: FPS tracking, CPU/memory usage, performance analysis
- **Format**: Live formatted output with emojis and timestamps

### 4. **Detailed JSON Logging** ✅ AVAILABLE

- **Output**: `detailed_simulation_log_*.json` files
- **Captures**: Complete simulation state history
- **Data**: Position/velocity/acceleration history, control inputs, AI decisions, system states
- **Format**: Structured JSON for analysis

## 📊 What Gets Logged

### Real-Time Status Information

```
📍 Step 41: Pos=(0.00, 0.00, -2.00) Vel=(0.00, 0.00, 0.00) Speed=0.00m/s Mode=manual
   ⚡ Thrust: 13.56N | FPS: 15.2 | CPU: 12.3% | Mem: 45.1%
   📈 Metrics: 150 samples, Avg FPS: 14.8
   🔬 Physics: ✅ | Violations: 0
```

### Control Input Analysis

```
🎮 Control: Thrust=15.20N, Moment=(0.00, 2.00, 0.00)
   📊 Control Stats: Magnitude=2.00, Total Commands=125
```

### AI Learning Progress

```
🤖 AI Progress: Episodes=7, Reward=-8.87, Success=60.00%
   🧠 Learning: Loss=0.0234, LR=0.001000, Explore=0.15
   📊 Trend: 📈 Recent avg reward: 12.45
```

### System Events

```
[00:17:32] 🚀 START: Starting simulation interface
[00:18:45] 🎮 Control: Manual control activated
[00:19:12] 🔬 Physics: Validation passed - 0 violations
[00:20:30] 🛑 STOP: Simulation stopped
```

### Physics Validation (Background)

```
INFO: validation_simulation_step: {
    "test_name": "Real-time Simulation",
    "violations": 0,
    "warnings": 0,
    "corrections": 0,
    "is_valid": true
}
```

## 🚀 How to Use Enhanced Logging

### Basic Usage (Already Working)

```bash
# Standard simulation with all logging active
python examples/realtime_simulation.py --mode manual --log
```

### Enhanced Usage with Detailed Logging

```bash
# AI mode with comprehensive logging
python examples/realtime_simulation.py --mode ai --detailed-log

# Challenging environment with full analysis
python examples/realtime_simulation.py --mode ai --environment challenging --detailed-log

# Hybrid mode with custom settings
python examples/realtime_simulation.py --mode hybrid --detailed-log --rtf 0.5
```

### Command Line Options

- `--log` - Enable console logging (default: True)
- `--no-log` - Disable console logging
- `--detailed-log` - Save comprehensive JSON log file
- `--log-interval` - Set status logging interval (default: 1.0s)

## 📁 Log File Locations and Contents

### Background Validation Logs

```
logs/background_validator_YYYYMMDD_HHMMSS/
├── background_validator_detailed.log     # Timestamped validation events
├── session_summary.json                  # Session statistics
├── llm_analysis_data.json               # Analysis-ready data
└── README.md                            # Log description
```

### Real-Time Simulation Logs

```
logs/realtime_simulation_YYYYMMDD_HHMMSS/
├── realtime_simulation_detailed.log     # Session events
├── test_real-time_simulation_session.json  # Session data
└── session_summary.json                 # Performance summary
```

### Detailed JSON Logs (with --detailed-log)

```
detailed_simulation_log_YYYYMMDD_HHMMSS.json
├── session_info                         # Duration, steps, timing
├── performance_summary                  # FPS, CPU, memory stats
├── detailed_history                     # Complete state history
│   ├── timestamps                       # Time series
│   ├── positions                        # Position history
│   ├── velocities                       # Velocity history
│   ├── accelerations                    # Acceleration history
│   ├── control_inputs                   # Control command history
│   ├── ai_decisions                     # AI decision history
│   └── system_states                    # System state snapshots
└── performance_metrics                  # Real-time metrics
```

## 🔧 Accessing Log Data Programmatically

### Reading Background Validation Data

```python
import json
from pathlib import Path

# Find latest background validator log
logs_dir = Path("logs")
bg_logs = [d for d in logs_dir.glob("background_validator_*")]
latest_log = max(bg_logs, key=lambda x: x.stat().st_mtime)

# Read session summary
with open(latest_log / "session_summary.json") as f:
    summary = json.load(f)
    print(f"Duration: {summary['session_duration']:.1f}s")
    print(f"Validation events: {summary['total_tests']}")
```

### Reading Detailed Simulation Data

```python
# Read detailed JSON log
log_files = list(Path(".").glob("detailed_simulation_log_*.json"))
if log_files:
    latest_detailed = max(log_files, key=lambda x: x.stat().st_mtime)

    with open(latest_detailed) as f:
        data = json.load(f)

        # Access performance metrics
        fps_stats = data['performance_summary']['fps_stats']
        print(f"Average FPS: {fps_stats['mean']:.1f}")

        # Access position history
        positions = data['detailed_history']['positions']
        print(f"Recorded {len(positions)} position samples")

        # Access AI decisions (if available)
        ai_decisions = data['detailed_history']['ai_decisions']
        print(f"AI made {len(ai_decisions)} decisions")
```

### Real-Time Log Monitoring

```python
import time
from drone_sim.utils.background_validator import BackgroundValidator

# Monitor validation in real-time
validator = BackgroundValidator()
validator.start_background_validation()

def on_anomaly(anomaly_report):
    print(f"⚠️ Anomaly detected: {anomaly_report.message}")

validator.register_anomaly_callback(on_anomaly)

# Get real-time stats
stats = validator.get_real_time_stats()
print(f"Queue size: {stats['queue_size']}")
print(f"Tests monitored: {stats['stats']['tests_monitored']}")
```

## 📈 Performance Analysis Examples

### FPS and System Performance

```python
# Analyze performance from detailed log
def analyze_performance(log_file):
    with open(log_file) as f:
        data = json.load(f)

    fps_data = data['performance_metrics']['fps']
    cpu_data = data['performance_metrics']['cpu_usage']

    print(f"FPS: min={min(fps_data):.1f}, max={max(fps_data):.1f}, avg={sum(fps_data)/len(fps_data):.1f}")
    print(f"CPU: min={min(cpu_data):.1f}%, max={max(cpu_data):.1f}%, avg={sum(cpu_data)/len(cpu_data):.1f}%")
```

### Control Input Analysis

```python
# Analyze control patterns
def analyze_controls(log_file):
    with open(log_file) as f:
        data = json.load(f)

    controls = data['detailed_history']['control_inputs']

    thrust_values = [c['thrust'] for c in controls]
    moment_magnitudes = [sum(m**2 for m in c['moment'])**0.5 for c in controls]

    print(f"Thrust range: {min(thrust_values):.1f}N to {max(thrust_values):.1f}N")
    print(f"Average moment magnitude: {sum(moment_magnitudes)/len(moment_magnitudes):.2f}")
```

### AI Learning Analysis

```python
# Analyze AI performance trends
def analyze_ai_learning(log_file):
    with open(log_file) as f:
        data = json.load(f)

    ai_decisions = data['detailed_history']['ai_decisions']

    if ai_decisions:
        rewards = [d['stats'].get('reward', 0) for d in ai_decisions]
        episodes = [d['stats'].get('episodes', 0) for d in ai_decisions]

        print(f"Learning progress: {len(episodes)} decision points")
        print(f"Reward trend: {rewards[0]:.2f} → {rewards[-1]:.2f}")
```

## 🧪 Testing and Validation

### Test Enhanced Logging

```bash
# Test all logging components
python examples/test_enhanced_logging.py --test

# Analyze existing logs
python examples/test_enhanced_logging.py --analyze

# Demo logging features
python examples/test_enhanced_logging.py --demo
```

### Verify Logging is Working

1. **Background Validation**: Check `logs/background_validator_*/` for recent files
2. **Console Output**: Look for real-time status updates with emojis
3. **Detailed Logs**: Use `--detailed-log` flag and check for JSON files
4. **Performance Metrics**: Monitor FPS, CPU, memory in console output

## 🎯 Key Benefits

### Real-Time Monitoring

- **Live Performance**: FPS, CPU, memory usage displayed in real-time
- **Physics Validation**: Automatic detection of physics violations
- **Control Analysis**: Real-time thrust and moment magnitude tracking
- **AI Progress**: Learning trends and decision analysis

### Comprehensive Data Capture

- **Complete History**: Every position, velocity, acceleration sample
- **Control Inputs**: All thrust and moment commands with timestamps
- **AI Decisions**: Learning progress, rewards, exploration data
- **System State**: CPU, memory, disk usage at critical events

### Analysis-Ready Output

- **Structured JSON**: Easy to parse and analyze programmatically
- **Timestamped Events**: Precise timing information for all events
- **Performance Metrics**: Statistical summaries and trends
- **Error Tracking**: Detailed error logs with system context

## 🔍 Example Session Output

When you run the simulation, you'll see output like this:

```
🚀 Starting Real-time Drone Simulation
==================================================
Mode: manual
Environment: default
Mass: 1.5 kg
Real-time factor: 1.0x
Time step: 0.002 s
Console logging: enabled
GUI mode: enabled
==================================================

🔍 Enhanced console logging enabled
📊 Capturing: Status, Controls, AI, Performance, Physics, System Metrics
============================================================

Environment: 3 obstacles, 5 waypoints
[00:17:26] 🔧 SETUP: Environment loaded: 3 obstacles, 5 waypoints
✅ GUI initialized successfully
[00:17:26] ⚙️ INIT: Interface initialized in manual mode
[00:17:26] 🚀 START: Starting simulation interface

📍 Step 1: Pos=(0.00, 0.00, -2.00) Vel=(0.00, 0.00, 0.00) Speed=0.00m/s Mode=manual
   ⚡ Thrust: 13.56N | FPS: 15.2 | CPU: 12.3% | Mem: 45.1%
   📈 Metrics: 1 samples, Avg FPS: 15.2

🎮 Control: Thrust=15.20N, Moment=(0.00, 2.00, 0.00)
   📊 Control Stats: Magnitude=2.00, Total Commands=1

[Background validation running every 0.1s...]
INFO: validation_simulation_step: {"test_name": "Real-time Simulation", "violations": 0, "warnings": 0, "corrections": 0, "is_valid": true}

[00:17:32] 🛑 STOP: Simulation stopped

==================================================
📊 SIMULATION PERFORMANCE SUMMARY
==================================================
Duration: 30.5s
Total Steps: 15250
Average FPS: 14.8
CPU Usage: 15.2% (max: 28.5%)
Memory Usage: 47.3% (max: 52.1%)
Control Commands: 1250
AI Updates: 0
==================================================

💾 Detailed log saved to: detailed_simulation_log_20250624_004530.json
[00:17:32] 🔌 SHUTDOWN: Simulation shutdown complete
```

## ✅ Summary

Your drone simulation **already has comprehensive logging** that captures:

1. **Real-time physics validation** (every 0.1s)
2. **Complete simulation state history**
3. **Performance metrics** (FPS, CPU, memory)
4. **Control input analysis**
5. **AI learning progress** (when applicable)
6. **System monitoring** with detailed error tracking

All this data is **automatically captured** during GUI simulation and saved to structured log files for analysis. The enhanced console logging provides real-time feedback, while the background systems ensure comprehensive data collection for post-simulation analysis.

**No additional setup required** - just run your simulation and all logging is active!
