# Console Logging Guide

## Overview

The real-time drone simulation now includes comprehensive console logging capabilities that output detailed simulation information to the terminal while the GUI is running. This is perfect for debugging, monitoring, and understanding what's happening during simulation.

## Features

### 🎯 Real-time Status Updates

- **Position & Velocity**: Live tracking of drone position and velocity
- **Control Inputs**: Thrust and moment values being applied
- **AI Progress**: Episode counts, rewards, and success rates
- **System Events**: Simulation start/stop, mode changes, errors

### 📊 Structured Logging

- **Timestamped Events**: All events include precise timestamps
- **Categorized Messages**: Different types of events (SETUP, START, STOP, etc.)
- **Performance Metrics**: Real-time performance and learning statistics
- **Error Tracking**: Detailed error messages and stack traces

## Usage

### Basic Commands

#### 1. GUI with Console Logging (Default)

```bash
python examples/realtime_simulation.py --mode manual --log
```

#### 2. GUI without Console Logging

```bash
python examples/realtime_simulation.py --mode manual --no-log
```

#### 3. AI Mode with Console Logging

```bash
python examples/realtime_simulation.py --mode ai --log
```

#### 4. Different Environments with Logging

```bash
python examples/realtime_simulation.py --environment challenging --log
```

### Command Line Options

| Flag                   | Description                                    | Default   |
| ---------------------- | ---------------------------------------------- | --------- |
| `--log`                | Enable console logging                         | `True`    |
| `--no-log`             | Disable console logging                        | `False`   |
| `--mode <type>`        | Simulation mode (manual/ai/hybrid)             | `manual`  |
| `--environment <type>` | Environment preset (default/challenging/empty) | `default` |
| `--mass <float>`       | Drone mass in kg                               | `1.5`     |
| `--rtf <float>`        | Real-time factor                               | `1.0`     |
| `--log-session <name>` | Enable detailed file logging                   | None      |

## Console Output Examples

### Manual Mode Output

```
🚀 Starting Real-time Drone Simulation
==================================================
Mode: manual
Environment: default
Mass: 1.5 kg
Real-time factor: 1.0x
Console logging: enabled
==================================================

[00:17:32] SETUP: Environment loaded: 3 obstacles, 5 waypoints
[00:17:32] INIT: Interface initialized in manual mode
[00:17:32] START: Starting simulation interface

📍 Step 1: Pos=(0.00, 0.00, -2.00) Vel=(0.00, 0.00, 0.00) Mode=manual
   Thrust: 14.70N
   Step: 1
   Time: 0.002s

🎮 Control: Thrust=15.20N, Moment=(0.00, 2.00, 0.00)

📍 Step 2: Pos=(0.01, 0.00, -2.00) Vel=(0.10, 0.00, 0.00) Mode=manual
   Thrust: 15.20N
```

### AI Mode Output

```
📍 Step 150: Pos=(3.45, 1.20, -1.80) Vel=(1.20, 0.30, 0.10) Mode=ai
   Thrust: 16.80N
   Episodes: 5
   Reward: 23.45

🤖 AI Progress: Episodes=5, Reward=23.45, Success=60.00%

[00:18:45] AI_EPISODE: Episode 6 completed, reward: 28.90
```

### Event Logging

```
[00:17:32] SETUP: Environment loaded: 3 obstacles, 5 waypoints
[00:17:32] INIT: Interface initialized in manual mode
[00:17:32] START: Starting simulation interface
[00:17:45] PAUSE: Simulation paused
[00:17:50] RESUME: Simulation resumed
[00:18:30] STOP: Simulation stopped
[00:18:30] SHUTDOWN: Simulation shutdown complete
```

## Debugging with Console Logging

### 1. Performance Monitoring

Watch the step timing and real-time factor:

```
📍 Step 1000: Pos=(10.50, 2.30, -1.50) Vel=(2.10, 0.50, 0.20) Mode=manual
   Time: 2.000s
```

### 2. Control Input Debugging

Monitor control inputs to understand behavior:

```
🎮 Control: Thrust=18.50N, Moment=(1.50, -0.80, 0.30)
```

### 3. AI Learning Progress

Track learning performance:

```
🤖 AI Progress: Episodes=25, Reward=45.67, Success=80.00%
```

### 4. Error Detection

Catch and understand errors:

```
[00:18:30] ERROR: Physics validation failed: thrust out of bounds
[00:18:30] CORRECTION: Thrust clamped to valid range
```

## Integration with File Logging

### Detailed Session Logging

```bash
python examples/realtime_simulation.py --mode ai --log --log-session "training_run_001"
```

This creates:

- **Console output**: Real-time monitoring
- **File logs**: Detailed JSON logs in `logs/training_run_001_YYYYMMDD_HHMMSS/`

### Log File Structure

```
logs/
├── training_run_001_20241224_001732/
│   ├── session_summary.json
│   ├── test_data.json
│   └── performance_metrics.json
```

## Advanced Usage

### Custom Environment with Logging

```bash
# Create custom environment
python examples/realtime_simulation.py --mode ai --log \
  --environment empty \
  --mass 2.0 \
  --rtf 2.0
```

### Performance Analysis

```bash
# High-speed simulation for training
python examples/realtime_simulation.py --mode ai --log \
  --rtf 5.0 \
  --log-session "fast_training"
```

### Debugging Physics Issues

```bash
# Slow motion for detailed analysis
python examples/realtime_simulation.py --mode manual --log \
  --rtf 0.1 \
  --mass 0.5
```

## Tips and Best Practices

### 1. **Use Logging for Development**

- Always enable logging during development and testing
- Monitor console output to understand system behavior
- Use file logging for training data collection

### 2. **Performance Monitoring**

- Watch step timing to ensure real-time performance
- Monitor memory usage during long AI training sessions
- Check for physics validation warnings

### 3. **AI Training**

- Use console logging to track learning progress
- Monitor reward trends and success rates
- Save logs for post-training analysis

### 4. **Debugging Issues**

- Console logging helps identify control problems
- Physics validation messages indicate constraint violations
- Error messages provide detailed troubleshooting information

## Troubleshooting

### Common Issues

#### 1. **No Console Output**

```bash
# Make sure logging is enabled
python examples/realtime_simulation.py --log
```

#### 2. **Too Much Output**

```bash
# Disable logging if overwhelming
python examples/realtime_simulation.py --no-log
```

#### 3. **Missing AI Progress**

```bash
# Ensure AI mode is selected
python examples/realtime_simulation.py --mode ai --log
```

### Performance Issues

#### 1. **Slow Real-time Factor**

- Console logging adds minimal overhead
- Check system resources and reduce visualization updates

#### 2. **Memory Usage**

- Long logging sessions use memory for history
- Restart simulation periodically for long training runs

## Integration Examples

### 1. **Training Pipeline**

```bash
#!/bin/bash
# Training script with logging
for i in {1..10}; do
    echo "Training run $i"
    python examples/realtime_simulation.py \
        --mode ai \
        --log \
        --log-session "training_run_$i" \
        --rtf 3.0 \
        --environment challenging
done
```

### 2. **Performance Testing**

```bash
# Test different configurations
python examples/realtime_simulation.py --mode manual --log --mass 1.0
python examples/realtime_simulation.py --mode manual --log --mass 2.0
python examples/realtime_simulation.py --mode manual --log --mass 3.0
```

### 3. **Debugging Session**

```bash
# Detailed debugging with slow motion
python examples/realtime_simulation.py \
    --mode hybrid \
    --log \
    --rtf 0.2 \
    --log-session "debug_session"
```

## Conclusion

Console logging provides powerful real-time monitoring and debugging capabilities for the drone simulation. Use it to:

- **Monitor** simulation performance and behavior
- **Debug** control and physics issues
- **Track** AI learning progress
- **Collect** training data
- **Analyze** system performance

The combination of real-time console output and detailed file logging makes it easy to understand and improve your drone simulation and AI training processes.
