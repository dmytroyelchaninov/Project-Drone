# Real-time Simulation System Guide

## Overview

The real-time simulation system provides interactive drone simulation with both manual keyboard navigation and AI autonomous modes. It includes background physics validation, comprehensive logging, and serves as a foundation for reinforcement learning model training.

## Features

### 🎮 Manual Control Mode

- **Keyboard Navigation**: WASD for horizontal movement, Space/C for vertical
- **Attitude Control**: QE for yaw, RF for pitch, TG for roll
- **Safety Features**: Emergency stop (ESC), hover mode (H), landing mode (L)
- **Control Modes**: Velocity, position, attitude, and direct thrust control

### 🤖 AI Autonomous Mode

- **Reinforcement Learning**: PyTorch-based DQN or simple Q-learning fallback
- **Obstacle Avoidance**: Dynamic obstacle detection and avoidance
- **Waypoint Navigation**: Multi-waypoint mission planning and execution
- **Adaptive Learning**: Real-time policy updates and experience replay

### 🔄 Hybrid Mode

- **Seamless Switching**: Automatic switching between manual and AI control
- **Human-AI Collaboration**: Manual override of AI decisions when needed
- **Learning from Demonstration**: AI observes manual control for training

### 🔍 Background Validation

- **Real-time Physics Checking**: Continuous validation of simulation physics
- **Anomaly Detection**: Automatic detection of unrealistic behavior
- **Auto-correction**: Intelligent correction of physics violations
- **Performance Monitoring**: Real-time performance metrics and alerts

### 📝 Comprehensive Logging

- **Structured Data**: Machine-readable logs optimized for ML analysis
- **Multi-format Output**: JSON, text logs, and human-readable reports
- **Real-time Metrics**: Live performance and validation statistics
- **Session Management**: Organized log directories with metadata

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install pygame matplotlib torch torchvision torchaudio

# Verify installation
python examples/simple_realtime_test.py
```

### Basic Usage

1. **Console Mode (Recommended for Testing)**

```bash
# Manual control
python examples/console_sim.py --mode manual

# AI autonomous mode
python examples/console_sim.py --mode ai

# Hybrid mode
python examples/console_sim.py --mode hybrid
```

2. **Full Real-time Interface**

```bash
# Default environment with manual control
python examples/realtime_simulation.py --mode manual

# Challenging environment for AI training
python examples/realtime_simulation.py --mode ai --environment challenging

# Custom parameters
python examples/realtime_simulation.py --mode hybrid --mass 2.0 --rtf 0.5
```

## Control Schemes

### Manual Control

| Key     | Action           | Description                                      |
| ------- | ---------------- | ------------------------------------------------ |
| W/S     | Forward/Backward | Pitch control for longitudinal movement          |
| A/D     | Left/Right       | Roll control for lateral movement                |
| Space/C | Up/Down          | Throttle control for vertical movement           |
| Q/E     | Yaw Left/Right   | Heading control                                  |
| R/F     | Pitch Up/Down    | Direct pitch control                             |
| T/G     | Roll Left/Right  | Direct roll control                              |
| H       | Hover            | Engage autonomous hover mode                     |
| L       | Land             | Initiate landing sequence                        |
| ESC     | Emergency Stop   | Immediate motor shutdown                         |
| 1/2/3/4 | Control Modes    | Switch between velocity/position/attitude/thrust |

### Console Commands

| Command                     | Description             |
| --------------------------- | ----------------------- |
| `start`                     | Start simulation        |
| `stop`                      | Stop simulation         |
| `pause`                     | Pause/resume simulation |
| `mode <manual\|ai\|hybrid>` | Change control mode     |
| `status`                    | Show detailed status    |
| `save`                      | Save simulation data    |
| `quit`                      | Exit simulation         |

## Environment Configuration

### Default Environment

- 3 obstacles in a simple navigation scenario
- 5 waypoints forming a basic mission
- Suitable for testing and demonstration

### Challenging Environment

- Dense obstacle field with 10+ obstacles
- Complex waypoint patterns requiring advanced navigation
- Ideal for AI training and stress testing

### Custom Environment

```python
# Create custom obstacles
obstacles = [
    Obstacle(
        position=np.array([x, y, z]),
        size=np.array([width, height, depth]),
        shape="box"  # or "sphere"
    )
]

# Create custom waypoints
waypoints = [
    Waypoint(
        position=np.array([x, y, z]),
        tolerance=0.5  # meters
    )
]
```

## AI Training Configuration

### Basic RL Configuration

```python
config = RLConfig(
    learning_rate=3e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    buffer_size=100000,
    batch_size=64
)
```

### Training Process

1. **Exploration Phase**: Random actions to explore environment
2. **Learning Phase**: Experience replay and policy updates
3. **Exploitation Phase**: Using learned policy for navigation
4. **Evaluation**: Periodic testing without exploration

### Reward Structure

- **Distance Reward**: +1.0 for moving closer to waypoints
- **Obstacle Penalty**: -10.0 for collision or near-miss
- **Goal Reward**: +100.0 for reaching waypoints
- **Crash Penalty**: -100.0 for collisions
- **Time Penalty**: -0.01 per step (encourages efficiency)

## Background Validation

### Physics Constraints

- **Real-time Factor**: 0.01x to 100x simulation speed
- **Energy Efficiency**: 10% to 100% realistic efficiency
- **Velocity Limits**: Maximum 20 m/s in any direction
- **Acceleration Limits**: Maximum 50 m/s² total acceleration
- **Attitude Rates**: Maximum 10 rad/s angular velocity

### Validation Levels

- **WARNING**: Minor physics inconsistencies (logged)
- **CRITICAL**: Significant violations (auto-corrected)
- **FATAL**: Simulation-breaking errors (emergency stop)

### Auto-correction

- Velocity clamping to realistic limits
- Energy efficiency normalization
- Quaternion renormalization
- State vector validation

## Logging System

### Log Structure

```
logs/
├── session_YYYYMMDD_HHMMSS/
│   ├── README.md                 # Human-readable guide
│   ├── session_summary.json      # Complete session data
│   ├── llm_analysis_data.json    # ML-optimized data
│   ├── detailed.log              # Chronological text log
│   └── test_*.json               # Individual test logs
```

### Data Format

```json
{
  "session_info": {
    "mode": "hybrid",
    "duration": 120.5,
    "real_time_factor": 1.0
  },
  "simulation_data": {
    "time": [0.0, 0.01, 0.02, ...],
    "position": [[0,0,-2], [0,0,-1.99], ...],
    "velocity": [[0,0,0], [0,0,0.1], ...],
    "control_thrust": [14.7, 14.8, 14.6, ...],
    "rewards": [0, -0.01, -0.02, ...]
  },
  "validation_events": [...],
  "anomalies_detected": [...]
}
```

## Performance Optimization

### Real-time Performance

- **Target**: 500 Hz simulation rate (2ms time step)
- **Minimum**: 100 Hz for stable control
- **Background Tasks**: Validation and logging run at 10 Hz

### Memory Management

- **Experience Buffer**: Circular buffer with configurable size
- **Log Rotation**: Automatic cleanup of old log files
- **State History**: Limited history for performance analysis

### Multi-threading

- **Main Thread**: User interface and command processing
- **Simulation Thread**: Physics and control updates
- **Validation Thread**: Background physics checking
- **Logging Thread**: Asynchronous data writing

## Troubleshooting

### Common Issues

1. **Pygame Not Found**

```bash
pip install pygame
```

2. **PyTorch Not Available**

- Falls back to simple Q-learning
- Install PyTorch for full RL capabilities

3. **High CPU Usage**

- Reduce real-time factor
- Decrease validation frequency
- Limit log detail level

4. **Physics Anomalies**

- Check mass and inertia parameters
- Verify control gains are reasonable
- Review time step size

### Debug Mode

```bash
# Enable detailed logging
python examples/realtime_simulation.py --log-session debug_session

# Run with validation disabled
python examples/console_sim.py --no-validation

# Slow motion for debugging
python examples/realtime_simulation.py --rtf 0.1
```

## Integration with ML Pipelines

### Training Data Export

```python
# Export training data
sim.save_data("training_data.json")

# Load for ML analysis
with open("training_data.json", "r") as f:
    data = json.load(f)

positions = np.array(data["simulation_data"]["position"])
rewards = np.array(data["simulation_data"]["rewards"])
```

### Model Integration

```python
# Save trained model
rl_controller.save_model("trained_model.pth")

# Load for inference
rl_controller.load_model("trained_model.pth")
rl_controller.set_mode(RLMode.INFERENCE)
```

### Batch Training

```python
# Run multiple training episodes
for episode in range(1000):
    sim.reset_environment()
    sim.run_episode()

    if episode % 100 == 0:
        rl_controller.save_model(f"checkpoint_{episode}.pth")
```

## Advanced Features

### Custom Controllers

```python
class CustomController(BaseController):
    def update(self, reference, state, dt):
        # Implement custom control logic
        return ControllerOutput(thrust=thrust, moment=moment)

# Register custom controller
sim.register_controller("custom", CustomController())
```

### Environment Scripting

```python
# Dynamic environment changes
def update_environment(time):
    if time > 30.0:
        # Add moving obstacle
        sim.add_obstacle(MovingObstacle(...))

    if time > 60.0:
        # Change wind conditions
        sim.environment.set_wind_gust(...)

sim.set_environment_callback(update_environment)
```

### Real-time Visualization

```python
# Enable live plotting
sim.enable_live_plots(["position", "velocity", "rewards"])

# Custom visualization
def custom_plot(data):
    plt.plot(data["time"], data["altitude"])

sim.add_plot_callback(custom_plot)
```

## Future Enhancements

### Planned Features

- **Multi-drone Simulation**: Support for swarm simulations
- **Advanced Physics**: More detailed aerodynamics and sensor models
- **VR Integration**: Virtual reality interface for immersive control
- **Cloud Training**: Distributed training across multiple instances
- **Web Interface**: Browser-based control and monitoring

### Research Applications

- **Reinforcement Learning**: Advanced RL algorithm testing
- **Sim-to-Real Transfer**: Bridge simulation and real-world deployment
- **Human-AI Interaction**: Study of human-AI collaborative control
- **Safety Research**: Validation of safety-critical flight systems

## Support and Contributing

### Getting Help

- Check the troubleshooting section above
- Review log files for error details
- Run diagnostic tests: `python examples/simple_realtime_test.py`

### Contributing

- Follow the existing code style and patterns
- Add comprehensive tests for new features
- Update documentation for any changes
- Submit pull requests with clear descriptions

### Reporting Issues

- Include system information and Python version
- Provide minimal reproduction steps
- Attach relevant log files
- Specify expected vs actual behavior
