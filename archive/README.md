# Drone Simulation Project

A physics-accurate, reactive simulator for multirotor drones with comprehensive acoustic analysis capabilities.

## Features

- **Physics-Accurate Simulation**: 6-DOF rigid body dynamics with quaternion-based attitude representation
- **Reactive Parameter Tuning**: Real-time adjustment of physical parameters (propeller radius, mass, payload)
- **Advanced Propeller Modeling**: Configurable propeller models with per-motor specifications
- **Acoustic Analysis**: Real-time noise spectrum calculation and SPL measurement
- **Modular Control System**: PID controllers with ML-ready interfaces
- **Environmental Modeling**: Wind, turbulence, and atmospheric effects
- **Real-time Visualization**: 3D visualization with parameter controls

## Project Structure

```
drone_sim/
├── core/                    # Base simulation infrastructure
│   ├── simulator.py         # Main simulation loop with fixed-time stepping
│   ├── state_manager.py     # Handles drone state transitions
│   └── event_system.py      # Pub-sub for parameter changes
│
├── physics/                 # Physical modeling
│   ├── aerodynamics/
│   │   ├── propeller.py     # Configurable propeller models (per-motor)
│   │   └── noise_model.py   # Acoustic noise spectrum calculation
│   ├── rigid_body.py        # 6DOF dynamics
│   └── environment.py       # Gravity, wind, disturbances
│
├── control/                 # Flight control stack
│   ├── base_controller.py   # Interface for all controllers
│   ├── pid_controller.py    # Classical PID implementation
│   ├── mixer.py             # Motor mixing algorithms
│   └── adaptive/            # ML-ready components
│       └── rl_interface.py  # Hook for reinforcement learning
│
├── sensors/                 # Sensor simulation
│   ├── imu.py               # With configurable noise
│   ├── microphone.py        # For noise measurement at points
│   └── noise_profiles/      # Different noise characteristic presets
│
├── analysis/                # Noise optimization tools
│   ├── fft_processor.py     # Frequency domain analysis
│   ├── optimizer.py         # Parameter optimization routines
│   └── metrics.py           # Noise/performance metrics
│
├── ui/                      # User interface
│   ├── web/                 # Future web interface
│   ├── cli.py               # Command line controls
│   └── visualizer.py        # 3D PyOpenGL visualization
│
├── configs/                 # Parameter configurations
│   ├── drone_presets/       # Different drone configurations
│   └── noise_scenarios/     # Predefined noise measurement setups
│
├── tests/                   # Testing infrastructure
│   ├── unit/                # Module tests
│   └── integration/         # Full system tests
│
└── utils/                   # Supporting code
    ├── math_tools.py        # Vector/matrix operations
    └── data_logger.py       # CSV/ROS bag output
```

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- PyYAML >= 6.0

### Install from source

```bash
git clone <repository-url>
cd drone_project
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

### Install with visualization support

```bash
pip install -e ".[visualization]"
```

## Quick Start

### Basic Simulation

```python
import numpy as np
from drone_sim import (
    Simulator, SimulationConfig,
    RigidBody, RigidBodyConfig,
    PIDController
)

# Create simulation
sim_config = SimulationConfig(dt=0.002, real_time_factor=1.0)
simulator = Simulator(sim_config)

# Set up drone physics
inertia = np.diag([0.02, 0.02, 0.04])
rigid_body = RigidBody(RigidBodyConfig(mass=1.5, inertia=inertia))
simulator.register_physics_engine(rigid_body)

# Set up controller
controller = PIDController()

# Run simulation
simulator.run(duration=10.0)  # 10 seconds
```

### Noise Analysis Example

```python
from drone_sim.physics.aerodynamics import PropellerNoiseModel, ObserverPosition

# Create noise model
propeller_config = {'diameter': 0.24, 'blades': 2}
noise_model = PropellerNoiseModel(propeller_config)

# Set observer position
observer = ObserverPosition(x=5.0, y=0.0, z=-2.0)

# Calculate noise spectrum
noise_data = noise_model.calculate_total_noise(
    rpm=3000,
    thrust=5.0,
    observer=observer
)

# Get noise metrics
metrics = noise_model.get_metrics()
print(f"OASPL: {metrics['oaspl_db']:.1f} dB")
```

## Key Concepts

### Coordinate Frames

- **Inertial Frame**: North-East-Down (NED) coordinate system
- **Body Frame**: Fixed to the drone with X-forward, Y-right, Z-down
- **Propeller Frame**: Individual propeller coordinate systems

### State Representation

The drone state is represented as a 13-DOF vector:

- Position (3 DOF): [x, y, z] in inertial frame
- Orientation (4 DOF): Quaternion [w, x, y, z]
- Linear velocity (3 DOF): [vx, vy, vz] in body frame
- Angular velocity (3 DOF): [wx, wy, wz] in body frame

### Acoustic Modeling

The noise model implements:

- **Thickness Noise**: Ffowcs Williams-Hawkings theory
- **Loading Noise**: Far-field approximation
- **Broadband Noise**: Turbulence-induced noise

## Configuration

### Drone Configuration

Drone configurations are stored in YAML files:

```yaml
# configs/drone_presets/quadcopter_default.yaml
name: "Default Quadcopter"
physics:
  mass: 1.5 # kg
  inertia: [0.02, 0.02, 0.04] # kg⋅m²

propellers:
  count: 4
  layout: "quad_x"
  motors:
    - id: 0
      position: [0.2, 0.2, 0.0]
      propeller:
        diameter: 0.24 # meters
        blades: 2
```

### Control Parameters

```yaml
control:
  pid_gains:
    position:
      kp: [1.0, 1.0, 2.0]
      ki: [0.1, 0.1, 0.2]
      kd: [0.5, 0.5, 1.0]
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black drone_sim/
flake8 drone_sim/
```

### Type Checking

```bash
mypy drone_sim/
```

## Examples

See the `examples/` directory for complete simulation examples:

- `basic_simulation.py`: Basic hover simulation
- `noise_analysis.py`: Acoustic analysis example
- `wind_disturbance.py`: Wind simulation
- `parameter_optimization.py`: Noise optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{drone_sim_2024,
  title={Drone Simulation: Physics-Accurate Multirotor Simulator with Acoustic Analysis},
  author={Drone Simulation Team},
  year={2024},
  url={https://github.com/username/drone-sim}
}
```

## Roadmap

- [ ] GPU-accelerated physics (CuPy integration)
- [ ] Swarm simulation support
- [ ] Machine learning control interfaces
- [ ] Web-based visualization
- [ ] ROS 2 integration
- [ ] Hardware-in-the-loop testing
- [ ] Advanced weather modeling
- [ ] Multi-fidelity acoustic models
