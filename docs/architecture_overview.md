# Drone Simulation Architecture Overview

## Project Vision

The Drone Simulation project is a physics-accurate, reactive simulator for multirotor drones with comprehensive acoustic analysis capabilities. It provides a modular, extensible platform for drone research, control system development, and acoustic modeling.

## Core Design Principles

### 1. Physics Accuracy

- 6-DOF rigid body dynamics with quaternion-based attitude representation
- RK4 numerical integration for stability and accuracy
- Real-time capable simulation with configurable time steps

### 2. Modularity

- Component-based architecture allowing easy extension
- Clear separation between physics, control, and analysis systems
- Plugin-style sensor and controller registration

### 3. Real-time Performance

- Efficient numerical algorithms optimized for real-time execution
- Configurable real-time factors for faster-than-real-time simulation
- Event-driven parameter changes without simulation restart

### 4. Acoustic Modeling

- Advanced propeller noise modeling using Ffowcs Williams-Hawkings equations
- Multiple noise sources: thickness noise, loading noise, broadband noise
- A-weighting and octave band analysis for realistic sound characterization

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DRONE SIMULATION SYSTEM                  │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Examples/     │  │   Analysis/     │  │   UI/       │ │
│  │   Scripts       │  │   Tools         │  │   Interface │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Core Simulation Engine                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Simulator     │  │  State Manager  │  │ Event System│ │
│  │   (RK4 Loop)    │  │  (13-DOF State) │  │ (Pub/Sub)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Physics Engine                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Rigid Body     │  │  Aerodynamics   │  │ Environment │ │
│  │  Dynamics       │  │  & Propellers   │  │ & Wind      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Control Systems                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  PID Controller │  │  Base Controller│  │  ML-Ready   │ │
│  │  (3-Layer)      │  │  Interface      │  │  Hooks      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Sensors & Analysis                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   IMU/GPS       │  │  Noise Model    │  │  Data       │ │
│  │   Sensors       │  │  (Acoustic)     │  │  Logging    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

### 1. Simulation Loop (Main Thread)

```
Initialize → Start Timer → Physics Step → Control Update →
Sensor Update → State Validation → Event Processing →
Time Advance → Check Termination → Loop
```

### 2. State Vector Flow

```
13-DOF State Vector: [px, py, pz, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]
                                    ↓
                            State Manager
                          (Validation & History)
                                    ↓
                            Physics Engine
                         (Force/Moment Calculation)
                                    ↓
                            RK4 Integrator
                          (Numerical Integration)
                                    ↓
                            Updated State Vector
```

### 3. Control Loop Flow

```
Reference Trajectory → Position Controller (PID) → Attitude Reference →
Attitude Controller (PID) → Rate Reference → Rate Controller (PID) →
Motor Commands → Propeller Forces → Physics Engine
```

## Key Algorithms

### 1. Quaternion-Based Attitude Dynamics

- **Why Quaternions**: Avoid gimbal lock, more numerically stable than Euler angles
- **Normalization**: Automatic quaternion normalization prevents drift
- **Integration**: Special quaternion integration using angular velocity

### 2. RK4 Integration Scheme

- **4th Order Accuracy**: Superior to Euler integration for long simulations
- **Fixed Time Step**: 2ms default for real-time performance
- **State Derivatives**: Computed from physics engine at each RK4 stage

### 3. Three-Layer PID Control

- **Position Loop**: Outer loop controlling position in inertial frame
- **Attitude Loop**: Middle loop controlling orientation
- **Rate Loop**: Inner loop controlling angular rates (fastest response)

### 4. Propeller Noise Modeling

- **Thickness Noise**: From propeller blade volume displacement
- **Loading Noise**: From aerodynamic forces on blades
- **Broadband Noise**: Turbulence and tip vortex effects

## Configuration System

### Hierarchical Configuration

```
Global Config
├── Simulation Config (time step, duration, real-time factor)
├── Drone Config (mass, inertia, propeller layout)
├── Controller Config (PID gains, limits)
├── Environment Config (gravity, wind, atmosphere)
└── Sensor Config (noise models, update rates)
```

### YAML-Based Presets

- Standardized drone configurations
- Easy parameter tuning and experimentation
- Version-controlled configuration management

## Performance Characteristics

### Computational Complexity

- **Physics Step**: O(n) where n = number of propellers
- **Control Update**: O(1) for PID controllers
- **State Integration**: O(1) for fixed-size state vector
- **Memory Usage**: O(h) where h = history buffer size

### Real-time Capability

- **Target**: 500 Hz simulation rate (2ms time step)
- **Typical Performance**: 1000+ Hz on modern hardware
- **Scalability**: Linear with number of vehicles

## Extension Points

### 1. Custom Controllers

```python
class CustomController(BaseController):
    def compute_control(self, state, reference, dt):
        # Implement custom control logic
        return control_output
```

### 2. Custom Sensors

```python
class CustomSensor:
    def update(self, state, time, dt):
        # Implement sensor model
        return sensor_data
```

### 3. Custom Environments

```python
class CustomEnvironment(Environment):
    def get_wind_velocity(self, position, time):
        # Implement custom wind model
        return wind_vector
```

## Quality Assurance

### Validation Framework

- **State Validation**: Physical bounds checking
- **Numerical Stability**: NaN/Inf detection and recovery
- **Performance Monitoring**: Real-time factor tracking

### Testing Strategy

- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end simulation scenarios
- **Performance Tests**: Benchmarking and profiling

## Future Roadmap

### Phase 1 (Current)

- ✅ Core physics engine
- ✅ PID control system
- ✅ Basic acoustic modeling
- ✅ Configuration system

### Phase 2 (Planned)

- 🔄 Advanced control algorithms (LQR, MPC)
- 🔄 Machine learning integration
- 🔄 Multi-vehicle simulation
- 🔄 Real-time visualization

### Phase 3 (Future)

- 📋 Hardware-in-the-loop support
- 📋 Distributed simulation
- 📋 Cloud deployment
- 📋 VR/AR visualization

## Getting Started

For new developers:

1. Start with `examples/basic_simulation.py`
2. Review `docs/component_reference.md` for detailed API
3. Examine `docs/algorithm_details.md` for mathematical foundations
4. See `docs/development_guide.md` for contribution guidelines
