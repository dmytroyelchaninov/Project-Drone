# System Relationships and Component Interactions

## Overview

The drone simulation system is built on a modular architecture where components interact through well-defined interfaces. This document explains how each component relates to others and how data flows through the system.

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SIMULATION ORCHESTRATION                         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Simulator     │    │  State Manager  │    │  Event System   │         │
│  │   (Core Loop)   │◄──►│  (13-DOF State) │◄──►│  (Pub/Sub)      │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
└───────────┼───────────────────────┼───────────────────────┼─────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PHYSICS LAYER                                  │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Rigid Body     │◄──►│  Environment    │◄──►│  Propellers     │         │
│  │  (Dynamics)     │    │  (Wind/Gravity) │    │  (Thrust/Torque)│         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
└───────────┼───────────────────────┼───────────────────────┼─────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CONTROL LAYER                                   │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Controllers    │◄──►│  Motor Mixing   │◄──►│  Actuators      │         │
│  │  (PID/LQR/ML)   │    │  (Allocation)   │    │  (Motors/Servos)│         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
└───────────┼───────────────────────┼───────────────────────┼─────────────────┘
            │                       │                       │
            ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SENSING LAYER                                    │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   IMU/GPS       │◄──►│  Noise Models   │◄──►│  Data Fusion    │         │
│  │   (Sensors)     │    │  (Realistic)    │    │  (Estimation)   │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Analysis

### 1. Main Simulation Loop

The core simulation follows this data flow pattern:

```python
def simulation_step():
    # 1. Get current state from State Manager
    current_state = state_manager.get_current_state()

    # 2. Update sensors with current state
    sensor_data = sensors.update(current_state, time, dt)

    # 3. Compute control output
    control_output = controller.compute_control(
        current_state, reference_trajectory, dt
    )

    # 4. Apply control to propellers
    propeller_forces = propellers.compute_forces(control_output.motor_commands)

    # 5. Get environmental forces
    env_forces = environment.get_forces(current_state.position, time)

    # 6. Compute physics derivatives
    total_forces = propeller_forces + env_forces
    derivatives = physics.compute_derivatives(current_state, total_forces)

    # 7. Integrate using RK4
    new_state = rk4_integrate(current_state, derivatives, dt)

    # 8. Update state manager
    state_manager.set_state(new_state, time + dt)

    # 9. Process events
    event_system.process_events()
```

### 2. State Vector Flow

The 13-DOF state vector flows through the system as follows:

```
State Manager → Physics Engine → RK4 Integrator → State Manager
     ↓                                                   ↑
Controllers ← Sensors ← Noise Models                     │
     ↓                                                   │
Motor Commands → Propellers → Forces/Moments ────────────┘
```

**State Vector Components**:

- **Position** `[px, py, pz]`: Used by environment, controllers, sensors
- **Quaternion** `[qw, qx, qy, qz]`: Used by physics, controllers, visualization
- **Velocity** `[vx, vy, vz]`: Used by physics, propellers, controllers
- **Angular Velocity** `[ωx, ωy, ωz]`: Used by physics, controllers, sensors

### 3. Control System Data Flow

The three-layer control system has specific data dependencies:

```
Reference Trajectory
        ↓
┌─────────────────┐
│ Position Control│ ← Current Position (from State Manager)
└─────────────────┘
        ↓ (Attitude Reference)
┌─────────────────┐
│ Attitude Control│ ← Current Attitude (from State Manager)
└─────────────────┘
        ↓ (Rate Reference)
┌─────────────────┐
│   Rate Control  │ ← Current Angular Rates (from State Manager)
└─────────────────┘
        ↓ (Control Moments)
┌─────────────────┐
│  Motor Mixing   │ ← Vehicle Configuration
└─────────────────┘
        ↓ (Motor Commands)
┌─────────────────┐
│   Propellers    │ ← Air Density (from Environment)
└─────────────────┘
        ↓ (Forces/Moments)
    Physics Engine
```

## Component Dependencies

### Core Components

#### Simulator

**Dependencies**:

- State Manager (for state access)
- Physics Engine (for dynamics)
- Event System (for parameter changes)

**Provides**:

- Main simulation loop
- Time management
- Component orchestration

**Interaction Pattern**:

```python
class Simulator:
    def __init__(self):
        self.state_manager = StateManager()
        self.event_system = EventSystem()
        self.components = {}

    def register_physics_engine(self, physics):
        self.physics = physics
        physics.set_state_manager(self.state_manager)

    def register_controller(self, controller):
        self.controller = controller
        controller.set_event_system(self.event_system)
```

#### State Manager

**Dependencies**:

- None (core component)

**Provides**:

- State storage and validation
- State history
- Coordinate transformations

**Used By**:

- Simulator (main loop)
- Physics Engine (state derivatives)
- Controllers (current state)
- Sensors (measurement generation)

#### Event System

**Dependencies**:

- None (core component)

**Provides**:

- Parameter change notifications
- Emergency stop signals
- Custom event handling

**Used By**:

- Simulator (event processing)
- Controllers (parameter updates)
- User interfaces (command injection)

### Physics Components

#### Rigid Body

**Dependencies**:

- State Manager (current state)
- Environment (external forces)
- Propellers (thrust/torque)

**Provides**:

- State derivatives for integration
- Force/moment accumulation
- Quaternion dynamics

**Interaction Pattern**:

```python
def compute_derivatives(self, state, t):
    # Get forces from all sources
    gravity = self.environment.get_gravity()
    wind_force = self.environment.get_wind_force(state.position, t)
    propeller_forces = self.propellers.get_total_force()

    # Compute dynamics
    return self._compute_6dof_dynamics(state, total_forces)
```

#### Environment

**Dependencies**:

- Configuration (wind models, atmosphere)

**Provides**:

- Gravity vector
- Wind velocity fields
- Atmospheric properties

**Used By**:

- Rigid Body (external forces)
- Propellers (air density)
- Sensors (environmental effects)

#### Propellers

**Dependencies**:

- Environment (air density)
- Controllers (motor commands)

**Provides**:

- Thrust forces
- Reaction torques
- Power consumption

**Interaction Pattern**:

```python
def compute_forces(self, motor_commands, airspeed, air_density):
    total_force = np.zeros(3)
    total_moment = np.zeros(3)

    for i, cmd in enumerate(motor_commands):
        # Individual propeller calculation
        thrust = self.propellers[i].compute_thrust(cmd, airspeed, air_density)
        torque = self.propellers[i].compute_torque(cmd, airspeed, air_density)

        # Accumulate forces and moments
        total_force += thrust
        total_moment += torque + np.cross(self.positions[i], thrust)

    return total_force, total_moment
```

### Control Components

#### Base Controller

**Dependencies**:

- State Manager (current state)
- Event System (parameter changes)

**Provides**:

- Control interface
- Parameter management
- Debug information

**Extended By**:

- PID Controller
- LQR Controller
- ML Controllers

#### PID Controller

**Dependencies**:

- Base Controller (interface)
- Configuration (gains, limits)

**Provides**:

- Three-layer control
- Anti-windup protection
- Tuning parameters

**Interaction Pattern**:

```python
def compute_control(self, state, reference, dt):
    # Position control loop
    if reference.position is not None:
        pos_error = reference.position - state.position
        attitude_ref = self.position_controller.update(pos_error, dt)
    else:
        attitude_ref = reference.attitude

    # Attitude control loop
    att_error = self._quaternion_error(attitude_ref, state.quaternion)
    rate_ref = self.attitude_controller.update(att_error, dt)

    # Rate control loop
    rate_error = rate_ref - state.angular_velocity
    moments = self.rate_controller.update(rate_error, dt)

    return ControllerOutput(moments=moments)
```

## Inter-Component Communication

### 1. Direct Method Calls

Most components communicate through direct method calls:

```python
# Simulator calling physics engine
derivatives = self.physics.compute_derivatives(state, time)

# Controller calling propellers
forces = self.propellers.compute_forces(motor_commands)
```

### 2. Event-Driven Communication

Parameter changes and emergency signals use events:

```python
# Controller subscribing to parameter changes
event_system.subscribe(EventType.PARAMETER_CHANGE, self.update_parameters)

# User interface triggering emergency stop
event_system.publish(Event(EventType.EMERGENCY_STOP))
```

### 3. Configuration-Based Coupling

Components are loosely coupled through configuration:

```python
# Configuration defines relationships
config = {
    'propellers': {
        'count': 4,
        'positions': [[0.2, 0.2, 0], [-0.2, 0.2, 0], ...]
    },
    'control': {
        'motor_mixing_matrix': [[1, 1, 1, 1], [1, -1, -1, 1], ...]
    }
}
```

## Timing and Synchronization

### 1. Synchronous Execution

All components run synchronously in the main simulation loop:

```python
def simulation_step(self):
    # All operations happen in sequence
    self.update_sensors()      # ~1ms
    self.update_control()      # ~0.5ms
    self.update_physics()      # ~1ms
    self.integrate_state()     # ~0.5ms
    # Total: ~3ms per step
```

### 2. Different Update Rates

Components can have different effective update rates:

```python
class Sensor:
    def update(self, state, time, dt):
        # Only update at sensor rate
        if time - self.last_update >= 1.0 / self.update_rate:
            return self.generate_measurement(state)
        return None
```

### 3. Time Synchronization

All components use the same simulation time:

```python
def step(self):
    current_time = self.time
    dt = self.dt

    # All components receive same time
    self.controller.update(state, reference, current_time, dt)
    self.sensors.update(state, current_time, dt)
    self.physics.update(state, current_time, dt)

    self.time += dt
```

## Error Handling and Validation

### 1. State Validation Chain

State validation occurs at multiple levels:

```python
# State Manager validates physical bounds
def set_state(self, new_state):
    if not self.validator.is_valid(new_state):
        new_state = self.validator.sanitize(new_state)

    self.current_state = new_state

# Physics Engine validates dynamics
def compute_derivatives(self, state):
    if np.any(np.isnan(state)):
        raise ValueError("NaN detected in state")

    derivatives = self._compute_dynamics(state)

    if np.any(np.isinf(derivatives)):
        raise ValueError("Infinite derivatives")

    return derivatives
```

### 2. Component Health Monitoring

Components can report their health status:

```python
class Component:
    def get_health_status(self):
        return {
            'status': 'healthy',
            'last_update': self.last_update_time,
            'error_count': self.error_count,
            'warnings': self.warnings
        }
```

### 3. Graceful Degradation

System continues operating with reduced functionality:

```python
def update_sensors(self):
    for sensor_name, sensor in self.sensors.items():
        try:
            data = sensor.update(self.state, self.time, self.dt)
            self.sensor_data[sensor_name] = data
        except Exception as e:
            logger.warning(f"Sensor {sensor_name} failed: {e}")
            self.sensor_data[sensor_name] = None  # Continue without this sensor
```

## Performance Considerations

### 1. Computational Bottlenecks

- **Physics Integration**: Most computationally expensive
- **Propeller Calculations**: Scales with number of propellers
- **Control Updates**: Generally lightweight
- **State Validation**: Minimal overhead

### 2. Memory Usage Patterns

- **State History**: Grows linearly with simulation time
- **Sensor Buffers**: Fixed size circular buffers
- **Configuration Data**: Loaded once at startup
- **Temporary Arrays**: Reused across iterations

### 3. Optimization Strategies

- **Vectorization**: Use NumPy for array operations
- **Caching**: Cache expensive calculations
- **Lazy Evaluation**: Only compute when needed
- **Memory Pooling**: Reuse objects to reduce allocation

## Extension Points

### 1. Adding New Components

New components integrate through standard interfaces:

```python
class NewComponent:
    def __init__(self, config):
        self.config = config

    def update(self, state, time, dt):
        # Component-specific logic
        pass

    def get_output(self):
        # Return component output
        pass

# Registration
simulator.register_component('new_component', NewComponent(config))
```

### 2. Custom Communication Patterns

Components can implement custom communication:

```python
class CustomComponent:
    def set_data_source(self, source):
        self.data_source = source

    def get_data_sink(self):
        return self.data_sink

    def update(self, state, time, dt):
        # Get data from source
        input_data = self.data_source.get_data()

        # Process and provide to sink
        output_data = self.process(input_data)
        self.data_sink.set_data(output_data)
```

### 3. Plugin Architecture

Components can be loaded dynamically:

```python
def load_plugin(plugin_path, config):
    module = importlib.import_module(plugin_path)
    component_class = getattr(module, 'Component')
    return component_class(config)

# Usage
custom_controller = load_plugin('plugins.advanced_controller', config)
simulator.register_controller(custom_controller)
```

This system relationship analysis provides the foundation for understanding how the drone simulation components work together to create a cohesive, extensible simulation environment.
