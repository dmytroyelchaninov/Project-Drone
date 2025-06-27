# Component Reference Guide

## Core Simulation Engine

### `drone_sim/core/simulator.py`

**Purpose**: Main simulation loop with RK4 integration engine

**Key Classes**:

- `Simulator`: Main simulation orchestrator
- `SimulationConfig`: Configuration for time stepping and performance
- `SimulationState`: Enum for simulation states (STOPPED, RUNNING, PAUSED)

**Key Methods**:

```python
def step(self):
    """Execute single RK4 integration step"""
    # 1. Get current state vector
    # 2. Compute k1, k2, k3, k4 derivatives
    # 3. Apply RK4 formula: new_state = old + (k1 + 2k2 + 2k3 + k4) * dt/6
    # 4. Normalize quaternion to prevent drift
    # 5. Update all subsystems
```

**Dependencies**:

- Physics engine for state derivatives
- Control system for actuator commands
- Sensors for measurement updates
- Environment for external forces

**Configuration**:

```python
config = SimulationConfig(
    dt=0.002,           # 2ms time step
    max_steps=1000000,  # Maximum simulation steps
    real_time_factor=1.0 # Real-time multiplier
)
```

---

### `drone_sim/core/state_manager.py`

**Purpose**: Manages 13-DOF drone state with validation and history

**Key Classes**:

- `DroneState`: 13-DOF state representation
- `StateValidator`: Physical bounds and consistency checking
- `StateManager`: State transitions and history management

**State Vector Structure**:

```python
# 13-DOF State Vector
[px, py, pz,           # Position (3 DOF) - inertial frame
 qw, qx, qy, qz,       # Quaternion (4 DOF) - orientation
 vx, vy, vz,           # Velocity (3 DOF) - body frame
 ωx, ωy, ωz]           # Angular velocity (3 DOF) - body frame
```

**Validation Rules**:

- Position magnitude < 1000m
- Velocity magnitude < 50 m/s
- Angular velocity magnitude < 10 rad/s
- Quaternion normalized within 1e-4 tolerance
- No NaN or infinite values

**Key Methods**:

```python
def set_state(self, new_state, timestamp, validate=True):
    """Set state with optional validation and sanitization"""

def get_history(self, n_samples=None):
    """Retrieve state history for analysis"""

def get_euler_angles(self):
    """Convert quaternion to roll/pitch/yaw"""
```

---

### `drone_sim/core/event_system.py`

**Purpose**: Pub-sub event system for real-time parameter changes

**Key Classes**:

- `Event`: Event data structure with type and payload
- `EventType`: Enum for event categories
- `EventSystem`: Publisher-subscriber implementation

**Event Types**:

- `PARAMETER_CHANGE`: Runtime parameter updates
- `EMERGENCY_STOP`: Immediate simulation halt
- `RESET_SIMULATION`: Return to initial conditions
- `CUSTOM`: User-defined events

**Usage Example**:

```python
# Subscribe to events
event_system.subscribe(EventType.PARAMETER_CHANGE, my_callback)

# Publish events
event_system.publish(Event(
    type=EventType.PARAMETER_CHANGE,
    data={'controller.kp': 2.5}
))
```

---

## Physics Engine

### `drone_sim/physics/rigid_body.py`

**Purpose**: 6-DOF rigid body dynamics with quaternion attitude

**Key Classes**:

- `RigidBody`: Main dynamics implementation
- `RigidBodyConfig`: Mass and inertia properties

**Mathematical Foundation**:

```python
# Newton's Second Law (Linear)
F_total = m * a_body + m * ω × v_body

# Euler's Equation (Angular)
M_total = I * α + ω × (I * ω)

# Quaternion Kinematics
q̇ = 0.5 * q ⊗ [0, ωx, ωy, ωz]
```

**Key Methods**:

```python
def compute_derivatives(self, state, t):
    """Compute state derivatives for RK4 integration"""
    # 1. Extract position, quaternion, velocity, angular velocity
    # 2. Convert quaternion to rotation matrix
    # 3. Compute linear acceleration in body frame
    # 4. Compute angular acceleration using Euler's equation
    # 5. Return 13-element derivative vector

def apply_force(self, force, position=None):
    """Apply force at specified position (generates moment if offset)"""

def quaternion_to_rotation_matrix(self, q):
    """Convert quaternion to 3x3 rotation matrix"""
```

---

### `drone_sim/physics/environment.py`

**Purpose**: Environmental effects (gravity, wind, atmosphere)

**Key Classes**:

- `Environment`: Main environment manager
- `EnvironmentConfig`: Environmental parameters
- `WindModel`: Various wind implementations

**Wind Models**:

1. **Constant Wind**: Steady uniform wind field
2. **Turbulent Wind**: Dryden turbulence model
3. **Gust Wind**: Discrete gust encounters
4. **Custom Wind**: User-defined wind fields

**Key Methods**:

```python
def get_wind_velocity(self, position, time):
    """Get wind velocity at given position and time"""

def get_air_density(self, altitude):
    """Compute air density vs altitude (ISA model)"""

def update(self, time, dt):
    """Update time-varying environmental conditions"""
```

---

### `drone_sim/physics/aerodynamics/propeller.py`

**Purpose**: Propeller thrust and torque modeling

**Key Classes**:

- `Propeller`: Individual propeller model
- `PropellerConfig`: Propeller specifications
- `PropellerArray`: Multi-propeller management

**Propeller Models**:

1. **Simple**: Basic thrust = k \* ω² model
2. **Advanced**: Momentum theory with disk loading
3. **Lookup**: Tabulated performance data
4. **Blade Element**: Detailed blade aerodynamics

**Key Parameters**:

```python
PropellerConfig(
    diameter=0.24,        # Propeller diameter (m)
    pitch=0.12,          # Propeller pitch (m)
    blades=2,            # Number of blades
    thrust_coeff=0.1,    # Thrust coefficient
    power_coeff=0.05,    # Power coefficient
    direction=1          # Rotation direction (±1)
)
```

**Key Methods**:

```python
def compute_forces_and_moments(self, rpm, airspeed, air_density):
    """Compute thrust force and reaction torque"""
    # 1. Calculate advance ratio J = V / (n * D)
    # 2. Lookup or compute thrust coefficient CT
    # 3. Lookup or compute power coefficient CP
    # 4. Compute thrust: T = CT * ρ * n² * D⁴
    # 5. Compute torque: Q = CP * ρ * n² * D⁵
```

---

### `drone_sim/physics/aerodynamics/noise_model.py`

**Purpose**: Propeller acoustic noise modeling

**Key Classes**:

- `PropellerNoiseModel`: Main acoustic model
- `ObserverPosition`: Microphone/listener position
- `NoiseSpectrum`: Frequency domain noise data

**Noise Sources**:

1. **Thickness Noise**: Blade volume displacement

   ```python
   # Ffowcs Williams-Hawkings equation
   p_thickness = (ρ * c / 4π) * ∂²/∂t² ∫[V_n]/|r⃗ - r⃗_s| dS
   ```

2. **Loading Noise**: Aerodynamic forces

   ```python
   # Pressure from blade loading
   p_loading = (1 / 4π) * ∂/∂t ∫[F⃗ · r̂]/c|r⃗ - r⃗_s| dS
   ```

3. **Broadband Noise**: Turbulence effects
   ```python
   # Stochastic noise from tip vortices and boundary layer
   ```

**Key Methods**:

```python
def compute_noise_spectrum(self, propeller_state, observer_pos):
    """Compute noise spectrum at observer position"""
    # 1. Calculate geometric relationships
    # 2. Compute thickness noise contribution
    # 3. Compute loading noise contribution
    # 4. Add broadband noise
    # 5. Apply Doppler effects
    # 6. Convert to A-weighted SPL

def get_octave_bands(self, spectrum):
    """Convert spectrum to 1/3 octave bands"""
```

---

## Control Systems

### `drone_sim/control/base_controller.py`

**Purpose**: Abstract base class for all controllers

**Key Classes**:

- `BaseController`: Abstract controller interface
- `ControllerState`: Controller internal state
- `ControllerReference`: Desired trajectory/setpoint
- `ControllerOutput`: Control commands

**Interface Definition**:

```python
class BaseController(ABC):
    @abstractmethod
    def compute_control(self, state, reference, dt):
        """Compute control output given current state and reference"""
        pass

    @abstractmethod
    def reset(self):
        """Reset controller to initial conditions"""
        pass

    def get_debug_info(self):
        """Return controller internals for ML/analysis"""
        return self.debug_data
```

**ML Integration**:

- Exposes controller internals for learning algorithms
- Provides hooks for parameter adaptation
- Supports online controller switching

---

### `drone_sim/control/pid_controller.py`

**Purpose**: Three-layer PID control system

**Control Architecture**:

```
Reference Position → [Position PID] → Attitude Reference
                                              ↓
Current Attitude ← [Attitude PID] ← Attitude Reference
        ↓
Angular Rate Reference → [Rate PID] → Motor Commands
```

**Key Classes**:

- `PIDController`: Main three-layer controller
- `PIDConfig`: PID gains and limits
- `PIDState`: Controller internal states

**Control Loops**:

1. **Position Loop** (Outer, ~10 Hz):

   ```python
   # Convert position error to attitude reference
   pos_error = reference.position - current.position
   attitude_ref = position_pid.update(pos_error, dt)
   ```

2. **Attitude Loop** (Middle, ~100 Hz):

   ```python
   # Convert attitude error to rate reference
   att_error = quaternion_error(reference.quaternion, current.quaternion)
   rate_ref = attitude_pid.update(att_error, dt)
   ```

3. **Rate Loop** (Inner, ~500 Hz):
   ```python
   # Convert rate error to motor commands
   rate_error = reference.angular_velocity - current.angular_velocity
   motor_cmd = rate_pid.update(rate_error, dt)
   ```

**Anti-Windup Protection**:

- Integral clamping when output saturates
- Back-calculation method for smooth recovery
- Separate limits for each axis

**Key Methods**:

```python
def update(self, state, reference, dt):
    """Execute all three control loops in sequence"""
    # 1. Position control (if position reference provided)
    # 2. Attitude control
    # 3. Rate control
    # 4. Convert to motor commands
    # 5. Apply limits and anti-windup
```

---

## Configuration System

### `drone_sim/configs/drone_presets/quadcopter_default.yaml`

**Purpose**: Standard quadcopter configuration template

**Configuration Structure**:

```yaml
name: "Default Quadcopter"

physics:
  mass: 1.5 # Total mass (kg)
  inertia: [0.02, 0.02, 0.04] # Inertia matrix diagonal (kg⋅m²)

propellers:
  count: 4
  layout: "quad_x" # X-configuration
  motors:
    - id: 0
      position: [0.2, 0.2, 0.0] # Front-right
      direction: -1 # Counter-clockwise
    - id: 1
      position: [-0.2, 0.2, 0.0] # Front-left
      direction: 1 # Clockwise
    # ... etc

control:
  position_pid:
    kp: [2.0, 2.0, 4.0] # Position gains [x, y, z]
    ki: [0.1, 0.1, 0.2] # Integral gains
    kd: [1.0, 1.0, 2.0] # Derivative gains
  attitude_pid:
    kp: [8.0, 8.0, 4.0] # Attitude gains [roll, pitch, yaw]
    ki: [0.1, 0.1, 0.1]
    kd: [0.3, 0.3, 0.1]
  rate_pid:
    kp: [0.2, 0.2, 0.1] # Rate gains
    ki: [0.05, 0.05, 0.02]
    kd: [0.01, 0.01, 0.005]

environment:
  gravity: 9.81
  air_density: 1.225
  wind_model: "none"
```

---

## Package Structure

### `drone_sim/__init__.py`

**Purpose**: Main package interface and exports

**Exported Components**:

```python
# Core simulation
from .core.simulator import Simulator, SimulationConfig
from .core.state_manager import StateManager, DroneState
from .core.event_system import EventSystem, Event, EventType

# Physics
from .physics.rigid_body import RigidBody, RigidBodyConfig
from .physics.environment import Environment, EnvironmentConfig
from .physics.aerodynamics.propeller import Propeller, PropellerArray
from .physics.aerodynamics.noise_model import PropellerNoiseModel

# Control
from .control.base_controller import BaseController
from .control.pid_controller import PIDController
```

**Usage Pattern**:

```python
from drone_sim import (
    Simulator, DroneState, PIDController,
    RigidBody, Environment, PropellerArray
)

# Create simulation
sim = Simulator()
physics = RigidBody(config)
controller = PIDController(config)

# Register components
sim.register_physics_engine(physics)
sim.register_control_system(controller)

# Run simulation
sim.run(duration=10.0)
```

---

## Examples and Applications

### `examples/basic_simulation.py`

**Purpose**: Complete working example demonstrating all systems

**Simulation Flow**:

1. **Setup Phase**:

   ```python
   # Load configuration
   config = yaml.load(open('quadcopter_default.yaml'))

   # Create components
   physics = RigidBody(config.physics)
   controller = PIDController(config.control)
   propellers = PropellerArray(config.propellers)
   ```

2. **Integration Phase**:

   ```python
   # Register components
   sim.register_physics_engine(physics)
   sim.register_control_system(controller)
   sim.register_propellers(propellers)
   ```

3. **Execution Phase**:

   ```python
   # Set initial conditions
   state.position = [0, 0, -1]  # Start 1m below ground

   # Define mission
   reference.position = [0, 0, 10]  # Climb to 10m

   # Run simulation
   sim.run(duration=20.0)
   ```

4. **Analysis Phase**:

   ```python
   # Extract data
   states, times = state_manager.get_history()

   # Generate plots
   plot_trajectory(states, times)
   plot_control_performance(controller.get_history())
   ```

---

## Development Patterns

### Component Registration Pattern

```python
class Simulator:
    def register_component(self, name, component):
        self.components[name] = component
        if hasattr(component, 'set_simulator'):
            component.set_simulator(self)
```

### Configuration Override Pattern

```python
# Base configuration
base_config = load_yaml('quadcopter_default.yaml')

# Override specific parameters
override_config = {
    'control.position_pid.kp': [3.0, 3.0, 5.0],
    'physics.mass': 2.0
}

# Merge configurations
final_config = merge_configs(base_config, override_config)
```

### Plugin Architecture Pattern

```python
class PluginManager:
    def load_plugin(self, plugin_path):
        module = importlib.import_module(plugin_path)
        return module.create_component()
```

This component reference provides the foundation for understanding how each piece of the drone simulation works together to create a comprehensive, physics-accurate simulation environment.
