# Voltage-Based Control Architecture

## Overview

The new voltage-based control system provides a modular, physics-accurate approach to drone manual control. The system bridges the gap between user inputs (keyboard/joystick) and the physics simulation by implementing realistic voltage-to-thrust conversion.

## Architecture Components

### 1. BaseDevice Class

The foundation for all input devices and sensors:

- **Polling System**: Threaded polling with configurable rates
- **Connection Management**: Automatic reconnection and error handling
- **Data Validation**: Built-in data validation and error detection
- **Callback System**: Event-driven architecture for real-time responses

```python
class BaseDevice(ABC):
    def start() -> bool          # Start device polling
    def stop()                   # Stop device and cleanup
    def poll() -> Dict[str, Any] # Single poll operation
    def get_latest_data()        # Thread-safe data access
```

### 2. VoltageController

Converts voltage commands to physics-compatible thrust values:

**Key Features:**

- Individual engine voltage control (0-12V)
- Voltage ramping for realistic response (24V/s default)
- Safety limits and emergency stop
- Real-time voltage → RPM → thrust conversion
- Per-engine calibration support

**Physics Pipeline:**

```
Voltage Input → RPM Calculation → Propeller Physics → Thrust Output
     (V)             (RPM)           (Aerodynamics)        (N)
```

**Conversion Formulas:**

```python
# Voltage to RPM (linear relationship)
RPM = voltage * max_rpm_per_volt  # Default: 800 RPM/V

# RPM to Thrust (propeller physics)
if using_propeller_physics:
    thrust = propeller.calculate_thrust(rpm)  # Momentum theory
else:
    thrust = k * rpm²  # Simplified quadratic (k=1e-8)
```

### 3. KeyboardDevice

Handles keyboard input and converts to voltage commands:

**Control Mapping:**

- `SPACE/SHIFT`: Throttle up/down (modifies all engines)
- `←/→`: Roll left/right (differential thrust)
- `↑/↓`: Pitch forward/backward (front/back differential)
- `A/D`: Yaw left/right (diagonal pairs)
- `ESC`: Emergency stop

**Control Mixing (+ Configuration):**

```
Engine Layout:     Voltage Mixing:
     0 (Front)     V₀ = base + throttle + pitch - yaw
   3   1           V₁ = base + throttle + roll + yaw
     2 (Back)      V₂ = base + throttle - pitch - yaw
                   V₃ = base + throttle - roll + yaw
```

### 4. InputHub

Central coordinator for all devices and sensors:

- **Device Management**: Start/stop all devices
- **Data Aggregation**: Collect data from all sources
- **Safety Monitoring**: Watchdog timers and emergency handling
- **Callback System**: Real-time event notifications

## Configuration System

### Device Configuration

```python
@dataclass
class VoltageControllerConfig(DeviceConfig):
    num_engines: int = 4
    min_voltage: float = 0.0
    max_voltage: float = 12.0
    hover_voltage: float = 6.0
    max_rpm_per_volt: float = 800.0
    voltage_ramp_rate: float = 24.0  # V/s
    emergency_cutoff_enabled: bool = True
```

### Hub Configuration

```python
@dataclass
class HubConfig:
    name: str = "drone_input_hub"
    update_rate: float = 100.0  # Hz
    keyboard_enabled: bool = True
    sensors_enabled: bool = True
    emergency_stop_enabled: bool = True
    watchdog_timeout: float = 1.0  # seconds
```

## Physics Integration

### Current Physics System

Your existing physics already supports individual engine thrust:

```python
# In QuadcopterPhysics
def set_engine_thrusts(self, thrusts: np.ndarray):
    self.state.engine_thrusts = np.clip(thrusts, min_thrust, max_thrust)

def _calculate_forces_and_moments(self):
    # Convert individual thrusts to total force and moments
    total_thrust = np.sum(self.state.engine_thrusts)
    # Calculate moments from engine positions...
```

### Integration Points

1. **Voltage Controller** → **Physics Engine**:

   ```python
   # Get thrust values from voltage controller
   voltage_data = hub.get_voltage_commands()
   engine_thrusts = voltage_data['thrusts']

   # Apply to physics
   physics.set_engine_thrusts(engine_thrusts)
   ```

2. **Sensor Data** → **AI Controller**:
   ```python
   # AI uses sensor data for autonomous control
   sensor_data = hub.get_sensor_data()
   ai_voltages = ai_controller.compute_voltages(sensor_data)
   hub.set_ai_voltages(ai_voltages)
   ```

## Safety Features

### Emergency Stop System

- **Immediate Response**: ESC key triggers instant voltage cutoff
- **Watchdog Timer**: Automatic stop if devices become unresponsive
- **Voltage Limits**: Hardware-level voltage clamping
- **Graceful Recovery**: Reset to hover state after emergency

### Data Validation

- **Range Checking**: All voltages/RPMs/thrusts within realistic bounds
- **Consistency Checks**: Cross-validate related measurements
- **Error Handling**: Graceful degradation on sensor failures

## Usage Examples

### Basic Setup

```python
from src.input import InputHub, HubConfig, KeyboardDeviceConfig

# Configure keyboard
keyboard_config = KeyboardDeviceConfig(
    name="drone_keyboard",
    poll_rate=50.0,
    voltage_sensitivity=1.5
)

# Configure hub
hub_config = HubConfig(
    keyboard_enabled=True,
    keyboard_config=keyboard_config
)

# Create and start hub
hub = InputHub(hub_config)
hub.start()

# Get control data
voltage_data = hub.get_voltage_commands()
engine_thrusts = hub.get_engine_thrusts()
```

### Integration with Physics

```python
# In your main simulation loop
while running:
    # Get control inputs
    voltage_data = hub.get_voltage_commands()

    if voltage_data:
        # Apply to physics
        physics.set_engine_thrusts(voltage_data['thrusts'])

    # Update physics
    physics.update(dt)

    # Get updated state
    state = physics.get_state_dict()
```

## Benefits of This Architecture

### 1. **Realistic Physics**

- Accurate voltage-to-thrust conversion
- Engine response time modeling
- Proper control mixing for quadcopter dynamics

### 2. **Modularity**

- Easy to add new input devices (joystick, gamepad)
- Pluggable sensor system
- Configurable control mappings

### 3. **Safety**

- Multiple layers of safety checks
- Emergency stop functionality
- Graceful error handling

### 4. **AI-Human Compatibility**

- Same voltage interface for AI and human control
- Seamless switching between control modes
- Shared physics model

### 5. **Real-time Performance**

- Threaded polling for responsive control
- Minimal latency from input to physics
- Configurable update rates

## Next Steps

### Immediate Implementation

1. **Test the Demo**: Run `examples/voltage_control_demo.py`
2. **Integrate with Physics**: Connect voltage controller to your existing physics
3. **Validate Control Response**: Test control sensitivity and responsiveness

### Future Enhancements

1. **Joystick Support**: Add GamepadDevice class
2. **Sensor Integration**: Implement IMU, GPS, and other sensors
3. **AI Integration**: Bridge AI control with voltage system
4. **Calibration Tools**: Add motor/propeller calibration utilities
5. **Data Logging**: Integrate with your existing logging system

This architecture provides a solid foundation for both manual and AI control while maintaining realistic physics simulation.
