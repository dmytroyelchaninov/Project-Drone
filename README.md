# Drone Project

A comprehensive drone simulation and control system with modular architecture supporting both real hardware and simulation modes.

## Architecture Overview

The system follows a modular design based on the specification in `src/prompts.md`:

```
src/
├── cfg/                    # Configuration management (YAML-based)
├── input/                  # Input handling system
│   ├── devices/           # Input devices (keyboard, joystick)
│   ├── sensors/           # All sensor implementations
│   ├── hub.py            # Central data hub (SINGLETON)
│   └── poller.py         # Polling system manager
├── physics/               # Physics simulation modules
├── drone/                 # Main drone control logic
└── prompts.md            # Complete system specification
```

## Key Components

### Hub (Singleton)

- Central data storage and coordination
- Manages all input/output data
- Stores device states and system configuration
- Thread-safe with asynchronous real-time updates

### Poller

- Manages polling of all devices and sensors
- Runs in separate thread with configurable frequency
- Includes error handling and recovery mechanisms
- Provides system testing capabilities

### Drone Control

- Main control logic transforming input to voltage output
- Supports multiple modes: manual, hybrid, AI
- Implements safety monitoring and emergency procedures
- Integrates with physics simulation

### Input System

- **Devices**: Keyboard, joystick control with voltage output
- **Sensors**: GPS, barometer, gyroscope, compass, camera, etc.
- **VoltageController**: Converts voltages to thrust using propeller physics

### Physics Simulation

- Complete 3D rigid body dynamics with quaternions
- Propeller aerodynamics and thrust calculation
- Environmental effects (wind, ground effect, atmosphere)
- Real-time physics integration

## Quick Start

### Run the Complete System

```bash
cd scripts
python run_system.py
```

### Test Individual Components

```bash
cd scripts
python test_components.py
```

### Configuration

Edit configuration files in `src/cfg/`:

- `settings.yaml` - Main settings
- `user.yaml` - User-specific overrides
- `emulation.yaml` - Simulation-specific settings

## Control Modes

### Manual Mode

- Direct voltage control from input device (keyboard/joystick)
- Real-time response to user input
- Safety limits and emergency override

### Hybrid Mode

- Manual control with AI assistance
- AI provides stability and safety corrections
- Blended control for enhanced performance

### AI Mode

- Full autonomous control
- Task-based operation (takeoff, land, follow, return to base)
- Sensor-based navigation and control

## Control States

### Go States

- **off**: All engines off
- **idle**: Minimal power, engines spinning
- **float**: Hover in place
- **operate**: Active control mode

### Tasks (AI/Hybrid)

- **take_off**: Autonomous takeoff sequence
- **land**: Controlled landing
- **follow**: Follow target (placeholder)
- **back_to_base**: Return to home position
- **projectile**: Ballistic trajectory mode

## Input Devices

### Keyboard Controls

- **WASD**: Roll/Pitch control
- **QE**: Yaw control
- **RF**: Throttle up/down
- **TAB**: Cycle control modes
- **SPACE**: Emergency stop

### Device Interface

All devices inherit from `BaseDevice` with common polling interface:

```python
class BaseDevice:
    def poll(self) -> Dict[str, Any]
    def start(self)
    def stop(self)
```

## Sensors

All sensors inherit from `BaseSensor` with real/fake modes:

- **GPS**: Position, velocity, heading
- **Barometer**: Altitude, vertical speed, atmospheric data
- **Gyroscope**: Angular velocity with calibration
- **Compass**: Magnetic heading and field strength
- **Camera**: Visual data, object detection, optical flow
- **Anemometer**: Wind speed and direction
- **Temperature/Humidity**: Environmental monitoring
- **LiDAR**: Distance measurement and obstacle detection

## Physics System

### QuadcopterPhysics

- 6-DOF rigid body dynamics
- Engine thrust to force/moment conversion
- Integration with environmental effects
- Real-time state updates

### Environment (Singleton)

- Atmospheric properties vs altitude
- Wind modeling with turbulence
- Ground effects and boundaries
- Aerodynamic force calculation

### Propeller Model

- Voltage to RPM to thrust conversion
- Realistic aerodynamic coefficients
- Performance analysis and hover calculations

## Safety Features

### Emergency Systems

- Automatic emergency stops on danger detection
- Watchdog timeouts for input devices
- Altitude and angular rate limiting
- Recovery procedures with back-to-base capability

### Error Handling

- Poller error recovery with multiple attempts
- Graceful degradation on sensor failures
- Thread-safe data access with locks
- Comprehensive logging and diagnostics

## Data Flow

```
Input Devices → Hub Input ←  Sensors
     ↓                          ↑
   Poller ←→ Hub (Central Data Store)
     ↓                          ↓
Drone Control → Hub Output → Physics Simulation
     ↓                          ↓
Engine Voltages → Real Hardware / Simulation
```

## Configuration System

The system uses YAML-based configuration with inheritance:

1. `default.yaml` - Safe defaults
2. `settings.yaml` - Main configuration
3. `user.yaml` - User overrides
4. `emulation.yaml` - Simulation settings

Settings are accessible through the singleton `Settings` class.

## Development

### Adding New Sensors

1. Inherit from `BaseSensor`
2. Implement `_poll_real()` and `_poll_fake()` methods
3. Add to sensor factory in Hub
4. Update configuration files

### Adding New Devices

1. Inherit from `BaseDevice`
2. Implement polling and voltage output
3. Add to device factory in Hub
4. Update configuration

### Extending AI Control

The drone control system supports easy extension of AI behaviors:

- Add new tasks to `_compute_ai_control()`
- Implement task-specific control logic
- Integrate sensor feedback for autonomous operation

## Logging

The system provides comprehensive logging:

- Console output for real-time monitoring
- File logging for detailed analysis
- Configurable log levels
- Component-specific loggers

## Hardware Integration

The system is designed for easy hardware integration:

- Device abstraction layer for input hardware
- Sensor abstraction for real hardware interfaces
- Physics simulation can be replaced with hardware interfaces
- Configuration-driven real/simulation mode switching

## Dependencies

- NumPy for numerical computation
- PyYAML for configuration management
- Threading for concurrent operation
- Logging for system monitoring

## License

[Add your license information here]
