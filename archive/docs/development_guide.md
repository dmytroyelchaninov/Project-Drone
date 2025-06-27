# Development Guide

## Getting Started

### Prerequisites

**System Requirements**:

- Python 3.8 or higher
- Git for version control
- C++ compiler (for optional native extensions)

**Recommended IDE Setup**:

- VS Code with Python extension
- PyCharm Professional/Community
- Vim/Neovim with Python LSP

### Installation for Development

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd drone_project
   ```

2. **Create Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in Development Mode**:

   ```bash
   pip install -e ".[dev,visualization,ml]"
   ```

4. **Verify Installation**:
   ```bash
   python examples/basic_simulation.py
   ```

---

## Project Structure Deep Dive

### Core Architecture

```
drone_sim/
├── core/                    # Simulation engine
│   ├── simulator.py         # Main simulation loop
│   ├── state_manager.py     # State handling
│   └── event_system.py      # Event management
├── physics/                 # Physics models
│   ├── rigid_body.py        # 6DOF dynamics
│   ├── environment.py       # Environmental effects
│   └── aerodynamics/        # Aerodynamic models
├── control/                 # Control systems
│   ├── base_controller.py   # Controller interface
│   └── pid_controller.py    # PID implementation
├── sensors/                 # Sensor models
├── analysis/                # Post-processing tools
├── ui/                      # User interfaces
└── utils/                   # Utility functions
```

### Configuration System

**Hierarchical Configuration**:

```yaml
# configs/drone_presets/custom_drone.yaml
name: "Custom Drone"
extends: "quadcopter_default" # Inherit from base config

physics:
  mass: 2.0 # Override specific parameters

control:
  position_pid:
    kp: [3.0, 3.0, 5.0] # Custom PID gains
```

**Loading Configurations**:

```python
from drone_sim.utils.config import load_config

config = load_config('configs/drone_presets/custom_drone.yaml')
```

---

## Adding New Components

### 1. Creating a Custom Controller

**Step 1: Implement Base Interface**

```python
# drone_sim/control/my_controller.py
from .base_controller import BaseController, ControllerOutput
import numpy as np

class MyController(BaseController):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reset()

    def compute_control(self, state, reference, dt):
        """Implement your control logic here"""
        # Extract state components
        position = state.position
        velocity = state.velocity
        attitude = state.quaternion

        # Compute control output
        thrust = self._compute_thrust(position, reference.position)
        moments = self._compute_moments(attitude, reference.quaternion)

        return ControllerOutput(
            thrust=thrust,
            moments=moments,
            motor_commands=self._thrust_to_motors(thrust, moments)
        )

    def reset(self):
        """Reset controller state"""
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)

    def _compute_thrust(self, current_pos, desired_pos):
        # Your thrust computation logic
        pass

    def _compute_moments(self, current_att, desired_att):
        # Your moment computation logic
        pass
```

**Step 2: Add to Package**

```python
# drone_sim/control/__init__.py
from .my_controller import MyController
```

**Step 3: Register in Main Package**

```python
# drone_sim/__init__.py
from .control.my_controller import MyController
```

### 2. Creating a Custom Sensor

**Step 1: Implement Sensor Interface**

```python
# drone_sim/sensors/my_sensor.py
import numpy as np
from typing import Dict, Any

class MySensor:
    def __init__(self, config):
        self.config = config
        self.noise_std = config.get('noise_std', 0.01)
        self.update_rate = config.get('update_rate', 100)  # Hz
        self.last_update = 0

    def update(self, state, time, dt):
        """Update sensor measurements"""
        # Check if it's time to update
        if time - self.last_update < 1.0 / self.update_rate:
            return None

        # Generate measurement with noise
        true_value = self._extract_true_value(state)
        noise = np.random.normal(0, self.noise_std, true_value.shape)
        measurement = true_value + noise

        self.last_update = time
        return {
            'measurement': measurement,
            'timestamp': time,
            'sensor_id': 'my_sensor'
        }

    def _extract_true_value(self, state):
        """Extract the true value this sensor measures"""
        # Example: measuring position
        return state.position
```

**Step 2: Integration Example**

```python
# In your simulation script
from drone_sim.sensors.my_sensor import MySensor

# Create sensor
sensor_config = {'noise_std': 0.05, 'update_rate': 50}
my_sensor = MySensor(sensor_config)

# Register with simulator
simulator.register_sensor('my_sensor', my_sensor)
```

### 3. Creating Custom Physics Models

**Step 1: Extend Environment**

```python
# drone_sim/physics/my_environment.py
from .environment import Environment
import numpy as np

class MyEnvironment(Environment):
    def __init__(self, config):
        super().__init__(config)
        self.custom_field = config.get('custom_field', 0.0)

    def get_external_forces(self, position, velocity, time):
        """Compute custom external forces"""
        base_forces = super().get_external_forces(position, velocity, time)

        # Add custom force field
        custom_force = self._compute_custom_force(position, time)

        return base_forces + custom_force

    def _compute_custom_force(self, position, time):
        """Implement your custom force field"""
        # Example: magnetic field effect
        return np.array([0, 0, self.custom_field * np.sin(time)])
```

---

## Testing Framework

### Unit Testing

**Test Structure**:

```python
# tests/unit/test_my_controller.py
import unittest
import numpy as np
from drone_sim.control.my_controller import MyController
from drone_sim.core.state_manager import DroneState

class TestMyController(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        config = {
            'kp': [1.0, 1.0, 1.0],
            'ki': [0.1, 0.1, 0.1],
            'kd': [0.01, 0.01, 0.01]
        }
        self.controller = MyController(config)

        # Create test state
        self.test_state = DroneState()
        self.test_state.position = np.array([0, 0, -1])
        self.test_state.velocity = np.array([0, 0, 0])

    def test_initialization(self):
        """Test controller initialization"""
        self.assertIsNotNone(self.controller)
        self.assertEqual(len(self.controller.integral_error), 3)

    def test_control_output(self):
        """Test control computation"""
        reference = DroneState()
        reference.position = np.array([1, 0, -1])

        output = self.controller.compute_control(
            self.test_state, reference, dt=0.01
        )

        self.assertIsNotNone(output.thrust)
        self.assertIsNotNone(output.moments)
        self.assertEqual(len(output.motor_commands), 4)

    def test_reset(self):
        """Test controller reset"""
        # Modify internal state
        self.controller.integral_error = np.array([1, 2, 3])

        # Reset
        self.controller.reset()

        # Check reset worked
        np.testing.assert_array_equal(
            self.controller.integral_error,
            np.zeros(3)
        )

if __name__ == '__main__':
    unittest.main()
```

**Running Tests**:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_my_controller.py

# Run with coverage
python -m pytest --cov=drone_sim tests/
```

### Integration Testing

**Example Integration Test**:

```python
# tests/integration/test_simulation_integration.py
import unittest
from drone_sim import Simulator, RigidBody, PIDController
from drone_sim.utils.config import load_config

class TestSimulationIntegration(unittest.TestCase):
    def setUp(self):
        """Set up full simulation"""
        config = load_config('configs/drone_presets/quadcopter_default.yaml')

        self.simulator = Simulator()
        self.physics = RigidBody(config['physics'])
        self.controller = PIDController(config['control'])

        # Register components
        self.simulator.register_physics_engine(self.physics)
        self.simulator.register_control_system(self.controller)

    def test_simulation_runs(self):
        """Test that simulation runs without errors"""
        try:
            self.simulator.run(duration=1.0)  # Short test run
            self.assertTrue(True)  # If we get here, no exceptions
        except Exception as e:
            self.fail(f"Simulation failed: {e}")

    def test_state_evolution(self):
        """Test that state evolves correctly"""
        initial_state = self.simulator.get_state()
        self.simulator.step()
        final_state = self.simulator.get_state()

        # State should have changed
        self.assertFalse(np.allclose(
            initial_state.position,
            final_state.position
        ))
```

---

## Performance Optimization

### Profiling

**Basic Profiling**:

```python
# profile_simulation.py
import cProfile
import pstats
from drone_sim import Simulator

def run_simulation():
    sim = Simulator()
    # ... setup simulation
    sim.run(duration=10.0)

if __name__ == '__main__':
    # Profile the simulation
    cProfile.run('run_simulation()', 'simulation_profile.prof')

    # Analyze results
    stats = pstats.Stats('simulation_profile.prof')
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

**Line Profiling**:

```bash
# Install line_profiler
pip install line_profiler

# Add @profile decorator to functions
# Run with:
kernprof -l -v profile_simulation.py
```

### Memory Optimization

**Memory Profiling**:

```python
# memory_profile.py
from memory_profiler import profile
from drone_sim import Simulator

@profile
def run_simulation():
    sim = Simulator()
    # ... setup and run
    sim.run(duration=10.0)

if __name__ == '__main__':
    run_simulation()
```

**Optimization Strategies**:

1. **Use NumPy arrays** instead of Python lists
2. **Pre-allocate arrays** for known sizes
3. **Circular buffers** for history storage
4. **Object pooling** for frequently created objects

---

## Code Style and Standards

### Formatting

**Use Black for consistent formatting**:

```bash
pip install black
black drone_sim/
```

**Configuration** (pyproject.toml):

```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
```

### Linting

**Use flake8 for style checking**:

```bash
pip install flake8
flake8 drone_sim/
```

**Configuration** (.flake8):

```ini
[flake8]
max-line-length = 88
ignore = E203, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist
```

### Type Hints

**Use type hints consistently**:

```python
from typing import List, Dict, Optional, Union
import numpy as np

def compute_forces(
    positions: np.ndarray,
    velocities: np.ndarray,
    time: float
) -> np.ndarray:
    """Compute forces given positions and velocities.

    Args:
        positions: Array of shape (n, 3) with positions
        velocities: Array of shape (n, 3) with velocities
        time: Current simulation time

    Returns:
        Array of shape (n, 3) with computed forces
    """
    # Implementation here
    pass
```

### Documentation

**Use Google-style docstrings**:

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Longer description if needed. Can span multiple lines
    and include mathematical equations using LaTeX:

    .. math::
        F = ma

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: If param1 is negative
        TypeError: If param2 is not a string

    Example:
        >>> result = my_function(5, "hello")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")

    return len(param2) > param1
```

---

## Debugging Strategies

### Logging

**Set up comprehensive logging**:

```python
# drone_sim/utils/logging.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# In your modules
import logging
logger = logging.getLogger(__name__)

class MyClass:
    def method(self):
        logger.debug("Entering method")
        logger.info("Processing data")
        logger.warning("Potential issue detected")
        logger.error("Error occurred")
```

### Debugging Tools

**Use debugger effectively**:

```python
# Set breakpoints
import pdb; pdb.set_trace()

# Or use ipdb for better interface
import ipdb; ipdb.set_trace()

# In VS Code, use built-in debugger with launch.json:
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Simulation",
            "type": "python",
            "request": "launch",
            "program": "examples/basic_simulation.py",
            "console": "integratedTerminal"
        }
    ]
}
```

### Visualization for Debugging

**Plot intermediate results**:

```python
import matplotlib.pyplot as plt

def debug_plot_trajectory(states, times):
    """Plot trajectory for debugging"""
    positions = np.array([s.position for s in states])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 3D trajectory
    axes[0, 0].plot(positions[:, 0], positions[:, 1])
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Horizontal Trajectory')

    # Altitude vs time
    axes[0, 1].plot(times, positions[:, 2])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Altitude (m)')
    axes[0, 1].set_title('Altitude Profile')

    plt.tight_layout()
    plt.savefig('debug_trajectory.png')
    plt.show()
```

---

## Continuous Integration

### GitHub Actions

**Example workflow** (.github/workflows/ci.yml):

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with flake8
        run: |
          flake8 drone_sim/

      - name: Test with pytest
        run: |
          pytest tests/ --cov=drone_sim --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v1
```

---

## Contributing Guidelines

### Git Workflow

**Branch Naming**:

- `feature/new-controller` - New features
- `bugfix/quaternion-drift` - Bug fixes
- `hotfix/critical-issue` - Critical fixes
- `docs/api-reference` - Documentation updates

**Commit Messages**:

```
feat: add LQR controller implementation

- Implement Linear Quadratic Regulator
- Add configuration options for Q and R matrices
- Include unit tests and documentation
- Closes #123

fix: resolve quaternion normalization drift

The quaternion normalization was causing accumulation of
numerical errors. Added automatic normalization after each
integration step.

Fixes #456
```

### Pull Request Process

1. **Create Feature Branch**:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Changes and Test**:

   ```bash
   # Make your changes
   git add .
   git commit -m "feat: implement new feature"

   # Run tests
   pytest tests/
   flake8 drone_sim/
   ```

3. **Push and Create PR**:

   ```bash
   git push origin feature/my-new-feature
   # Create PR through GitHub interface
   ```

4. **PR Template**:

   ```markdown
   ## Description

   Brief description of changes

   ## Type of Change

   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing

   - [ ] Tests pass locally
   - [ ] Added tests for new functionality
   - [ ] Manual testing completed

   ## Checklist

   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No breaking changes (or marked as such)
   ```

---

## Release Process

### Versioning

**Semantic Versioning** (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. **Update Version**:

   ```python
   # setup.py
   version="1.2.3"
   ```

2. **Update Changelog**:

   ```markdown
   ## [1.2.3] - 2024-01-15

   ### Added

   - New LQR controller
   - Improved noise modeling

   ### Fixed

   - Quaternion drift issue
   - Memory leak in state manager
   ```

3. **Create Release**:

   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

4. **Build and Upload**:
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

This development guide provides the foundation for contributing to and extending the drone simulation project while maintaining code quality and consistency.
