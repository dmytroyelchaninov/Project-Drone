# Directory Reorganization Guide

## Overview

The `examples/` directory has been reorganized to better separate different types of files and improve code maintainability. This guide explains the new structure and how to use it.

## New Directory Structure

### 🗂️ **examples/** (Executable Examples)

Contains ready-to-run demonstration scripts that showcase the drone simulation capabilities:

- `basic_simulation.py` - Simple drone simulation example
- `console_sim.py` - Console-based simulation interface
- `console_realtime_sim.py` - Real-time console simulation
- `demo_realtime_system.py` - Real-time system demonstration
- `realtime_simulation.py` - Advanced real-time simulation
- `simple_realtime_test.py` - Simple real-time testing

**Usage:** Run these files directly to see the simulation in action:

```bash
python examples/basic_simulation.py
python examples/realtime_simulation.py
```

### 🧪 **drone_sim/tests/** (Test Infrastructure)

#### **drone_sim/tests/system/** (System Tests)

Integration tests, validation tests, and test orchestration:

- `integrated_test_runner.py` - Main test runner with background validation
- `run_all_tests_with_logging.py` - Comprehensive test suite runner
- `test_*.py` - Individual system tests (keyboard, controls, GUI, etc.)
- `verify_*.py` - Verification and validation scripts
- `validate_*.py` - Physics validation tests

**Usage:** Run system-wide integration tests:

```bash
python drone_sim/tests/system/integrated_test_runner.py
python drone_sim/tests/system/run_all_tests_with_logging.py
```

#### **drone_sim/tests/benchmarks/** (Performance Benchmarks)

Comprehensive test suites for performance evaluation:

- `advanced_tests.py` - Advanced feature testing
- `performance_benchmarks.py` - Performance measurement suite
- `maneuver_tests.py` - Flight maneuver testing
- `environmental_tests.py` - Environmental condition testing

**Usage:** Run specific benchmark suites:

```bash
python drone_sim/tests/benchmarks/performance_benchmarks.py
python drone_sim/tests/benchmarks/maneuver_tests.py
```

#### **drone_sim/tests/unit/** (Unit Tests)

Individual component tests (existing structure preserved).

#### **drone_sim/tests/integration/** (Integration Tests)

Component integration tests (existing structure preserved).

### 🔧 **examples/helpers/** (Helper Utilities)

Reusable utilities and analysis tools:

- `physics_constraint_enforcer.py` - Physics constraint validation
- `comprehensive_issue_analysis.py` - Issue analysis utilities
- `physics_analysis.py` - Physics analysis tools
- `background_validation_demo.py` - Background validation demonstration

**Usage:** Import as utilities in other scripts:

```python
from examples.helpers import PhysicsConstraintEnforcer
from examples.helpers import ComprehensiveIssueAnalyzer
```

## Migration Guide

### Import Path Changes

If you have existing code that imports from the old structure, update the imports:

**Old:**

```python
from examples.advanced_tests import AdvancedTestSuite
from examples.performance_benchmarks import PerformanceBenchmark
from examples.physics_constraint_enforcer import PhysicsConstraintEnforcer
```

**New:**

```python
from drone_sim.tests.benchmarks.advanced_tests import AdvancedTestSuite
from drone_sim.tests.benchmarks.performance_benchmarks import PerformanceBenchmark
from examples.helpers.physics_constraint_enforcer import PhysicsConstraintEnforcer
```

### Running Tests

**System Tests:**

```bash
# Run integrated test suite with background validation
python drone_sim/tests/system/integrated_test_runner.py

# Run all tests with comprehensive logging
python drone_sim/tests/system/run_all_tests_with_logging.py

# Run individual system tests
python drone_sim/tests/system/test_enhanced_controls.py
```

**Benchmark Tests:**

```bash
# Run performance benchmarks
python drone_sim/tests/benchmarks/performance_benchmarks.py

# Run maneuver tests
python drone_sim/tests/benchmarks/maneuver_tests.py

# Run environmental tests
python drone_sim/tests/benchmarks/environmental_tests.py
```

**Examples:**

```bash
# Run basic simulation
python examples/basic_simulation.py

# Run real-time simulation
python examples/realtime_simulation.py
```

## Benefits of New Structure

### 🎯 **Clear Separation of Concerns**

- **Examples**: Demonstration and learning
- **Tests**: Validation and quality assurance
- **Helpers**: Reusable utilities
- **Benchmarks**: Performance evaluation

### 📊 **Better Organization**

- Tests are properly categorized (unit, integration, system, benchmarks)
- Helper utilities are easily discoverable and reusable
- Examples focus on demonstration rather than testing

### 🔧 **Improved Maintainability**

- Related functionality is grouped together
- Import paths are more logical and consistent
- Easier to find and modify specific components

### 🚀 **Enhanced Workflow**

- Clearer development workflow (examples → tests → benchmarks)
- Better separation between learning and validation
- Easier to run specific types of tests

## File Categories

### 📋 **Test Files** (moved to `drone_sim/tests/system/`)

- `test_*.py` - Individual feature tests
- `verify_*.py` - Verification scripts
- `validate_*.py` - Validation scripts

### 🏃 **Test Runners** (moved to `drone_sim/tests/system/`)

- `integrated_test_runner.py` - Main test orchestrator
- `run_all_tests_with_logging.py` - Comprehensive test runner

### 📊 **Benchmark Suites** (moved to `drone_sim/tests/benchmarks/`)

- `advanced_tests.py` - Advanced feature benchmarks
- `performance_benchmarks.py` - Performance measurements
- `maneuver_tests.py` - Flight maneuver benchmarks
- `environmental_tests.py` - Environmental condition benchmarks

### 🔧 **Helper Utilities** (moved to `examples/helpers/`)

- `physics_constraint_enforcer.py` - Physics validation
- `comprehensive_issue_analysis.py` - Issue analysis
- `physics_analysis.py` - Physics analysis tools
- `background_validation_demo.py` - Validation demos

### 🎮 **Executable Examples** (remained in `examples/`)

- `basic_simulation.py` - Simple simulation
- `realtime_simulation.py` - Real-time simulation
- `console_*.py` - Console interfaces
- `demo_*.py` - Demonstration scripts

## Quick Reference

| Task                 | Command                                                       |
| -------------------- | ------------------------------------------------------------- |
| Run basic example    | `python examples/basic_simulation.py`                         |
| Run system tests     | `python drone_sim/tests/system/integrated_test_runner.py`     |
| Run benchmarks       | `python drone_sim/tests/benchmarks/performance_benchmarks.py` |
| Use helper utilities | `from examples.helpers import PhysicsConstraintEnforcer`      |
| Run all tests        | `python drone_sim/tests/system/run_all_tests_with_logging.py` |

This reorganization provides a cleaner, more maintainable structure that separates concerns and makes the codebase easier to navigate and understand.
