#!/usr/bin/env python3
"""
Simple Real-time Test
Test the real-time simulation components without full UI
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.core.simulator import Simulator, SimulationConfig
from drone_sim.core.state_manager import StateManager, DroneState
from drone_sim.physics.rigid_body import RigidBody, RigidBodyConfig
from drone_sim.physics.environment import Environment, EnvironmentConfig
from drone_sim.control.keyboard_controller import KeyboardController, KeyboardConfig
from drone_sim.control.rl_controller import RLController, RLConfig, Obstacle, Waypoint
from drone_sim.control.pid_controller import PIDController
from drone_sim.control.base_controller import ControllerReference, ControllerState
from drone_sim.utils.background_validator import BackgroundValidator
from drone_sim.utils.test_logger import TestLogger

def test_keyboard_controller():
    """Test keyboard controller functionality"""
    print("🎮 Testing Keyboard Controller...")
    
    config = KeyboardConfig()
    controller = KeyboardController(config)
    
    # Test basic functionality
    print(f"   ✅ Controller created: {controller.name}")
    print(f"   ✅ Control mode: {controller.control_mode}")
    print(f"   ✅ Emergency stop: {controller.emergency_stop}")
    
    # Test control info
    info = controller.get_control_info()
    print(f"   ✅ Control info: {len(info)} fields")
    
    controller.reset()
    print("   ✅ Controller reset successful")
    
    return True

def test_rl_controller():
    """Test RL controller functionality"""
    print("🤖 Testing RL Controller...")
    
    config = RLConfig()
    controller = RLController(config)
    
    # Create test environment
    obstacles = [
        Obstacle(
            position=np.array([5.0, 0.0, -2.0]),
            size=np.array([1.0, 1.0, 2.0])
        )
    ]
    
    waypoints = [
        Waypoint(position=np.array([10.0, 0.0, -2.0]), tolerance=0.5)
    ]
    
    controller.set_obstacles(obstacles)
    controller.set_waypoints(waypoints)
    
    print(f"   ✅ Controller created: {controller.name}")
    print(f"   ✅ Obstacles: {len(controller.obstacles)}")
    print(f"   ✅ Waypoints: {len(controller.waypoints)}")
    
    # Test learning stats
    stats = controller.get_learning_stats()
    print(f"   ✅ Learning stats: {len(stats)} fields")
    
    controller.reset()
    print("   ✅ Controller reset successful")
    
    return True

def test_simulation_components():
    """Test core simulation components"""
    print("⚙️ Testing Simulation Components...")
    
    # Create simulation config
    sim_config = SimulationConfig(
        dt=0.002,
        real_time_factor=1.0,
        physics_validation=True
    )
    
    # Create simulator
    simulator = Simulator(sim_config)
    print("   ✅ Simulator created")
    
    # Create rigid body
    inertia_matrix = np.diag([0.02, 0.02, 0.04])
    rigid_body_config = RigidBodyConfig(
        mass=1.5,
        inertia=inertia_matrix
    )
    rigid_body = RigidBody(rigid_body_config)
    print("   ✅ Rigid body created")
    
    # Create environment
    env_config = EnvironmentConfig(
        gravity_magnitude=9.81,
        air_density=1.225
    )
    environment = Environment(env_config)
    print("   ✅ Environment created")
    
    # Register components
    simulator.register_physics_engine(rigid_body)
    simulator.register_environment(environment)
    print("   ✅ Components registered")
    
    # Set initial state
    initial_state = DroneState()
    initial_state.position = np.array([0.0, 0.0, -2.0])
    initial_state.velocity = np.array([0.0, 0.0, 0.0])
    simulator.state_manager.set_state(initial_state)
    print("   ✅ Initial state set")
    
    return True

def test_background_validation():
    """Test background validation system"""
    print("🔍 Testing Background Validation...")
    
    validator = BackgroundValidator()
    validator.start_background_validation()
    print("   ✅ Background validator started")
    
    # Submit test events
    for i in range(5):
        validator.submit_test_event(
            "Test Event",
            "test_step",
            {
                'step': i,
                'value': np.random.random(),
                'position': [i * 1.0, 0.0, -2.0]
            },
            {'test_id': i}
        )
    
    time.sleep(0.5)  # Allow processing
    
    validator.stop_background_validation()
    print("   ✅ Background validator stopped")
    
    return True

def test_logging_system():
    """Test logging system"""
    print("📝 Testing Logging System...")
    
    logger = TestLogger("simple_realtime_test")
    logger.start_test("Simple Real-time Test")
    
    # Log some test data
    for i in range(3):
        logger.log_step(f"Test step {i+1}", {
            "step_number": i+1,
            "test_value": np.random.random(),
            "position": [i * 1.0, 0.0, -2.0]
        })
    
    logger.end_test()
    print(f"   ✅ Logs saved to: {logger.log_dir}")
    
    return True

def run_simple_simulation():
    """Run a simple simulation loop"""
    print("🚀 Running Simple Simulation Loop...")
    
    # Initialize components
    keyboard_controller = KeyboardController()
    rl_controller = RLController()
    pid_controller = PIDController()
    
    # Create test environment for RL
    obstacles = [
        Obstacle(position=np.array([5.0, 0.0, -2.0]), size=np.array([1.0, 1.0, 2.0]))
    ]
    waypoints = [
        Waypoint(position=np.array([10.0, 0.0, -2.0]), tolerance=0.5)
    ]
    rl_controller.set_obstacles(obstacles)
    rl_controller.set_waypoints(waypoints)
    
    # Simulation state
    current_state = ControllerState(
        position=np.array([0.0, 0.0, -2.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    reference = ControllerReference()
    dt = 0.002
    
    print("   Running simulation for 5 seconds...")
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < 5.0:
        # Test keyboard controller
        keyboard_output = keyboard_controller.update(reference, current_state, dt)
        
        # Test RL controller
        rl_output = rl_controller.update(reference, current_state, dt)
        
        # Test PID controller
        pid_output = pid_controller.update(reference, current_state, dt)
        
        # Simple state update (just for testing)
        current_state.position += current_state.velocity * dt
        
        step_count += 1
        
        # Print progress every second
        if step_count % 500 == 0:
            elapsed = time.time() - start_time
            print(f"   Step {step_count}, Time: {elapsed:.1f}s, Pos: {current_state.position}")
        
        time.sleep(dt)
    
    print(f"   ✅ Simulation completed: {step_count} steps")
    return True

def main():
    """Main test function"""
    print("🧪 Simple Real-time System Test")
    print("=" * 40)
    
    tests = [
        ("Keyboard Controller", test_keyboard_controller),
        ("RL Controller", test_rl_controller),
        ("Simulation Components", test_simulation_components),
        ("Background Validation", test_background_validation),
        ("Logging System", test_logging_system),
        ("Simple Simulation", run_simple_simulation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Real-time system is ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 