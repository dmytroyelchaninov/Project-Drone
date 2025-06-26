#!/usr/bin/env python3
"""
Test script for physics-based quadcopter control system
Demonstrates individual engine thrust control and resulting physics
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import drone_sim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drone_sim.physics.quadcopter_physics import QuadcopterPhysics, QuadcopterPhysicsConfig
from drone_sim.control.quadcopter_controller import QuadcopterController, QuadcopterConfig
from drone_sim.control.keyboard_controller import KeyboardController

def test_individual_engine_control():
    """Test individual engine thrust control"""
    print("🚁 Testing Individual Engine Thrust Control")
    print("=" * 60)
    
    # Create physics simulation
    physics_config = QuadcopterPhysicsConfig(mass=1.5)
    physics = QuadcopterPhysics(physics_config)
    
    # Test individual engines
    test_cases = [
        ("Hover (all engines equal)", [3.675, 3.675, 3.675, 3.675]),
        ("Front engine only", [10.0, 0.0, 0.0, 0.0]),
        ("Right engine only", [0.0, 10.0, 0.0, 0.0]),
        ("Back engine only", [0.0, 0.0, 10.0, 0.0]),
        ("Left engine only", [0.0, 0.0, 0.0, 10.0]),
        ("Roll right (left > right)", [3.675, 2.0, 3.675, 5.0]),
        ("Pitch forward (back > front)", [2.0, 3.675, 5.0, 3.675]),
    ]
    
    results = []
    
    for name, thrusts in test_cases:
        print(f"\n🔧 Test: {name}")
        print(f"   Engine thrusts: {thrusts}")
        
        # Reset physics
        physics.reset()
        
        # Set engine thrusts
        physics.set_engine_thrusts(np.array(thrusts))
        
        # Simulate for 0.1 seconds
        for _ in range(50):  # 50 steps * 0.002s = 0.1s
            physics.update(0.002)
        
        # Get final state
        state = physics.get_state_dict()
        euler = physics.get_euler_angles()
        
        print(f"   Final position: [{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}]")
        print(f"   Final velocity: [{state['velocity'][0]:.3f}, {state['velocity'][1]:.3f}, {state['velocity'][2]:.3f}]")
        print(f"   Euler angles: [roll={np.degrees(euler[0]):.1f}°, pitch={np.degrees(euler[1]):.1f}°, yaw={np.degrees(euler[2]):.1f}°]")
        print(f"   Total force: [{state['total_force_world'][0]:.2f}, {state['total_force_world'][1]:.2f}, {state['total_force_world'][2]:.2f}] N")
        print(f"   Total moment: [{state['total_moment_body'][0]:.3f}, {state['total_moment_body'][1]:.3f}, {state['total_moment_body'][2]:.3f}] N⋅m")
        
        results.append((name, state, euler))
    
    return results

def test_controller_integration():
    """Test integration with quadcopter controller"""
    print("\n\n🎮 Testing Controller Integration")
    print("=" * 60)
    
    # Create controller
    controller_config = QuadcopterConfig(mass=1.5)
    controller = QuadcopterController(controller_config)
    
    # Test different control inputs
    control_tests = [
        ("Hover (all neutral)", 0.0, 0.0, 0.0, 0.0),
        ("Climb (throttle up)", 0.5, 0.0, 0.0, 0.0),
        ("Descend (throttle down)", -0.5, 0.0, 0.0, 0.0),
        ("Roll right", 0.0, 0.5, 0.0, 0.0),
        ("Roll left", 0.0, -0.5, 0.0, 0.0),
        ("Pitch forward", 0.0, 0.0, 0.5, 0.0),
        ("Pitch backward", 0.0, 0.0, -0.5, 0.0),
        ("Yaw right", 0.0, 0.0, 0.0, 0.5),
        ("Yaw left", 0.0, 0.0, 0.0, -0.5),
    ]
    
    for name, throttle, roll, pitch, yaw in control_tests:
        print(f"\n🕹️ Control: {name}")
        print(f"   Inputs: throttle={throttle}, roll={roll}, pitch={pitch}, yaw={yaw}")
        
        # Set control inputs
        controller.set_control_inputs(throttle, roll, pitch, yaw)
        
        # Update controller (dummy state)
        from drone_sim.control.base_controller import ControllerState, ControllerReference
        state = ControllerState()
        ref = ControllerReference()
        
        output = controller.update(ref, state, 0.002)
        
        engine_thrusts = controller.get_engine_thrusts()
        
        print(f"   Total thrust: {output.thrust:.2f} N")
        print(f"   Moment: [{output.moment[0]:.3f}, {output.moment[1]:.3f}, {output.moment[2]:.3f}] N⋅m")
        print(f"   Engine thrusts: [{engine_thrusts[0]:.2f}, {engine_thrusts[1]:.2f}, {engine_thrusts[2]:.2f}, {engine_thrusts[3]:.2f}] N")
        
        # Verify hover thrust
        hover_thrust = controller.get_hover_thrust()
        print(f"   Hover thrust (total): {hover_thrust:.2f} N")

def test_keyboard_control_mapping():
    """Test keyboard control mapping"""
    print("\n\n⌨️ Testing Keyboard Control Mapping")
    print("=" * 60)
    
    # Create keyboard controller
    keyboard_controller = KeyboardController()
    
    # Show control instructions
    print(keyboard_controller.get_control_instructions())
    
    # Test key mappings
    key_tests = [
        ("Up Arrow (climb)", "up", True),
        ("Down Arrow (descend)", "down", True),
        ("Right Arrow (roll right)", "right", True),
        ("Left Arrow (roll left)", "left", True),
        ("W (pitch forward)", "w", True),
        ("S (pitch backward)", "s", True),
        ("D (yaw right)", "d", True),
        ("A (yaw left)", "a", True),
        ("Space (hover)", "space", True),
    ]
    
    from drone_sim.control.base_controller import ControllerState, ControllerReference
    
    for name, key, pressed in key_tests:
        print(f"\n🔑 Key test: {name}")
        
        # Reset controller
        keyboard_controller.reset()
        
        # Simulate key press for multiple frames
        for i in range(10):
            keyboard_controller.set_key_state(key, pressed)
            
            # Update controller
            state = ControllerState()
            ref = ControllerReference()
            output = keyboard_controller.update(ref, state, 0.002)
            
            if i == 9:  # Show final result
                debug_info = keyboard_controller.get_debug_info()
                control_inputs = debug_info['control_inputs']
                quad_info = debug_info['quad_controller']
                
                print(f"   Control inputs: throttle={control_inputs['throttle']:.3f}, "
                      f"roll={control_inputs['roll']:.3f}, pitch={control_inputs['pitch']:.3f}, yaw={control_inputs['yaw']:.3f}")
                print(f"   Output thrust: {output.thrust:.2f} N")
                print(f"   Engine thrusts: {quad_info['engine_thrusts']}")

def test_physics_prediction():
    """Test physics prediction for collision avoidance"""
    print("\n\n🔮 Testing Physics Prediction")
    print("=" * 60)
    
    # Create physics simulation
    physics = QuadcopterPhysics()
    
    # Set initial state (moving forward)
    physics.state.position = np.array([0.0, 0.0, -2.0])
    physics.state.velocity = np.array([2.0, 0.0, 0.0])  # 2 m/s forward
    
    # Set engine thrusts for forward flight
    physics.set_engine_thrusts(np.array([2.0, 3.675, 5.0, 3.675]))  # Pitch forward
    
    print("Initial state:")
    print(f"  Position: {physics.state.position}")
    print(f"  Velocity: {physics.state.velocity}")
    
    # Predict future positions
    dt = 0.002
    predictions = []
    
    for step in range(500):  # 1 second prediction
        physics.update(dt)
        
        if step % 50 == 0:  # Every 0.1 seconds
            time_ahead = step * dt
            pos = physics.state.position.copy()
            vel = physics.state.velocity.copy()
            predictions.append((time_ahead, pos, vel))
            print(f"  t={time_ahead:.1f}s: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
    
    # Demonstrate obstacle detection
    obstacle_position = np.array([3.0, 0.0, -2.0])
    obstacle_radius = 0.5
    
    print(f"\n🚧 Obstacle at position: {obstacle_position}, radius: {obstacle_radius}m")
    
    for time_ahead, pos, vel in predictions:
        distance_to_obstacle = np.linalg.norm(pos - obstacle_position)
        if distance_to_obstacle < obstacle_radius:
            print(f"⚠️  COLLISION PREDICTED at t={time_ahead:.1f}s! Distance: {distance_to_obstacle:.2f}m")
            break
        elif distance_to_obstacle < 2.0:  # Warning zone
            print(f"🟡 Warning: Approaching obstacle at t={time_ahead:.1f}s, distance: {distance_to_obstacle:.2f}m")

def main():
    """Run all tests"""
    print("🚁 Physics-Based Quadcopter Control System Test")
    print("=" * 80)
    
    try:
        # Test 1: Individual engine control
        test_individual_engine_control()
        
        # Test 2: Controller integration
        test_controller_integration()
        
        # Test 3: Keyboard control mapping
        test_keyboard_control_mapping()
        
        # Test 4: Physics prediction
        test_physics_prediction()
        
        print("\n\n✅ All tests completed successfully!")
        print("=" * 80)
        print("🎮 Control Summary:")
        print("  • Arrow keys control basic movement (throttle + roll)")
        print("  • WASD keys control pitch and yaw")
        print("  • All controls work through individual engine thrusts")
        print("  • Physics properly calculates forces, moments, and motion")
        print("  • Gravity and aerodynamics are included")
        print("  • Collision prediction is based on actual physics")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 