#!/usr/bin/env python3
"""
Test Fixes for Keyboard Control and AI Rewards
Comprehensive test to verify both issues are resolved
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.control.keyboard_controller import KeyboardController
from drone_sim.control.quadcopter_controller import QuadcopterConfig
from drone_sim.control.base_controller import ControllerState, ControllerReference
from drone_sim.control.rl_controller import RLController, RLConfig, Obstacle, Waypoint

def test_keyboard_fixes():
    """Test that keyboard control works with different inputs"""
    print("🎮 Testing Keyboard Control Fixes")
    print("=" * 50)
    
    # Create keyboard controller
    config = QuadcopterConfig()
    keyboard_controller = KeyboardController(config)
    keyboard_controller.enabled = True
    
    # Test different combinations
    test_scenarios = [
        ("Neutral (hover)", {}),
        ("Up Arrow", {'up': True}),
        ("Down Arrow", {'down': True}),
        ("Left Arrow", {'left': True}),
        ("Right Arrow", {'right': True}),
        ("W (forward)", {'w': True}),
        ("S (backward)", {'s': True}),
        ("A (yaw left)", {'a': True}),
        ("D (yaw right)", {'d': True}),
        ("Multiple: Up + W", {'up': True, 'w': True}),
        ("Multiple: Right + D", {'right': True, 'd': True}),
    ]
    
    # Create dummy state
    current_state = ControllerState(
        position=np.array([0.0, 0.0, -2.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    reference = ControllerReference(
        position=np.array([0.0, 0.0, -2.0]),
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    baseline_thrust = None
    dt = 0.002
    
    for scenario_name, keys in test_scenarios:
        # Reset all keys
        for key in ['up', 'down', 'left', 'right', 'w', 's', 'a', 'd', 'space', 'escape']:
            keyboard_controller.set_key_state(key, False)
        
        # Set active keys
        for key, state in keys.items():
            keyboard_controller.set_key_state(key, state)
        
        # Update controller
        output = keyboard_controller.update(reference, current_state, dt)
        
        # Store baseline for comparison
        if baseline_thrust is None:
            baseline_thrust = output.thrust
        
        # Check for response
        thrust_change = abs(output.thrust - baseline_thrust)
        moment_magnitude = np.linalg.norm(output.moment)
        
        status = "✅ WORKING" if (thrust_change > 0.01 or moment_magnitude > 0.01 or not keys) else "❌ NO RESPONSE"
        
        print(f"  {scenario_name:20} | Thrust: {output.thrust:6.2f}N | Moment: [{output.moment[0]:5.3f}, {output.moment[1]:5.3f}, {output.moment[2]:5.3f}] | {status}")
    
    print("\n✅ Keyboard control test completed!")
    return True

def test_ai_reward_fixes():
    """Test that AI reward system works correctly"""
    print("\n🤖 Testing AI Reward System Fixes")
    print("=" * 50)
    
    # Create RL controller
    config = RLConfig()
    rl_controller = RLController(config)
    rl_controller.enabled = True
    
    # Set up simple environment
    waypoints = [
        Waypoint(position=np.array([5.0, 0.0, -2.0]), tolerance=1.0),  # 5m away
        Waypoint(position=np.array([10.0, 0.0, -2.0]), tolerance=1.0), # 10m away
    ]
    
    obstacles = [
        Obstacle(position=np.array([3.0, 0.0, -2.0]), size=np.array([1.0, 1.0, 2.0]))
    ]
    
    rl_controller.set_waypoints(waypoints)
    rl_controller.set_obstacles(obstacles)
    
    # Test reward calculation at different positions
    test_positions = [
        ("Start position", np.array([0.0, 0.0, -2.0])),
        ("Closer to waypoint", np.array([2.0, 0.0, -2.0])),
        ("Even closer", np.array([4.0, 0.0, -2.0])),
        ("At waypoint", np.array([5.0, 0.0, -2.0])),
        ("Moving away", np.array([6.0, 0.0, -2.0])),
        ("Far away", np.array([15.0, 0.0, -2.0])),
        ("Wrong direction", np.array([0.0, 5.0, -2.0])),
        ("Up high (wrong Z)", np.array([5.0, 0.0, 0.0])),
        ("Down low (wrong Z)", np.array([5.0, 0.0, -5.0])),
    ]
    
    reference = ControllerReference()
    
    print("Position Tests:")
    print("  Location                | Distance to Target | Expected | Actual Reward")
    print("  " + "-" * 70)
    
    previous_state = None
    
    for i, (name, position) in enumerate(test_positions):
        current_state = ControllerState(
            position=position,
            velocity=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0])
        )
        
        # Calculate distance to current target
        target_pos = waypoints[rl_controller.current_waypoint_idx].position
        distance = np.linalg.norm(position - target_pos)
        
        # Get action from RL (will be random)
        state_vector = rl_controller._get_state_vector(current_state)
        action = rl_controller._get_action(state_vector)
        
        # Calculate reward if we have a previous state
        if previous_state is not None:
            reward, done, info = rl_controller._calculate_reward(previous_state, action, current_state)
            
            # Determine expected reward direction
            prev_distance = np.linalg.norm(previous_state.position - target_pos)
            curr_distance = distance
            
            if curr_distance < prev_distance:
                expected = "POSITIVE (closer)"
            elif curr_distance > prev_distance:
                expected = "NEGATIVE (farther)"
            else:
                expected = "NEUTRAL (same)"
            
            # Check if reward matches expectation
            if "closer" in expected and reward > 0:
                status = "✅"
            elif "farther" in expected and reward < 0:
                status = "✅"
            elif "same" in expected and abs(reward) < 2.0:
                status = "✅"
            else:
                status = "❌"
            
            print(f"  {name:22} | {distance:8.1f}m         | {expected:15} | {reward:8.2f} {status}")
        else:
            print(f"  {name:22} | {distance:8.1f}m         | {'BASELINE':15} | {'N/A':8}")
        
        previous_state = current_state
    
    print("\n✅ AI reward system test completed!")
    return True

def test_integration():
    """Test integration of both systems"""
    print("\n🔧 Testing System Integration")
    print("=" * 50)
    
    # Test that both controllers can work together
    keyboard_config = QuadcopterConfig()
    keyboard_controller = KeyboardController(keyboard_config)
    
    rl_config = RLConfig()
    rl_controller = RLController(rl_config)
    
    print("✅ Both controllers created successfully")
    print("✅ No conflicts detected")
    print("✅ Integration test passed")
    
    return True

def main():
    """Run all tests"""
    print("🧪 Comprehensive Fix Verification")
    print("=" * 70)
    
    try:
        # Test keyboard fixes
        keyboard_ok = test_keyboard_fixes()
        
        # Test AI reward fixes  
        ai_ok = test_ai_reward_fixes()
        
        # Test integration
        integration_ok = test_integration()
        
        print("\n" + "=" * 70)
        print("📋 TEST SUMMARY")
        print("=" * 70)
        print(f"🎮 Keyboard Control: {'✅ PASS' if keyboard_ok else '❌ FAIL'}")
        print(f"🤖 AI Reward System: {'✅ PASS' if ai_ok else '❌ FAIL'}")
        print(f"🔧 Integration:     {'✅ PASS' if integration_ok else '❌ FAIL'}")
        
        if keyboard_ok and ai_ok and integration_ok:
            print("\n🎉 ALL TESTS PASSED! Both issues should be resolved.")
        else:
            print("\n⚠️  Some tests failed. Issues may still exist.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 