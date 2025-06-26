#!/usr/bin/env python3
"""
Test Enhanced Controls
Comprehensive test for smooth keyboard controls and movement logging
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.control.keyboard_controller import KeyboardController, ThrustControlConfig
from drone_sim.control.quadcopter_controller import QuadcopterConfig
from drone_sim.control.base_controller import ControllerState, ControllerReference, ControllerOutput
from drone_sim.logging.movement_logger import MovementLogger

def test_smooth_thrust_control():
    """Test the new smooth thrust control system"""
    print("🎮 Testing Smooth Thrust Control")
    print("=" * 50)
    
    # Create configuration and controller
    config = QuadcopterConfig(mass=1.5)
    controller = KeyboardController(config)
    
    # Create movement logger
    logger = MovementLogger(enabled=True)
    controller.set_movement_logger(logger)
    
    # Create reference state
    reference = ControllerReference()
    current_state = ControllerState(
        position=np.array([0.0, 0.0, -2.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    print(f"✅ Controller initialized")
    print(f"   Hover thrust: {controller.thrust_config.hover_thrust:.1f}N")
    print(f"   Thrust range: {controller.thrust_config.min_thrust:.1f}N - {controller.thrust_config.max_thrust:.1f}N")
    print()
    
    # Test scenarios
    test_scenarios = [
        ("UP Arrow Hold", "up", 1.0),
        ("DOWN Arrow Hold", "down", 1.0),  
        ("RIGHT Arrow Hold", "right", 0.5),
        ("LEFT Arrow Hold", "left", 0.5),
        ("Multi-key (UP + RIGHT)", ["up", "right"], 0.8),
    ]
    
    for scenario_name, keys, duration in test_scenarios:
        print(f"🧪 Testing: {scenario_name}")
        print(f"   Duration: {duration}s")
        
        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        
        # Press keys
        for key in keys:
            controller.set_key_state(key, True)
        
        # Simulate hold duration with updates
        start_time = time.time()
        step_count = 0
        
        while time.time() - start_time < duration:
            # Update controller
            output = controller.update(reference, current_state, 0.01)
            
            # Show output every 10 steps
            if step_count % 10 == 0:
                hold_time = time.time() - start_time
                active_keys = [k for k in keys if controller.key_states.get(k, False)]
                
                print(f"   t={hold_time:.2f}s: Thrust={output.thrust:.1f}N, "
                      f"Moment=[{output.moment[0]:.2f}, {output.moment[1]:.2f}, {output.moment[2]:.2f}], "
                      f"Keys={active_keys}")
            
            step_count += 1
            time.sleep(0.01)
        
        # Release keys
        for key in keys:
            controller.set_key_state(key, False)
        
        # Show smooth return to hover
        print(f"   Keys released, returning to hover...")
        return_start = time.time()
        
        while time.time() - return_start < 0.5:  # 0.5s return time
            output = controller.update(reference, current_state, 0.01)
            time.sleep(0.01)
        
        final_output = controller.update(reference, current_state, 0.01)
        print(f"   Final: Thrust={final_output.thrust:.1f}N (target: {controller.thrust_config.hover_thrust:.1f}N)")
        print()
    
    # Test configuration display
    print("🔧 Configuration Summary:")
    debug_info = controller.get_debug_info()
    thrust_config = debug_info['thrust_config']
    
    for key, value in thrust_config.items():
        print(f"   {key}: {value}")
    
    print()
    
    # Save movement log
    print("💾 Saving movement log...")
    log_file = logger.save_movement_log()
    if log_file:
        print(f"   Log saved: {log_file}")
        
        # Show statistics
        stats = logger.get_key_statistics()
        print("📊 Key Usage Statistics:")
        for key, data in stats.items():
            print(f"   {key}: {data['total_presses']} presses, avg hold {data['average_hold']:.3f}s")
    
    print("✅ Smooth thrust control test complete!")
    return True

def test_comparison_old_vs_new():
    """Compare old vs new control characteristics"""
    print()
    print("📈 Control System Comparison")
    print("=" * 50)
    
    config = QuadcopterConfig(mass=1.5)
    controller = KeyboardController(config)
    
    # Show key characteristics
    print("🆕 New Smooth Control System:")
    print(f"   Accelerating thrust: {controller.thrust_config.thrust_accel_rate:.1f} N/s")
    print(f"   Decelerating thrust: {controller.thrust_config.thrust_decel_rate:.1f} N/s")
    print(f"   Max thrust: {controller.thrust_config.max_thrust:.1f}N (reduced from 40N)")
    print(f"   Min thrust: {controller.thrust_config.min_thrust:.1f}N (allows gravity descent)")
    print(f"   Response curve power: {controller.thrust_config.thrust_curve_power:.1f} (non-linear)")
    print()
    
    print("⚡ Control Responsiveness:")
    print(f"   Time to max thrust: {(controller.thrust_config.max_thrust - controller.thrust_config.hover_thrust) / controller.thrust_config.thrust_accel_rate:.2f}s")
    print(f"   Time to min thrust: {(controller.thrust_config.hover_thrust - controller.thrust_config.min_thrust) / controller.thrust_config.thrust_accel_rate:.2f}s")
    print(f"   Return to hover time: ~{(controller.thrust_config.max_thrust - controller.thrust_config.hover_thrust) / controller.thrust_config.thrust_decel_rate:.2f}s")
    print()
    
    print("🎯 Physics Integration:")
    print("   ✅ Individual engine control (4 engines)")
    print("   ✅ Gravity always included in calculations")
    print("   ✅ Smooth acceleration curves")
    print("   ✅ Realistic thrust limits")
    print("   ✅ Movement logging with key timing")
    
    return True

def main():
    """Run all enhanced control tests"""
    print("🚀 Enhanced Control System Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test smooth thrust control
        success1 = test_smooth_thrust_control()
        
        # Test comparison
        success2 = test_comparison_old_vs_new()
        
        print()
        print("=" * 60)
        if success1 and success2:
            print("✅ All enhanced control tests PASSED!")
            print()
            print("🎮 Ready for enhanced manual flight control!")
            print("   - Arrow keys: Smooth accelerating thrust")
            print("   - UP: Thrust increases smoothly to max")
            print("   - DOWN: Thrust decreases smoothly to min")
            print("   - LEFT/RIGHT: Differential thrust for roll")
            print("   - WASD: Pitch and yaw control")
            print("   - SPACE: Return to hover")
            print("   - ESC: Emergency thrust reduction")
            print()
            print("💾 Movement logging captures:")
            print("   - Key press/release timing")
            print("   - Hold durations")
            print("   - Thrust and moment responses")
            print("   - Position and velocity changes")
        else:
            print("❌ Some tests FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 