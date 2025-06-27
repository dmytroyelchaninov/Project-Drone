#!/usr/bin/env python3
"""
Test Manual Control
Simple test to verify keyboard control is working in the simulation
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

def test_manual_keyboard():
    """Test manual keyboard control directly"""
    print("🎮 Testing Manual Keyboard Control")
    print("=" * 50)
    
    # Create keyboard controller
    config = QuadcopterConfig()
    keyboard_controller = KeyboardController(config)
    keyboard_controller.enabled = True
    
    print("✅ Keyboard controller created")
    
    # Test key state setting (simulate GUI input)
    print("\n🔑 Testing key state simulation:")
    
    # Simulate arrow key presses
    test_keys = ['up', 'down', 'left', 'right', 'w', 's', 'a', 'd', 'space']
    
    for key in test_keys:
        print(f"\n  Testing key: {key}")
        
        # Press key
        keyboard_controller.set_key_state(key, True)
        print(f"    Key '{key}' pressed")
        
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
        
        # Update controller
        dt = 0.002
        output = keyboard_controller.update(reference, current_state, dt)
        
        print(f"    Output - Thrust: {output.thrust:.2f}N")
        print(f"    Output - Moment: [{output.moment[0]:.3f}, {output.moment[1]:.3f}, {output.moment[2]:.3f}]")
        print(f"    Output - Motor Commands: {output.motor_commands}")
        
        # Check if this is actually changing
        if hasattr(output, 'thrust') and output.thrust != 14.71:
            print(f"    ✅ Control response detected!")
        else:
            print(f"    ⚠️  No control response - thrust should change from hover (14.71N)")
        
        # Release key
        keyboard_controller.set_key_state(key, False)
        print(f"    Key '{key}' released")
        
        # Update again to see decay
        output2 = keyboard_controller.update(reference, current_state, dt)
        print(f"    After release - Thrust: {output2.thrust:.2f}N")
        
    print("\n🎮 Manual keyboard control test completed!")
    print("\nIf you see different thrust values and moments for each key,")
    print("then the keyboard controller is working correctly.")

if __name__ == "__main__":
    test_manual_keyboard() 