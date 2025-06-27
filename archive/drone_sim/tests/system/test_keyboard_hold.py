#!/usr/bin/env python3
"""
Test script to verify keyboard hold functionality works correctly.
This will show real-time thrust values when keys are pressed and held.
"""

import time
import threading
from drone_sim.control.keyboard_controller import KeyboardController, ThrustControlConfig
from drone_sim.control.quadcopter_controller import QuadcopterConfig
from drone_sim.control.base_controller import ControllerReference, ControllerState
import numpy as np

def test_keyboard_hold():
    """Test keyboard controller with simulated key holds"""
    print("🎮 Testing Keyboard Hold Functionality")
    print("=" * 50)
    
    # Create controller
    config = QuadcopterConfig()
    controller = KeyboardController(config)
    
    # Test sequence
    print("\n📋 Test Sequence:")
    print("1. Starting with no keys pressed (hover)")
    print("2. Simulating UP arrow press and hold")
    print("3. Simulating key release")
    print("4. Simulating DOWN arrow press and hold") 
    print("5. Simulating key release")
    print("6. Testing RIGHT arrow for roll control")
    
    # Create reference and state objects
    reference = ControllerReference()
    reference.simulation_time = 0.0
    
    state = ControllerState()
    state.position = np.array([0.0, 0.0, -2.0])
    state.velocity = np.array([0.0, 0.0, 0.0])
    
    dt = 0.1  # 10 Hz updates
    
    print(f"\n🚀 Starting test (dt={dt}s)...")
    print(f"⚡ Expected hover thrust: {controller.thrust_config.hover_thrust:.2f}N")
    print("-" * 50)
    
    # Phase 1: No keys (should hover)
    print("\n📍 Phase 1: No keys pressed")
    for i in range(5):
        output = controller.update(reference, state, dt)
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N, Moment={output.moment}")
        time.sleep(0.1)
    
    # Phase 2: UP arrow press and hold
    print("\n📍 Phase 2: UP arrow pressed and held")
    controller.set_key_state('up', True)
    
    for i in range(15):  # Hold for 1.5 seconds
        output = controller.update(reference, state, dt) 
        hold_duration = controller.get_key_hold_duration('up')
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N, Hold={hold_duration:.2f}s")
        time.sleep(0.1)
    
    # Phase 3: Release UP arrow
    print("\n📍 Phase 3: UP arrow released")
    controller.set_key_state('up', False)
    
    for i in range(10):  # Watch thrust return to hover
        output = controller.update(reference, state, dt)
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N (returning to hover)")
        time.sleep(0.1)
    
    # Phase 4: DOWN arrow press and hold  
    print("\n📍 Phase 4: DOWN arrow pressed and held")
    controller.set_key_state('down', True)
    
    for i in range(15):  # Hold for 1.5 seconds
        output = controller.update(reference, state, dt)
        hold_duration = controller.get_key_hold_duration('down')
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N, Hold={hold_duration:.2f}s")
        time.sleep(0.1)
    
    # Phase 5: Release DOWN arrow
    print("\n📍 Phase 5: DOWN arrow released")
    controller.set_key_state('down', False)
    
    for i in range(10):  # Watch thrust return to hover
        output = controller.update(reference, state, dt)
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N (returning to hover)")
        time.sleep(0.1)
    
    # Phase 6: RIGHT arrow for roll
    print("\n📍 Phase 6: RIGHT arrow for roll control")
    controller.set_key_state('right', True)
    
    for i in range(10):  # Hold for 1 second
        output = controller.update(reference, state, dt)
        hold_duration = controller.get_key_hold_duration('right')
        roll_moment = output.moment[0]  # Roll is X-axis
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N, Roll={roll_moment:.2f}Nm, Hold={hold_duration:.2f}s")
        time.sleep(0.1)
    
    # Release RIGHT arrow
    controller.set_key_state('right', False)
    
    for i in range(5):
        output = controller.update(reference, state, dt)
        roll_moment = output.moment[0]
        print(f"   Step {i+1}: Thrust={output.thrust:.2f}N, Roll={roll_moment:.2f}Nm (released)")
        time.sleep(0.1)
    
    print("\n" + "=" * 50)
    print("✅ Keyboard hold test completed!")
    print("\n📊 Expected Results:")
    print(f"   - Hover thrust: ~{controller.thrust_config.hover_thrust:.1f}N")
    print(f"   - Max UP thrust: ~{controller.thrust_config.max_thrust:.1f}N")
    print(f"   - Min DOWN thrust: ~{controller.thrust_config.min_thrust:.1f}N")
    print(f"   - Roll differential: ~{controller.thrust_config.max_differential:.1f}N")
    print("\n🎯 If values above match expectations, keyboard hold is working!")

if __name__ == "__main__":
    test_keyboard_hold() 