#!/usr/bin/env python3
"""
Voltage Control Demonstration
Shows the voltage-based engine control system working with keyboard input
"""
import sys
import time
import threading
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from input.devices import KeyboardDevice, VoltageController
from input.hub import Hub
from physics import QuadcopterPhysics, Environment

def main():
    """Demonstrate voltage control system"""
    print("=== VOLTAGE CONTROL DEMONSTRATION ===")
    print("This demo shows keyboard → voltage → thrust → physics")
    print("Controls: WASD (roll/pitch), QE (yaw), RF (throttle)")
    print("Press Ctrl+C to exit")
    print()
    
    # Initialize components
    keyboard = KeyboardDevice()
    voltage_controller = VoltageController()
    physics = QuadcopterPhysics()
    environment = Environment()
    
    keyboard.start()
    
    try:
        for i in range(200):  # Run for ~20 seconds at 10Hz
            # Get keyboard input
            device_data = keyboard.poll()
            
            if device_data:
                # Extract voltages from device
                voltages = device_data['voltages']
                
                # Convert voltages to thrusts
                thrusts = voltage_controller.voltages_to_thrusts(voltages)
                
                # Apply to physics
                physics.set_engine_thrusts(thrusts)
                physics.update(0.1)  # 10Hz update
                
                # Get current state
                state = physics.get_state_dict()
                position = state['position']
                euler = state['euler_angles']
                
                # Display status
                print(f"\rVoltages: [{voltages[0]:.1f}, {voltages[1]:.1f}, {voltages[2]:.1f}, {voltages[3]:.1f}]V "
                      f"Pos: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]m "
                      f"Attitude: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°", end='')
            
            time.sleep(0.1)  # 10 Hz
    
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    finally:
        keyboard.stop()
        print("\nVoltage control demo complete")

if __name__ == "__main__":
    main() 