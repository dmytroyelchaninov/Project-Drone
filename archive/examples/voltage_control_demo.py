#!/usr/bin/env python3
"""
Voltage-based Control System Demo
Demonstrates the new modular input architecture with voltage control
"""
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from input.hub import InputHub, HubConfig
from input.devices import KeyboardDeviceConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main demo function"""
    print("🚁 Voltage-Based Drone Control Demo")
    print("=" * 50)
    
    # Create hub configuration
    keyboard_config = KeyboardDeviceConfig(
        name="drone_keyboard",
        poll_rate=50.0,  # 50 Hz polling
        voltage_sensitivity=1.5,  # Moderate sensitivity
        differential_sensitivity=0.8
    )
    
    hub_config = HubConfig(
        name="drone_control_hub",
        update_rate=100.0,  # 100 Hz main loop
        keyboard_enabled=True,
        keyboard_config=keyboard_config,
        sensors_enabled=False,  # No sensors for this demo
        emergency_stop_enabled=True,
        watchdog_timeout=2.0
    )
    
    # Create and start the input hub
    hub = InputHub(hub_config)
    
    # Add callbacks to monitor the system
    hub.add_control_callback(log_control_data)
    hub.add_emergency_callback(handle_emergency)
    
    print("\n📡 Starting input hub...")
    if not hub.start():
        print("❌ Failed to start input hub")
        return
    
    print("✅ Input hub started successfully!")
    print("\n🎮 Control Instructions:")
    print("  SPACE     - Throttle up")
    print("  SHIFT     - Throttle down") 
    print("  ←/→       - Roll left/right")
    print("  ↑/↓       - Pitch forward/backward")
    print("  A/D       - Yaw left/right")
    print("  ESC       - Emergency stop")
    print("  CTRL+C    - Exit")
    print("\n" + "=" * 50)
    
    try:
        # Main demo loop
        last_status_time = 0
        
        while hub.running:
            current_time = time.time()
            
            # Print status every 2 seconds
            if current_time - last_status_time > 2.0:
                print_status(hub)
                last_status_time = current_time
            
            # Sleep briefly
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")
    
    finally:
        print("🔌 Stopping input hub...")
        hub.stop()
        print("✅ Demo completed")

def log_control_data(control_data: dict):
    """Log control data (called by hub callback)"""
    if 'keyboard' in control_data:
        keyboard_data = control_data['keyboard']
        
        # Only log if there's meaningful activity
        if any(abs(keyboard_data.get(key, 0)) > 0.1 for key in ['throttle_delta', 'roll_delta', 'pitch_delta', 'yaw_delta']):
            voltage_data = keyboard_data.get('voltage_data', {})
            if voltage_data and 'thrusts' in voltage_data:
                thrusts = voltage_data['thrusts']
                total_thrust = voltage_data.get('total_thrust', 0)
                
                logger.info(f"Control Active - Total Thrust: {total_thrust:.2f}N, "
                           f"Engine Thrusts: [{thrusts[0]:.2f}, {thrusts[1]:.2f}, {thrusts[2]:.2f}, {thrusts[3]:.2f}]N")

def handle_emergency():
    """Handle emergency stop events"""
    print("\n🚨 EMERGENCY STOP ACTIVATED! 🚨")
    print("All engines stopped for safety")
    print("Press any key to reset (except ESC)")

def print_status(hub: InputHub):
    """Print current system status"""
    status = hub.get_control_status()
    voltage_data = hub.get_voltage_commands()
    
    print(f"\n📊 System Status:")
    print(f"  Hub Running: {'✅' if status['running'] else '❌'}")
    print(f"  Emergency Stop: {'🚨' if status['emergency_stop'] else '✅'}")
    print(f"  Devices Connected: {status['devices_connected']}")
    
    if 'keyboard' in status:
        kb_status = status['keyboard']
        print(f"  Active Keys: {kb_status['active_keys']}")
        print(f"  Throttle Delta: {kb_status['throttle_delta']:.2f}V")
        print(f"  Roll Delta: {kb_status['roll_delta']:.2f}V")
        print(f"  Pitch Delta: {kb_status['pitch_delta']:.2f}V")
        print(f"  Yaw Delta: {kb_status['yaw_delta']:.2f}V")
    
    if voltage_data:
        print(f"  Total Thrust: {voltage_data.get('total_thrust', 0):.2f}N")
        voltages = voltage_data.get('voltages', [0, 0, 0, 0])
        print(f"  Engine Voltages: [{voltages[0]:.1f}, {voltages[1]:.1f}, {voltages[2]:.1f}, {voltages[3]:.1f}]V")

def test_voltage_physics():
    """Test the voltage to thrust conversion"""
    print("\n🔬 Testing Voltage-to-Thrust Physics")
    print("-" * 40)
    
    from input.devices import VoltageController, VoltageControllerConfig
    
    # Create voltage controller
    config = VoltageControllerConfig(
        name="test_controller",
        hover_voltage=6.0,
        max_rpm_per_volt=800.0
    )
    
    controller = VoltageController(config)
    controller.start()
    
    # Test different voltage levels
    test_voltages = [0, 3, 6, 9, 12]  # 0V to 12V
    
    print("Voltage -> RPM -> Thrust conversion:")
    print("Voltage(V) | RPM    | Thrust(N)")
    print("-" * 30)
    
    for voltage in test_voltages:
        # Set all engines to same voltage
        voltages = np.full(4, voltage)
        controller.set_engine_voltages(voltages)
        
        # Wait for update
        time.sleep(0.1)
        data = controller.poll()
        
        if data:
            avg_rpm = np.mean(data['rpms'])
            avg_thrust = np.mean(data['thrusts'])
            print(f"{voltage:7.1f}    | {avg_rpm:6.0f} | {avg_thrust:8.3f}")
    
    controller.stop()
    print()

if __name__ == "__main__":
    # Run physics test first
    test_voltage_physics()
    
    # Then run main demo
    main() 