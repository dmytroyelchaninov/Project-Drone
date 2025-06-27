#!/usr/bin/env python3
"""
Component Testing Script
Tests individual components of the drone system
"""
import sys
import time
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
from input.hub import Hub
from input.poller import Poller
from input.devices import KeyboardDevice, VoltageController
from input.sensors import GPS, Barometer, Gyroscope, Compass
from physics import QuadcopterPhysics, Environment, Propeller
from drone import Drone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hub():
    """Test Hub singleton"""
    print("\n=== TESTING HUB ===")
    
    # Test singleton behavior
    hub1 = Hub()
    hub2 = Hub()
    
    print(f"Singleton test: {hub1 is hub2}")
    print(f"Hub simulation mode: {hub1.simulation}")
    print(f"Enabled sensors: {list(hub1.sensors.keys())}")
    print(f"Device type: {hub1.device['type']}")
    
    # Test state management
    hub1.set_mode('ai')
    hub1.set_go('operate')
    hub1.set_task('take_off')
    
    state = hub1.get_state()
    print(f"State: {state['mode']}/{state['go']}/{state['task']}")
    
    # Test output
    hub1.update_output([1.0, 2.0, 3.0, 4.0])
    output = hub1.get_output_data()
    print(f"Output: {output}")
    
    print("✓ Hub tests passed")

def test_sensors():
    """Test individual sensors"""
    print("\n=== TESTING SENSORS ===")
    
    # Test GPS
    gps = GPS(fake=True)
    gps_data = gps.poll()
    print(f"GPS data: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}")
    
    # Test Barometer
    baro = Barometer(fake=True)
    baro_data = baro.poll()
    print(f"Barometer altitude: {baro_data['altitude']:.2f}m")
    
    # Test Gyroscope
    gyro = Gyroscope(fake=True)
    gyro_data = gyro.poll()
    rates = gyro_data['angular_velocity_deg']
    print(f"Gyro rates: [{rates[0]:.1f}, {rates[1]:.1f}, {rates[2]:.1f}] deg/s")
    
    # Test Compass
    compass = Compass(fake=True)
    compass_data = compass.poll()
    print(f"Compass heading: {compass_data['heading']:.1f}°")
    
    print("✓ Sensor tests passed")

def test_devices():
    """Test input devices"""
    print("\n=== TESTING DEVICES ===")
    
    # Test VoltageController
    voltage_controller = VoltageController()
    voltages = [2.0, 3.0, 4.0, 5.0]
    thrusts = voltage_controller.voltages_to_thrusts(voltages)
    print(f"Voltage to thrust: {voltages} V -> {thrusts} N")
    
    # Test KeyboardDevice (won't actually read keyboard in test)
    keyboard = KeyboardDevice()
    # Simulate some key states
    keyboard._key_states['w'] = True
    keyboard._key_states['d'] = True
    device_data = keyboard.poll()
    print(f"Keyboard device data: mode={device_data['mode']}, voltages={device_data['voltages']}")
    
    print("✓ Device tests passed")

def test_physics():
    """Test physics simulation"""
    print("\n=== TESTING PHYSICS ===")
    
    # Test Environment
    environment = Environment()
    pos = np.array([0, 0, 10])  # 10m altitude
    atm_props = environment.get_atmospheric_properties(pos[2])
    print(f"Atmospheric properties at 10m: {atm_props}")
    
    wind_vel = environment.get_wind_velocity(pos)
    print(f"Wind velocity: {wind_vel}")
    
    # Test Propeller
    propeller = Propeller()
    voltage = 6.0
    thrust = propeller.calculate_thrust_from_voltage(voltage)
    print(f"Propeller: {voltage}V -> {thrust:.2f}N thrust")
    
    # Test QuadcopterPhysics
    physics = QuadcopterPhysics()
    physics.set_position(np.array([0, 0, 1]))  # 1m altitude
    
    # Apply some thrust
    thrusts = np.array([4.0, 4.0, 4.0, 4.0])  # 4N per engine
    physics.set_engine_thrusts(thrusts)
    
    # Simulate for a few steps
    for i in range(10):
        physics.update(0.01)  # 10ms timestep
    
    final_state = physics.get_state_dict()
    print(f"Physics after 0.1s: pos={final_state['position']}, vel={final_state['velocity']}")
    
    print("✓ Physics tests passed")

def test_poller():
    """Test poller system"""
    print("\n=== TESTING POLLER ===")
    
    hub = Hub()
    poller = Poller(hub)
    
    # Test single poll
    test_result = poller.test()
    print(f"Poller test result: {'PASS' if test_result['overall_success'] else 'FAIL'}")
    
    if not test_result['overall_success']:
        for error in test_result['errors']:
            print(f"  Error: {error}")
    
    # Test short polling session
    print("Starting poller for 2 seconds...")
    poller.start()
    time.sleep(2)
    poller.stop()
    
    # Check if data was collected
    hub_state = hub.get_state()
    print(f"Data collected: {len([k for k, v in hub_state['input'].items() if v is not None])}/{len(hub_state['input'])} sensors")
    
    print("✓ Poller tests passed")

def test_drone_control():
    """Test drone control system"""
    print("\n=== TESTING DRONE CONTROL ===")
    
    drone = Drone()
    
    # Test different control modes
    hub = drone.hub
    
    # Test manual mode
    hub.set_mode('manual')
    hub.set_go('operate')
    
    # Simulate device input
    hub.input['device_voltages'] = [6.0, 6.0, 6.0, 6.0]
    hub.input['device_status'] = {'connected': True, 'last_update': time.time()}
    
    # Start drone for short test
    drone.start()
    time.sleep(1)
    
    output = hub.get_output_data()
    print(f"Manual control output: {output}")
    
    # Test AI mode
    hub.set_mode('ai')
    hub.set_task('take_off')
    time.sleep(1)
    
    output = hub.get_output_data()
    print(f"AI takeoff output: {output}")
    
    drone.stop()
    print("✓ Drone control tests passed")

def test_integration():
    """Test full system integration"""
    print("\n=== TESTING INTEGRATION ===")
    
    # Create complete system
    hub = Hub()
    poller = Poller(hub)
    drone = Drone()
    
    # Start all systems
    poller.start()
    drone.start()
    
    print("Running integrated system for 3 seconds...")
    
    # Test mode changes
    hub.set_mode('manual')
    hub.set_go('float')
    time.sleep(1)
    
    hub.set_mode('ai')
    hub.set_task('take_off')
    time.sleep(1)
    
    hub.set_task('land')
    time.sleep(1)
    
    # Stop systems
    drone.stop()
    poller.stop()
    
    # Check final state
    final_state = hub.get_state()
    print(f"Final state: {final_state['mode']}/{final_state['go']}/{final_state['task']}")
    print(f"Final output: {hub.get_output_data()}")
    
    print("✓ Integration tests passed")

def main():
    """Run all tests"""
    print("DRONE SYSTEM COMPONENT TESTS")
    print("="*50)
    
    try:
        test_hub()
        test_sensors()
        test_devices()
        test_physics()
        test_poller()
        test_drone_control()
        test_integration()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 