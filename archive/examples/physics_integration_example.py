#!/usr/bin/env python3
"""
Physics Integration Example
Shows how to connect the voltage control system to existing drone physics
"""
import sys
import os
import time
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import existing physics
from drone_sim.physics.quadcopter_physics import QuadcopterPhysics, QuadcopterPhysicsConfig
from drone_sim.core.state_manager import StateManager
from drone_sim.core.simulator import DroneState

# Import new voltage control system
from input.hub import InputHub, HubConfig
from input.devices import KeyboardDeviceConfig

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoltagePhysicsIntegration:
    """
    Integration class that connects voltage control to physics simulation
    """
    
    def __init__(self):
        # Initialize physics
        self.physics_config = QuadcopterPhysicsConfig(
            mass=1.5,
            arm_length=0.225,
            max_thrust_per_engine=10.0
        )
        self.physics = QuadcopterPhysics(self.physics_config)
        
        # Initialize state manager
        self.state_manager = StateManager()
        
        # Initialize voltage control hub
        keyboard_config = KeyboardDeviceConfig(
            name="physics_keyboard",
            poll_rate=100.0,  # Match physics update rate
            voltage_sensitivity=1.0,
            differential_sensitivity=0.5
        )
        
        hub_config = HubConfig(
            name="physics_control_hub",
            update_rate=100.0,
            keyboard_enabled=True,
            keyboard_config=keyboard_config,
            emergency_stop_enabled=True
        )
        
        self.input_hub = InputHub(hub_config)
        
        # Simulation state
        self.running = False
        self.dt = 0.01  # 100 Hz simulation
        
    def start(self):
        """Start the integrated simulation"""
        logger.info("Starting voltage-physics integration...")
        
        # Start input hub
        if not self.input_hub.start():
            logger.error("Failed to start input hub")
            return False
        
        # Initialize physics state
        initial_state = DroneState()
        initial_state.position = np.array([0.0, 0.0, -1.0])  # 1m above ground
        self.state_manager.set_state(initial_state)
        
        self.running = True
        logger.info("Integration started successfully")
        return True
    
    def stop(self):
        """Stop the simulation"""
        logger.info("Stopping integration...")
        self.running = False
        self.input_hub.stop()
        logger.info("Integration stopped")
    
    def update(self):
        """Single update step"""
        # Get voltage control data
        voltage_data = self.input_hub.get_voltage_commands()
        
        if voltage_data and not self.input_hub.emergency_stop:
            # Apply engine thrusts from voltage controller
            engine_thrusts = voltage_data['thrusts']
            self.physics.set_engine_thrusts(engine_thrusts)
        else:
            # No input or emergency stop - maintain hover
            hover_thrust = self.physics.get_hover_thrust_per_engine()
            hover_thrusts = np.full(4, hover_thrust)
            self.physics.set_engine_thrusts(hover_thrusts)
        
        # Update physics
        self.physics.update(self.dt)
        
        # Update state manager with new physics state
        physics_state = self.physics.get_state_dict()
        
        # Convert to DroneState format
        drone_state = DroneState()
        drone_state.position = physics_state['position']
        drone_state.quaternion = physics_state['quaternion']
        drone_state.velocity = physics_state['velocity']
        drone_state.angular_velocity = physics_state['angular_velocity']
        
        self.state_manager.set_state(drone_state)
    
    def get_status(self):
        """Get current simulation status"""
        physics_state = self.physics.get_state_dict()
        control_status = self.input_hub.get_control_status()
        voltage_data = self.input_hub.get_voltage_commands()
        
        return {
            'physics': {
                'position': physics_state['position'].tolist(),
                'velocity': physics_state['velocity'].tolist(),
                'euler_angles': self.physics.get_euler_angles().tolist(),
                'engine_thrusts': physics_state['engine_thrusts'].tolist()
            },
            'control': control_status,
            'voltage': voltage_data
        }
    
    def run_simulation(self, duration=60.0):
        """Run simulation for specified duration"""
        if not self.start():
            return
        
        print("\n🚁 Physics Integration Demo")
        print("=" * 50)
        print("Controls:")
        print("  SPACE/SHIFT - Throttle up/down")
        print("  Arrow keys  - Roll/pitch")
        print("  A/D         - Yaw")
        print("  ESC         - Emergency stop")
        print("  CTRL+C      - Exit")
        print("=" * 50)
        
        start_time = time.time()
        last_status_time = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                # Update simulation
                self.update()
                
                # Print status every 2 seconds
                if time.time() - last_status_time > 2.0:
                    self.print_status()
                    last_status_time = time.time()
                
                # Maintain 100 Hz update rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        finally:
            self.stop()
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        
        pos = status['physics']['position']
        vel = status['physics']['velocity']
        euler = status['physics']['euler_angles']
        thrusts = status['physics']['engine_thrusts']
        
        print(f"\n📊 Simulation Status:")
        print(f"  Position: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] m")
        print(f"  Velocity: [{vel[0]:6.2f}, {vel[1]:6.2f}, {vel[2]:6.2f}] m/s")
        print(f"  Attitude: [{np.degrees(euler[0]):6.1f}, {np.degrees(euler[1]):6.1f}, {np.degrees(euler[2]):6.1f}] deg")
        print(f"  Thrusts:  [{thrusts[0]:5.2f}, {thrusts[1]:5.2f}, {thrusts[2]:5.2f}, {thrusts[3]:5.2f}] N")
        
        if status['control']['emergency_stop']:
            print("  🚨 EMERGENCY STOP ACTIVE")
        elif status['control']['keyboard']['active_keys'] > 0:
            print(f"  🎮 Active control ({status['control']['keyboard']['active_keys']} keys)")
        else:
            print("  ✈️  Hovering")

def test_voltage_to_thrust_mapping():
    """Test the voltage to thrust mapping"""
    print("\n🔬 Testing Voltage-to-Thrust Mapping")
    print("-" * 40)
    
    from input.devices import VoltageController, VoltageControllerConfig
    
    config = VoltageControllerConfig(
        name="test_mapping",
        hover_voltage=6.0,
        max_rpm_per_volt=800.0
    )
    
    controller = VoltageController(config)
    controller.start()
    
    # Test voltage range
    voltages = np.linspace(0, 12, 13)  # 0V to 12V in 1V steps
    
    print("Voltage(V) | RPM    | Thrust(N) | Total(N)")
    print("-" * 45)
    
    for voltage in voltages:
        voltages_array = np.full(4, voltage)
        controller.set_engine_voltages(voltages_array)
        time.sleep(0.05)  # Let it settle
        
        data = controller.poll()
        if data:
            rpm = data['rpms'][0]  # All engines same
            thrust = data['thrusts'][0]
            total = data['total_thrust']
            print(f"{voltage:8.1f}   | {rpm:6.0f} | {thrust:8.3f} | {total:7.3f}")
    
    controller.stop()
    print()

def main():
    """Main function"""
    # Test voltage mapping first
    test_voltage_to_thrust_mapping()
    
    # Run integration demo
    integration = VoltagePhysicsIntegration()
    integration.run_simulation(duration=120.0)  # 2 minutes

if __name__ == "__main__":
    main() 