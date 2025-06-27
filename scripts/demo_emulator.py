#!/usr/bin/env python3
"""
Emulator Demo Script
Demonstrates the 3D drone emulator with various scenarios

Usage:
    python scripts/demo_emulator.py [scenario_name]
    
Available scenarios:
    - basic: Basic hover demonstration
    - flight_pattern: Automated flight pattern
    - manual: Manual keyboard control
"""
import sys
import os
import argparse
import time
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cfg import settings
from input.hub import Hub
from input.poller import Poller
from physics import QuadcopterPhysics, Environment
from drone.main import Drone
from ui.transmission import Transmission
from ui.emulator import Emulator

class EmulatorDemo:
    """Demo scenarios for the 3D emulator"""
    
    def __init__(self):
        self.hub = None
        self.physics = None
        self.transmission = None
        self.emulator = None
        
    def setup_components(self):
        """Initialize all components for demo"""
        # Set simulation mode
        settings.set('SIMULATION', True)
        
        # Initialize components
        self.hub = Hub()
        environment = Environment()
        self.physics = QuadcopterPhysics()
        poller = Poller(self.hub)
        drone = Drone()
        
        # Start systems
        poller.start()
        drone.start()
        
        # Initialize UI components
        self.transmission = Transmission(self.hub, environment, self.physics)
        self.emulator = Emulator(self.transmission)
        
        return True
    
    def run_basic_scenario(self):
        """Basic hover demonstration"""
        print("Running basic hover scenario...")
        print("The drone will hover in place demonstrating basic physics")
        
        # Set hover voltages
        hover_voltage = 8.0  # Approximate hover voltage
        self.hub.set_output_data([hover_voltage] * 4)
        
        # Run emulator
        self.emulator.run()
    
    def run_flight_pattern_scenario(self):
        """Automated flight pattern demonstration"""
        print("Running flight pattern scenario...")
        print("The drone will execute an automated flight pattern")
        
        # This would require implementing a flight controller
        # For now, just run basic hover
        self.run_basic_scenario()
    
    def run_manual_scenario(self):
        """Manual keyboard control demonstration"""
        print("Running manual control scenario...")
        print("Use keyboard to control the drone:")
        print("  Arrow keys: Pitch/Roll")
        print("  A/D: Yaw left/right")
        print("  Space/Shift: Throttle up/down")
        print("  ESC: Exit")
        
        # Run emulator with keyboard control
        self.emulator.run()

def main():
    parser = argparse.ArgumentParser(description='Emulator Demo')
    parser.add_argument('scenario', nargs='?', default='basic',
                       choices=['basic', 'flight_pattern', 'manual'],
                       help='Demo scenario to run')
    
    args = parser.parse_args()
    
    demo = EmulatorDemo()
    
    try:
        print("Initializing emulator demo...")
        if not demo.setup_components():
            print("Failed to setup components")
            return 1
        
        # Run selected scenario
        if args.scenario == 'basic':
            demo.run_basic_scenario()
        elif args.scenario == 'flight_pattern':
            demo.run_flight_pattern_scenario()
        elif args.scenario == 'manual':
            demo.run_manual_scenario()
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 