#!/usr/bin/env python3
"""
Drone Simulator - Main executable
Launches 3D simulation with UI as described in prompts.md

Usage:
    python src/simulator [--training]
    
    --training: Launch without UI for AI training (future implementation)
"""
import sys
import os
import argparse
import logging
import time
import threading
from typing import Optional

# Add src to Python path
sys.path.insert(0, os.path.dirname(__file__))

from cfg import settings
from input.hub import Hub
from input.poller import Poller
from physics import QuadcopterPhysics, Environment
from drone.main import Drone
from ui.transmission import Transmission
from ui.emulator import Emulator

logger = logging.getLogger(__name__)

class DroneSimulator:
    """
    Main Simulator class that orchestrates all components
    As specified in prompts.md:
    1. Creates basic UI, suggests user to set up initial settings
    2. Loading Phase: Instantiates Hub, Environment, Poller, Drone, Emulator, Transmission
    3. Tests Poller if it's receiving data from sensors/device
    """
    
    def __init__(self, training_mode: bool = False):
        self.training_mode = training_mode
        
        # Core components (as specified in prompts.md)
        self.hub: Optional[Hub] = None
        self.poller: Optional[Poller] = None
        self.physics: Optional[QuadcopterPhysics] = None
        self.environment: Optional[Environment] = None
        self.drone: Optional[Drone] = None
        self.transmission: Optional[Transmission] = None
        self.emulator: Optional[Emulator] = None
        
        # Threading
        self.shutdown_event = threading.Event()
        
        logger.info(f"Drone Simulator initialized (training_mode={training_mode})")
    
    def setup_logging(self):
        """Configure logging for simulator"""
        os.makedirs('logs', exist_ok=True)
        log_level = logging.DEBUG if settings.get('GENERAL.debug', True) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/simulator.log')
            ]
        )
    
    def show_startup_ui(self) -> bool:
        """
        Creates basic UI, suggests user to set up initial settings
        As specified in prompts.md: All settings have some default mode.
        Such as connect joystick (default is keyboard), select Drone config, 
        select Environment config (obstacles, gravity value)
        """
        print("="*80)
        print("🚁 DRONE SIMULATOR")
        print("="*80)
        print()
        print("📋 INITIAL SETTINGS CONFIGURATION")
        print("-" * 40)
        
        # Current settings display
        current_device = settings.get('CURRENT_DEVICE.type', 'keyboard')
        available_sensors = list(settings.get('AVAILABLE_SENSORS', {}).keys())
        
        print(f"🎮 Device Control: {current_device.upper()}")
        print(f"📡 Sensors Available: {', '.join(available_sensors)}")
        print(f"🌍 Simulation Mode: {'ENABLED' if settings.get('SIMULATION', True) else 'DISABLED'}")
        print(f"⚙️  Poll Frequency: {settings.get('GENERAL.poll_frequency', 100)} Hz")
        print()
        
        print("🚁 DRONE CONFIGURATION")
        print("-" * 40)
        drone_cfg = settings.get_section('DRONE')
        print(f"📏 Mass: {drone_cfg.get('mass', 1.5)} kg")
        print(f"📏 Arm Length: {drone_cfg.get('arm_length', 0.225)} m")
        print(f"⚡ Max Voltage: {drone_cfg.get('max_voltage', 12.0)} V")
        print(f"🔋 Battery Voltage: {drone_cfg.get('battery_voltage', 11.1)} V")
        print()
        
        print("🌍 ENVIRONMENT CONFIGURATION")
        print("-" * 40)
        env_cfg = settings.get_section('ENVIRONMENT')
        print(f"🌍 Gravity: {env_cfg.get('gravity', 9.81)} m/s²")
        print(f"🌪️  Wind Enabled: {'YES' if env_cfg.get('wind_enabled', True) else 'NO'}")
        print(f"🌡️  Temperature: {env_cfg.get('temperature', 15.0)} °C")
        print()
        
        if not self.training_mode:
            print("🎮 UI CONFIGURATION")
            print("-" * 40)
            ui_cfg = settings.get_section('UI')
            print(f"📺 Window Size: {ui_cfg.get('window_width', 1200)}x{ui_cfg.get('window_height', 800)}")
            print(f"🎞️  Target FPS: {ui_cfg.get('target_fps', 60)}")
            print(f"📊 Show Sensors: {'YES' if ui_cfg.get('show_sensors', True) else 'NO'}")
            print()
        
        print("="*80)
        
        if not self.training_mode:
            print("💡 CONTROLS HELP:")
            print("   Mouse: Drag to rotate camera, wheel to zoom")
            print("   F1: Toggle sensor display")
            print("   F2: Toggle debug info")
            print("   F3: Toggle trajectory")
            print("   R: Reset camera")
            print("   ESC: Exit simulation")
            print()
            print("   Keyboard Flight Controls:")
            print("   Arrow keys: Pitch/Roll")
            print("   A/D: Yaw left/right")
            print("   Space/Shift: Throttle up/down")
            print()
            response = input("🚀 Press Enter to start simulation (or 'q' to quit): ")
            if response.lower() == 'q':
                return False
        else:
            print("🤖 TRAINING MODE: No UI will be displayed")
            print("   Full computational resources allocated for AI training")
            print()
        
        return True
    
    def initialize_components(self) -> bool:
        """
        Loading Phase: Instantiates all components in correct order
        As specified in prompts.md:
        Hub, Environment(Hub), Poller(hub), Drone(hub, environment), 
        Emulator(transmission), Transmission(hub, environment, Physics)
        """
        try:
            print("🔄 LOADING PHASE")
            print("=" * 50)
            
            # 1. Set simulation mode
            print("  ⚙️  Setting simulation mode...")
            settings.set('SIMULATION', True)
            
            # 2. Initialize Hub (SINGLETON)
            print("  🏢 Initializing Hub (SINGLETON)...")
            self.hub = Hub()
            
            # 3. Initialize Environment (SINGLETON) 
            print("  🌍 Initializing Environment (SINGLETON)...")
            self.environment = Environment()
            
            # 4. Initialize Physics (SINGLETON)
            print("  ⚛️  Initializing Physics (SINGLETON)...")
            self.physics = QuadcopterPhysics()
            
            # 5. Initialize Poller
            print("  📡 Initializing Poller(Hub)...")
            self.poller = Poller(self.hub)
            
            # 6. Initialize Drone control
            print("  🚁 Initializing Drone(Hub, Environment)...")
            self.drone = Drone()
            
            # 7. Initialize Transmission
            print("  🔗 Initializing Transmission(Hub, Environment, Physics)...")
            self.transmission = Transmission(self.hub, self.environment, self.physics)
            
            # 8. Initialize Emulator (only if not training mode)
            if not self.training_mode:
                print("  🎮 Initializing Emulator(Transmission)...")
                self.emulator = Emulator(self.transmission)
            
            print("  ✅ All components initialized successfully!")
            print()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            print(f"  ❌ Initialization failed: {e}")
            return False
    
    def test_components(self) -> bool:
        """
        Tests Poller if it's receiving data from sensors/device
        As specified in prompts.md: Just one time poll, to check if everything is correct
        """
        print("🧪 COMPONENT TESTING")
        print("=" * 50)
        
        try:
            # Test Hub
            print("  🏢 Testing Hub...")
            hub_state = self.hub.get_state()
            print(f"     ✅ Hub OK (mode={hub_state['mode']}, go={hub_state['go']})")
            
            # Test Poller with one-time poll (as specified in prompts.md)
            print("  📡 Testing Poller (one-time poll)...")
            if self.poller.test():
                print("     ✅ Poller OK - receiving data from sensors/device")
            else:
                print("     ⚠️  Poller WARNING - Some sensors/devices not responding")
            
            # Test Physics
            print("  ⚛️  Testing Physics...")
            physics_state = self.physics.get_state_dict()
            pos = physics_state['position']
            print(f"     ✅ Physics OK (position=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}])")
            
            # Test Environment
            print("  🌍 Testing Environment...")
            env_data = self.transmission.get_environment_data()
            print(f"     ✅ Environment OK (ground_height={env_data['ground_height']:.1f}m)")
            
            # Test Drone
            print("  🚁 Testing Drone...")
            drone_status = self.drone.get_status()
            print(f"     ✅ Drone OK (running={drone_status['running']})")
            
            # Test Transmission
            print("  🔗 Testing Transmission...")
            drone_data = self.transmission.get_drone_state()
            print(f"     ✅ Transmission OK (altitude={drone_data['position']['z']:.1f}m)")
            
            # Test Emulator (if not training mode)
            if not self.training_mode:
                print("  🎮 Testing Emulator...")
                if self.emulator.initialize():
                    print("     ✅ Emulator OK (3D rendering ready)")
                    # Close the test window
                    import pygame
                    pygame.quit()
                else:
                    print("     ❌ Emulator FAILED (3D rendering not available)")
                    return False
            
            print("  ✅ All component tests completed successfully!")
            print()
            return True
            
        except Exception as e:
            logger.error(f"Component testing failed: {e}")
            print(f"  ❌ Testing failed: {e}")
            return False
    
    def start_simulation(self):
        """
        Start all simulation components and main loop
        """
        try:
            print("🚀 STARTING SIMULATION")
            print("=" * 50)
            
            # Start Poller
            print("  📡 Starting Poller...")
            if not self.poller.start():
                raise RuntimeError("Failed to start Poller")
            print("     ✅ Poller started")
            
            # Start Drone control
            print("  🚁 Starting Drone control...")
            if not self.drone.start():
                raise RuntimeError("Failed to start Drone")
            print("     ✅ Drone control started")
            
            print("  ✅ All systems operational!")
            print()
            
            if self.training_mode:
                self._run_training_loop()
            else:
                self._run_interactive_simulation()
                
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            print(f"❌ Simulation start failed: {e}")
            self.shutdown()
    
    def _run_training_loop(self):
        """
        Run simulation in training mode (no UI)
        For AI training - full computational resources for model training
        """
        print("🤖 TRAINING MODE ACTIVE")
        print("=" * 50)
        print("🧠 AI training mode - no UI, full GPU/CPU for model training")
        print("⏱️  High-frequency physics updates for optimal training")
        print("📊 Real-time physics validation enabled")
        print("⏹️  Press Ctrl+C to stop training")
        print()
        
        training_start = time.time()
        update_count = 0
        
        try:
            while not self.shutdown_event.is_set():
                # High-frequency updates for training
                self.transmission.update_physics_from_hub()
                update_count += 1
                
                # Print progress every 1000 updates
                if update_count % 1000 == 0:
                    elapsed = time.time() - training_start
                    rate = update_count / elapsed
                    drone_state = self.transmission.get_drone_state()
                    pos = drone_state['position']
                    print(f"🔄 Training: {update_count} updates, {rate:.1f} Hz, "
                          f"Pos: [{pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}]")
                
                # High-frequency update (1000 Hz for training)
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            elapsed = time.time() - training_start
            print(f"\n🏁 Training stopped after {elapsed:.1f}s ({update_count} updates)")
    
    def _run_interactive_simulation(self):
        """
        Run simulation with 3D UI
        """
        print("🎮 INTERACTIVE SIMULATION MODE")
        print("=" * 50)
        print("🖼️  Launching 3D emulator with real-time sensor data...")
        print("🎯 Third-person drone view with interactive controls")
        print("📊 Live sensor data overlay enabled")
        print()
        
        # Start emulator in main thread
        try:
            self.emulator.run()
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")
        except Exception as e:
            logger.error(f"Emulator error: {e}")
            print(f"❌ Emulator error: {e}")
    
    def shutdown(self):
        """
        Shutdown all components gracefully
        """
        print("\n🛑 SHUTTING DOWN SIMULATOR")
        print("=" * 50)
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop emulator
            if self.emulator:
                self.emulator.stop()
                print("  🎮 Emulator stopped")
            
            # Stop drone control
            if self.drone:
                self.drone.stop()
                print("  🚁 Drone control stopped")
            
            # Stop poller
            if self.poller:
                self.poller.stop()
                print("  📡 Poller stopped")
            
            print("  ✅ Simulator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            print(f"  ❌ Shutdown error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Drone Simulator')
    parser.add_argument('--training', action='store_true', 
                       help='Run in training mode (no UI, optimized for AI training)')
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = DroneSimulator(training_mode=args.training)
    
    try:
        # Setup logging
        simulator.setup_logging()
        
        # Show startup UI (as specified in prompts.md)
        if not simulator.show_startup_ui():
            print("🚫 Simulator cancelled by user")
            return 0
        
        # Initialize all components (Loading Phase from prompts.md)
        if not simulator.initialize_components():
            print("❌ Failed to initialize components")
            return 1
        
        # Test components (as specified in prompts.md)
        if not simulator.test_components():
            print("❌ Component testing failed")
            return 1
        
        # Start simulation
        simulator.start_simulation()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"Simulator error: {e}")
        print(f"❌ Simulator error: {e}")
        return 1
    finally:
        simulator.shutdown()
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 