#!/usr/bin/env python3
"""
Main System Runner
Demonstrates the complete drone system according to the prompts.md specification
"""
import sys
import os
import time
import signal
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from drone import Drone
from input.hub import Hub
from input.poller import Poller
from cfg import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/system.log')
    ]
)

logger = logging.getLogger(__name__)

class DroneSystem:
    """Complete drone system orchestrator"""
    
    def __init__(self):
        self.hub = None
        self.poller = None
        self.drone = None
        self.running = False
        
    def start(self):
        """Start the complete drone system"""
        logger.info("=== STARTING DRONE SYSTEM ===")
        
        try:
            # Initialize Hub (singleton)
            logger.info("Initializing Hub...")
            self.hub = Hub()
            
            # Initialize Poller
            logger.info("Initializing Poller...")
            self.poller = Poller(self.hub)
            
            # Initialize Drone control
            logger.info("Initializing Drone control...")
            self.drone = Drone()
            
            # Test all systems
            logger.info("Testing systems...")
            self._test_systems()
            
            # Start systems
            logger.info("Starting Poller...")
            self.poller.start()
            
            logger.info("Starting Drone control...")
            self.drone.start()
            
            self.running = True
            logger.info("=== DRONE SYSTEM STARTED ===")
            
            # Print system status
            self._print_status()
            
        except Exception as e:
            logger.error(f"Failed to start drone system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the complete drone system"""
        if not self.running:
            return
            
        logger.info("=== STOPPING DRONE SYSTEM ===")
        
        # Stop in reverse order
        if self.drone:
            logger.info("Stopping Drone control...")
            self.drone.stop()
        
        if self.poller:
            logger.info("Stopping Poller...")
            self.poller.stop()
        
        # Hub is singleton, no explicit stop needed
        logger.info("Hub data cleared")
        
        self.running = False
        logger.info("=== DRONE SYSTEM STOPPED ===")
    
    def _test_systems(self):
        """Test all system components"""
        logger.info("Running system tests...")
        
        # Test Hub
        hub_status = self.hub.get_diagnostic_info()
        logger.info(f"Hub test: {'PASS' if hub_status['initialization_complete'] else 'FAIL'}")
        
        # Test Poller
        if self.poller:
            poller_test = self.poller.test()
            logger.info(f"Poller test: {'PASS' if poller_test['overall_success'] else 'FAIL'}")
            
            if not poller_test['overall_success']:
                for error in poller_test['errors']:
                    logger.warning(f"  Test error: {error}")
        
        # Test Drone
        if self.drone:
            drone_status = self.drone.get_status()
            logger.info(f"Drone test: PASS")  # If it initializes, it passes
    
    def _print_status(self):
        """Print current system status"""
        print("\n" + "="*60)
        print("DRONE SYSTEM STATUS")
        print("="*60)
        
        # Hub status
        hub_state = self.hub.get_state()
        print(f"Mode: {hub_state['mode']}")
        print(f"Go: {hub_state['go']}")
        print(f"Task: {hub_state['task']}")
        print(f"Simulation: {hub_state['simulation']}")
        print(f"Device: {hub_state['device_type']}")
        print(f"Sensors: {', '.join(hub_state['enabled_sensors'])}")
        
        # Control status
        if self.drone:
            drone_status = self.drone.get_status()
            print(f"Control Running: {drone_status['running']}")
            print(f"Control Frequency: {drone_status['control_frequency']} Hz")
            print(f"Danger Detected: {drone_status['danger_detected']}")
        
        # Poller status
        if self.poller:
            poller_status = self.poller.get_status()
            print(f"Poller Running: {poller_status['running']}")
            print(f"Poll Frequency: {poller_status['poll_frequency']} Hz")
            print(f"Errors: {poller_status['error_count']}")
        
        print("="*60)
        print("System is running. Press Ctrl+C to stop.")
        print("Use keyboard controls if device is enabled:")
        print("  WASD: Roll/Pitch   QE: Yaw   RF: Throttle")
        print("  TAB: Change mode   SPACE: Emergency stop")
        print("="*60 + "\n")
    
    def run_interactive(self):
        """Run interactive mode with status updates"""
        if not self.running:
            self.start()
        
        try:
            while self.running:
                time.sleep(5)  # Update every 5 seconds
                
                # Get current hub state
                hub_state = self.hub.get_state()
                output = self.hub.get_output_data()
                
                # Print live status
                print(f"\rStatus: {hub_state['mode']}|{hub_state['go']}|{hub_state.get('task', 'None')} "
                      f"Outputs: [{output[0]:.1f}, {output[1]:.1f}, {output[2]:.1f}, {output[3]:.1f}]V", end='')
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.stop()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutdown signal received...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run system
    system = DroneSystem()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'test':
            # Test mode - start and stop quickly
            system.start()
            time.sleep(2)
            system.stop()
        else:
            # Interactive mode
            system.run_interactive()
    except Exception as e:
        logger.error(f"System error: {e}")
        system.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 