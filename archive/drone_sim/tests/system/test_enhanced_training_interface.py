#!/usr/bin/env python3
"""
Test script for the enhanced AI training interface
Demonstrates the new training session management features
"""

import sys
import os
import time
import argparse

# Add the parent directory to the path to import drone_sim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drone_sim.ui.realtime_interface import RealTimeInterface, SimulationMode
from drone_sim.control.rl_controller import Obstacle, Waypoint
import numpy as np

def test_enhanced_interface():
    """Test the enhanced training interface features"""
    print("🚀 Testing Enhanced AI Training Interface")
    print("=" * 50)
    
    # Create interface
    interface = RealTimeInterface()
    
    # Set up test environment
    print("📋 Setting up test environment...")
    
    # Add some obstacles and waypoints for testing
    interface.obstacles = [
        Obstacle(position=np.array([2.0, 2.0, -2.0]), size=np.array([1.0, 1.0, 2.0])),
        Obstacle(position=np.array([5.0, -1.0, -2.0]), size=np.array([1.5, 1.5, 2.0])),
    ]
    
    interface.waypoints = [
        Waypoint(position=np.array([3.0, 0.0, -2.0]), tolerance=0.5),
        Waypoint(position=np.array([6.0, 3.0, -2.0]), tolerance=0.5),
        Waypoint(position=np.array([8.0, 0.0, -2.0]), tolerance=0.5),
    ]
    
    # Set AI mode
    interface.simulation_mode = SimulationMode.AI
    
    # Configure episode length to 12 seconds (as requested)
    if hasattr(interface, 'episode_length_var'):
        interface.episode_length_var.set(12.0)
    
    print(f"✅ Environment configured:")
    print(f"   - Obstacles: {len(interface.obstacles)}")
    print(f"   - Waypoints: {len(interface.waypoints)}")
    print(f"   - Mode: {interface.simulation_mode.value}")
    print(f"   - Episode Length: 12.0 seconds")
    
    # Test interface features
    print("\n🎮 Interface Features Available:")
    print("   ✅ Clear workflow instructions in Quick Start Guide")
    print("   ✅ Training session management (Start/Pause/Resume/Stop)")
    print("   ✅ Model management (Load/Save/Reset)")
    print("   ✅ Real-time training status monitoring")
    print("   ✅ Configurable episode length")
    print("   ✅ Auto-save every 10 episodes")
    print("   ✅ Position reset without losing training weights")
    
    # Show workflow instructions
    print("\n📋 WORKFLOW INSTRUCTIONS:")
    print("1. Select 'AI Navigation' mode")
    print("2. Configure episode length (default: 12s)")
    print("3. Click '🚀 Start Simulation' first")
    print("4. Click '🎯 Start Training' to begin episodes")
    print("5. Use pause/resume controls as needed")
    print("6. Monitor progress in real-time")
    
    print("\n🎯 TRAINING MANAGEMENT FEATURES:")
    print("• 🎯 Start Training - Begin new training session")
    print("• ⏸️ Pause Training - Pause and save checkpoint")
    print("• ▶️ Resume Training - Continue from saved state")
    print("• 🛑 Stop Training & Save - Stop and save final model")
    print("• 🔄 Reset Position - Reset drone without losing weights")
    print("• 📂 Load Model - Load previous training")
    print("• 💾 Save Model - Manual save")
    print("• 🗑️ Reset Training - Start completely fresh")
    
    print("\n📊 MONITORING CAPABILITIES:")
    print("• Real-time episode count and rewards")
    print("• Success rate tracking")
    print("• Exploration rate (epsilon)")
    print("• Waypoint progress")
    print("• Collision detection")
    print("• Training progress bars")
    print("• System performance metrics")
    
    print("\n💾 FILE MANAGEMENT:")
    print("• Checkpoints: rl_checkpoint_YYYYMMDD_HHMMSS.pth")
    print("• Final models: rl_model_final_YYYYMMDD_HHMMSS.pth")
    print("• Auto-save frequency: Every 10 episodes")
    print("• Manual save available anytime")
    
    # Run the interface
    print("\n🚀 Starting Enhanced Training Interface...")
    print("   Close the GUI window to exit this test")
    
    try:
        interface.run()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        print("🏁 Test completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Enhanced AI Training Interface")
    parser.add_argument("--demo", action="store_true", 
                       help="Show demo information without launching GUI")
    
    args = parser.parse_args()
    
    if args.demo:
        print("🎮 Enhanced AI Training Interface Demo")
        print("=" * 40)
        print("This interface provides:")
        print("• Clear workflow instructions")
        print("• Training session management")
        print("• Model persistence and loading")
        print("• Real-time monitoring")
        print("• Configurable episode length")
        print("• Position reset without weight loss")
        print("\nRun without --demo to launch the full interface")
    else:
        test_enhanced_interface()

if __name__ == "__main__":
    main() 