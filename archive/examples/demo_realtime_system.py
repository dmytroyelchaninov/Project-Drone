#!/usr/bin/env python3
"""
Real-time System Demonstration
Shows the capabilities of the real-time simulation system
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.control.keyboard_controller import KeyboardController
from drone_sim.control.rl_controller import RLController, Obstacle, Waypoint
from drone_sim.control.base_controller import ControllerState, ControllerReference
from drone_sim.utils.background_validator import BackgroundValidator
from drone_sim.utils.test_logger import TestLogger

def demo_manual_control():
    """Demonstrate manual keyboard control"""
    print("🎮 Manual Control Demo")
    print("=" * 40)
    
    controller = KeyboardController()
    print("✅ Keyboard controller created")
    
    # Show control info
    info = controller.get_control_info()
    print(f"Control mode: {info['mode']}")
    print(f"Emergency stop: {info['emergency_stop']}")
    
    # Simulate some control updates
    state = ControllerState(
        position=np.array([0.0, 0.0, -2.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    reference = ControllerReference()
    
    print("Simulating control updates...")
    for i in range(5):
        output = controller.update(reference, state, 0.01)
        print(f"  Step {i+1}: Thrust={output.thrust:.2f}N")
        time.sleep(0.1)
    
    controller.reset()
    print("✅ Manual control demo completed\n")

def demo_ai_control():
    """Demonstrate AI reinforcement learning control"""
    print("🤖 AI Control Demo")
    print("=" * 40)
    
    # Create environment
    obstacles = [
        Obstacle(position=np.array([5.0, 0.0, -2.0]), size=np.array([1.0, 1.0, 2.0]))
    ]
    waypoints = [
        Waypoint(position=np.array([10.0, 0.0, -2.0]), tolerance=0.5)
    ]
    
    controller = RLController()
    controller.set_obstacles(obstacles)
    controller.set_waypoints(waypoints)
    print("✅ RL controller created with environment")
    
    # Show learning stats
    stats = controller.get_learning_stats()
    print(f"Episodes: {stats['episode_count']}")
    print(f"Waypoints: {stats['total_waypoints']}")
    
    # Simulate some learning steps
    state = ControllerState(
        position=np.array([0.0, 0.0, -2.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    reference = ControllerReference()
    
    print("Simulating AI learning...")
    for i in range(10):
        output = controller.update(reference, state, 0.01)
        print(f"  Step {i+1}: Thrust={output.thrust:.2f}N, Reward={controller.total_reward:.2f}")
        
        # Simple state update for demo
        state.position[0] += 0.1  # Move forward
        time.sleep(0.05)
    
    controller.reset()
    print("✅ AI control demo completed\n")

def demo_background_validation():
    """Demonstrate background validation system"""
    print("🔍 Background Validation Demo")
    print("=" * 40)
    
    validator = BackgroundValidator()
    validator.start_background_validation()
    print("✅ Background validator started")
    
    # Submit test events
    print("Submitting test events...")
    for i in range(5):
        validator.submit_test_event(
            "Demo Test",
            "test_step",
            {
                'step': i,
                'position': [i * 1.0, 0.0, -2.0],
                'velocity': [1.0, 0.0, 0.0],
                'thrust': 14.7 + i * 0.1
            },
            {'demo': True}
        )
        time.sleep(0.1)
    
    # Get real-time stats
    time.sleep(0.5)  # Allow processing
    stats = validator.get_real_time_stats()
    print(f"Queue size: {stats['queue_size']}")
    print(f"Tests monitored: {stats['stats']['tests_monitored']}")
    
    validator.stop_background_validation()
    print("✅ Background validation demo completed\n")

def demo_logging_system():
    """Demonstrate logging system"""
    print("📝 Logging System Demo")
    print("=" * 40)
    
    logger = TestLogger("realtime_demo")
    logger.start_test("Demo Session")
    print("✅ Logger started")
    
    # Log some data
    for i in range(3):
        logger.log_step(f"Demo step {i+1}", {
            "step_number": i+1,
            "position": [i * 2.0, 0.0, -2.0],
            "velocity": [1.0, 0.0, 0.0],
            "timestamp": time.time()
        })
        time.sleep(0.1)
    
    logger.end_test("COMPLETED", {
        "total_steps": 3,
        "demo_complete": True
    })
    
    log_dir = logger.finalize_session()
    print(f"✅ Logs saved to: {log_dir}")
    print("✅ Logging demo completed\n")

def demo_integrated_system():
    """Demonstrate integrated real-time system"""
    print("🚀 Integrated System Demo")
    print("=" * 40)
    
    # Initialize all components
    keyboard_controller = KeyboardController()
    rl_controller = RLController()
    validator = BackgroundValidator()
    logger = TestLogger("integrated_demo")
    
    # Setup environment
    obstacles = [
        Obstacle(position=np.array([3.0, 0.0, -2.0]), size=np.array([1.0, 1.0, 2.0]))
    ]
    waypoints = [
        Waypoint(position=np.array([6.0, 0.0, -2.0]), tolerance=0.5)
    ]
    rl_controller.set_obstacles(obstacles)
    rl_controller.set_waypoints(waypoints)
    
    # Start background systems
    validator.start_background_validation()
    logger.start_test("Integrated Demo")
    
    print("✅ All systems initialized")
    
    # Simulation state
    state = ControllerState(
        position=np.array([0.0, 0.0, -2.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )
    
    reference = ControllerReference()
    
    print("Running integrated simulation for 3 seconds...")
    start_time = time.time()
    step = 0
    
    while time.time() - start_time < 3.0:
        # Use AI controller
        output = rl_controller.update(reference, state, 0.01)
        
        # Simple physics update
        state.position[0] += 0.02  # Move forward
        
        # Log data
        logger.log_step(f"Sim step {step}", {
            "position": state.position.tolist(),
            "velocity": state.velocity.tolist(),
            "thrust": output.thrust,
            "reward": rl_controller.total_reward
        })
        
        # Submit validation event
        validator.submit_test_event(
            "Integrated Demo",
            "simulation_step",
            {
                'position': state.position.tolist(),
                'velocity': state.velocity.tolist(),
                'thrust': output.thrust
            },
            {'step': step}
        )
        
        step += 1
        time.sleep(0.01)
    
    # Cleanup
    validator.stop_background_validation()
    logger.end_test("COMPLETED")
    log_dir = logger.finalize_session()
    
    print(f"✅ Integrated demo completed: {step} steps")
    print(f"✅ Logs saved to: {log_dir}")
    print()

def main():
    """Run all demonstrations"""
    print("🎯 Real-time Simulation System Demonstration")
    print("=" * 60)
    print("This demo shows the key capabilities of the real-time system:")
    print("- Manual keyboard control")
    print("- AI reinforcement learning")
    print("- Background physics validation")
    print("- Comprehensive logging")
    print("- Integrated system operation")
    print("=" * 60)
    print()
    
    try:
        # Run individual demos
        demo_manual_control()
        demo_ai_control()
        demo_background_validation()
        demo_logging_system()
        demo_integrated_system()
        
        print("🎉 All demonstrations completed successfully!")
        print()
        print("Next steps:")
        print("1. Try the console simulation: python examples/console_sim.py")
        print("2. Read the documentation: docs/realtime_simulation_guide.md")
        print("3. Explore the full interface: python examples/realtime_simulation.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 