#!/usr/bin/env python3
"""
Console Real-time Simulation
A simple console-based real-time simulation
"""

import sys
import numpy as np
import time
import threading
import platform
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.control.keyboard_controller import KeyboardController
from drone_sim.control.rl_controller import RLController, Obstacle, Waypoint
from drone_sim.control.base_controller import ControllerState, ControllerReference
from drone_sim.utils.background_validator import BackgroundValidator
from drone_sim.utils.test_logger import TestLogger

class ConsoleKeyboardController:
    """A console-friendly keyboard controller that doesn't use pygame"""
    
    def __init__(self):
        self.active_keys = set()
        self.running = False
        self.input_thread = None
        
    def start_input_monitoring(self):
        """Start monitoring keyboard input"""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        print("🎮 Console keyboard input started")
        print("Commands: w/a/s/d (movement), space/c (up/down), q/e (yaw), h (hover), l (land), x (stop)")
    
    def stop_input_monitoring(self):
        """Stop monitoring keyboard input"""
        self.running = False
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
        print("🎮 Console keyboard input stopped")
    
    def _input_loop(self):
        """Input loop for console commands"""
        while self.running:
            try:
                cmd = input(">> ").strip().lower()
                if cmd:
                    self._process_command(cmd)
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
    
    def _process_command(self, cmd):
        """Process a single command"""
        if cmd == 'x' or cmd == 'stop':
            self.running = False
        elif cmd in ['w', 'a', 's', 'd', 'space', 'c', 'q', 'e', 'h', 'l']:
            self.active_keys.add(cmd)
            # Auto-remove after short duration
            threading.Timer(0.5, lambda: self.active_keys.discard(cmd)).start()
        elif cmd == 'help':
            print("Commands: w(forward) a(left) s(back) d(right) space(up) c(down) q(yaw-left) e(yaw-right) h(hover) l(land) x(stop)")
    
    def get_control_output(self, dt=0.01):
        """Get control output based on active keys"""
        # Simple control mapping
        thrust = 14.7  # Hover thrust
        moment = np.zeros(3)
        
        if 'w' in self.active_keys:
            moment[1] += 2.0  # Forward
        if 's' in self.active_keys:
            moment[1] -= 2.0  # Backward
        if 'a' in self.active_keys:
            moment[0] -= 2.0  # Left
        if 'd' in self.active_keys:
            moment[0] += 2.0  # Right
        if 'space' in self.active_keys:
            thrust += 5.0  # Up
        if 'c' in self.active_keys:
            thrust -= 5.0  # Down
        if 'q' in self.active_keys:
            moment[2] += 1.0  # Yaw left
        if 'e' in self.active_keys:
            moment[2] -= 1.0  # Yaw right
        if 'h' in self.active_keys:
            # Hover - reset to neutral
            thrust = 14.7
            moment = np.zeros(3)
        if 'l' in self.active_keys:
            # Land - reduce thrust
            thrust = 10.0
        
        from drone_sim.control.base_controller import ControllerOutput
        return ControllerOutput(thrust=max(0.1, thrust), moment=moment)

class SimpleConsoleSimulation:
    def __init__(self, mode="manual"):
        self.mode = mode
        self.running = False
        self.dt = 0.01
        
        # Controllers
        self.keyboard_controller = None
        self.rl_controller = None
        
        # State
        self.position = np.array([0.0, 0.0, -2.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Environment
        self.obstacles = [
            Obstacle(position=np.array([5.0, 0.0, -2.0]), size=np.array([1.0, 1.0, 2.0]))
        ]
        self.waypoints = [
            Waypoint(position=np.array([10.0, 0.0, -2.0]), tolerance=0.5)
        ]
        
        self._initialize_controllers()
    
    def _initialize_controllers(self):
        if self.mode in ["manual", "hybrid"]:
            # Use console-friendly controller instead of pygame
            self.keyboard_controller = ConsoleKeyboardController()
            print("✅ Console keyboard controller initialized")
        
        if self.mode in ["ai", "hybrid"]:
            self.rl_controller = RLController()
            self.rl_controller.set_obstacles(self.obstacles)
            self.rl_controller.set_waypoints(self.waypoints)
            print("✅ RL controller initialized")
    
    def start(self):
        self.running = True
        if self.keyboard_controller:
            self.keyboard_controller.start_input_monitoring()
        
        print("🚀 Simulation started")
        print("Enter commands in the input prompt to control the drone")
        
        # Start simulation loop
        threading.Thread(target=self._simulation_loop, daemon=True).start()
    
    def stop(self):
        self.running = False
        if self.keyboard_controller:
            self.keyboard_controller.stop_input_monitoring()
        print("🛑 Simulation stopped")
    
    def _simulation_loop(self):
        step = 0
        last_print = time.time()
        
        while self.running:
            # Create controller state
            state = ControllerState(
                position=self.position,
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                velocity=self.velocity,
                angular_velocity=np.zeros(3)
            )
            
            reference = ControllerReference()
            
            # Get control output
            if self.mode == "manual" and self.keyboard_controller:
                output = self.keyboard_controller.get_control_output(self.dt)
            elif self.mode == "ai" and self.rl_controller:
                output = self.rl_controller.update(reference, state, self.dt)
            else:
                from drone_sim.control.base_controller import ControllerOutput
                output = ControllerOutput(thrust=14.7, moment=np.zeros(3))
            
            # Simple physics update
            if output:
                # Vertical control
                thrust_accel = (output.thrust - 14.7) / 1.5  # Simplified
                self.velocity[2] += thrust_accel * self.dt
                
                # Horizontal control (simplified)
                if hasattr(output, 'moment') and output.moment is not None:
                    self.velocity[0] += output.moment[1] * self.dt * 0.1
                    self.velocity[1] -= output.moment[0] * self.dt * 0.1
            
            # Update position
            self.position += self.velocity * self.dt
            
            # Apply drag
            self.velocity *= 0.99
            
            # Print status every 2 seconds
            if time.time() - last_print > 2.0:
                print(f"📍 Pos: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}) "
                      f"Vel: ({self.velocity[0]:.1f}, {self.velocity[1]:.1f}, {self.velocity[2]:.1f})")
                if self.mode == "ai" and self.rl_controller:
                    stats = self.rl_controller.get_learning_stats()
                    print(f"🤖 Episodes: {stats['episode_count']}, Reward: {self.rl_controller.total_reward:.1f}")
                last_print = time.time()
            
            step += 1
            time.sleep(self.dt)
    
    def run_console(self):
        print("🎮 Console Real-time Simulation")
        print("=" * 50)
        print("Available commands:")
        print("  start       - Start the simulation")
        print("  stop        - Stop the simulation")
        print("  mode <type> - Change mode (manual/ai/hybrid)")
        print("  status      - Show current status")
        print("  quit        - Exit the simulation")
        print("=" * 50)
        
        while True:
            try:
                cmd = input("sim> ").strip().lower()
                
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "start":
                    if not self.running:
                        self.start()
                    else:
                        print("Simulation already running")
                elif cmd == "stop":
                    if self.running:
                        self.stop()
                    else:
                        print("Simulation not running")
                elif cmd.startswith("mode "):
                    new_mode = cmd.split()[1]
                    if new_mode in ["manual", "ai", "hybrid"]:
                        was_running = self.running
                        if was_running:
                            self.stop()
                            time.sleep(0.5)
                        self.mode = new_mode
                        self._initialize_controllers()
                        print(f"✅ Mode changed to: {new_mode}")
                        if was_running:
                            self.start()
                    else:
                        print("❌ Invalid mode. Use: manual, ai, or hybrid")
                elif cmd == "status":
                    print(f"Mode: {self.mode}")
                    print(f"Running: {self.running}")
                    print(f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")
                    print(f"Velocity: ({self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f})")
                elif cmd == "help":
                    print("Commands: start, stop, mode <manual|ai|hybrid>, status, quit")
                elif cmd == "":
                    continue
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
            
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting...")
                break
        
        if self.running:
            self.stop()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Console Real-time Drone Simulation")
    parser.add_argument('--mode', choices=['manual', 'ai', 'hybrid'], default='manual',
                       help='Initial control mode')
    args = parser.parse_args()
    
    print("🚁 Console Drone Simulation")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Initial mode: {args.mode}")
    print()
    
    sim = SimpleConsoleSimulation(args.mode)
    sim.run_console()

if __name__ == "__main__":
    main() 