#!/usr/bin/env python3
"""
Console Real-time Simulation
A console-based real-time simulation with manual and AI modes
"""

import sys
import os
import numpy as np
import time
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.core.simulator import Simulator, SimulationConfig
from drone_sim.core.state_manager import StateManager, DroneState
from drone_sim.physics.rigid_body import RigidBody, RigidBodyConfig
from drone_sim.physics.environment import Environment, EnvironmentConfig
from drone_sim.control.keyboard_controller import KeyboardController, KeyboardConfig, ControlMode
from drone_sim.control.rl_controller import RLController, RLConfig, RLMode, Obstacle, Waypoint
from drone_sim.control.pid_controller import PIDController
from drone_sim.control.base_controller import ControllerReference, ControllerState
from drone_sim.utils.background_validator import BackgroundValidator
from drone_sim.utils.test_logger import TestLogger

class ConsoleSimulation:
    """Console-based real-time simulation"""
    
    def __init__(self):
        self.mode = "manual"  # "manual", "ai", "hybrid"
        self.running = False
        self.paused = False
        
        # Simulation components
        self.simulator = None
        self.keyboard_controller = None
        self.rl_controller = None
        self.pid_controller = None
        self.background_validator = None
        self.test_logger = None
        
        # Simulation parameters
        self.dt = 0.002
        self.real_time_factor = 1.0
        self.mass = 1.5
        
        # Environment
        self.obstacles = []
        self.waypoints = []
        
        # Data tracking
        self.simulation_data = {
            'time': [],
            'position': [],
            'velocity': [],
            'control_thrust': [],
            'rewards': []
        }
        
        # Simulation thread
        self.sim_thread = None
        
    def setup_environment(self, env_type="default"):
        """Setup simulation environment"""
        if env_type == "default":
            self.obstacles = [
                Obstacle(
                    position=np.array([5.0, 0.0, -2.0]),
                    size=np.array([1.0, 1.0, 2.0])
                ),
                Obstacle(
                    position=np.array([10.0, 3.0, -1.5]),
                    size=np.array([1.5, 1.5, 1.5])
                )
            ]
            
            self.waypoints = [
                Waypoint(position=np.array([3.0, 0.0, -2.0]), tolerance=0.5),
                Waypoint(position=np.array([8.0, 2.0, -1.0]), tolerance=0.5),
                Waypoint(position=np.array([15.0, 0.0, -2.0]), tolerance=0.5)
            ]
        elif env_type == "challenging":
            # Create challenging environment
            self.obstacles = []
            for i in range(6):
                for j in range(2):
                    if np.random.random() > 0.4:
                        self.obstacles.append(Obstacle(
                            position=np.array([i * 3.0 + 2.0, j * 4.0 - 2.0, -2.0]),
                            size=np.array([1.0, 1.0, 2.0])
                        ))
            
            self.waypoints = []
            for i in range(5):
                self.waypoints.append(Waypoint(
                    position=np.array([i * 3.5 + 1.0, np.random.uniform(-1, 1), -1.5]),
                    tolerance=0.8
                ))
        
        print(f"Environment setup: {len(self.obstacles)} obstacles, {len(self.waypoints)} waypoints")
    
    def initialize_simulation(self):
        """Initialize simulation components"""
        print("🚀 Initializing simulation components...")
        
        # Create simulation config
        sim_config = SimulationConfig(
            dt=self.dt,
            real_time_factor=self.real_time_factor,
            physics_validation=True
        )
        
        # Create simulator
        self.simulator = Simulator(sim_config)
        
        # Create physics components
        inertia_matrix = np.diag([0.02, 0.02, 0.04])
        rigid_body_config = RigidBodyConfig(
            mass=self.mass,
            inertia=inertia_matrix
        )
        rigid_body = RigidBody(rigid_body_config)
        
        env_config = EnvironmentConfig(
            gravity_magnitude=9.81,
            air_density=1.225
        )
        environment = Environment(env_config)
        
        # Register components
        self.simulator.register_physics_engine(rigid_body)
        self.simulator.register_environment(environment)
        
        # Create controllers
        if self.mode in ["manual", "hybrid"]:
            self.keyboard_controller = KeyboardController()
            print("   ✅ Keyboard controller created")
        
        if self.mode in ["ai", "hybrid"]:
            self.rl_controller = RLController()
            self.rl_controller.set_obstacles(self.obstacles)
            self.rl_controller.set_waypoints(self.waypoints)
            print("   ✅ RL controller created")
        
        self.pid_controller = PIDController()
        print("   ✅ PID controller created")
        
        # Set initial state
        initial_state = DroneState()
        initial_state.position = np.array([0.0, 0.0, -2.0])
        initial_state.velocity = np.array([0.0, 0.0, 0.0])
        self.simulator.state_manager.set_state(initial_state)
        
        # Initialize background validation
        self.background_validator = BackgroundValidator()
        self.background_validator.start_background_validation()
        print("   ✅ Background validation started")
        
        # Initialize logging
        self.test_logger = TestLogger("console_realtime_sim")
        self.test_logger.start_test("Console Real-time Simulation")
        print("   ✅ Logging initialized")
        
        print("✅ Simulation initialized successfully")
    
    def start_simulation(self):
        """Start the simulation"""
        if self.running:
            print("⚠️ Simulation already running")
            return
        
        self.running = True
        self.paused = False
        
        # Start keyboard monitoring if needed
        if self.keyboard_controller:
            self.keyboard_controller.start_input_monitoring()
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        print("🚀 Simulation started")
        self._print_controls()
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        if self.keyboard_controller:
            self.keyboard_controller.stop_input_monitoring()
        
        if self.background_validator:
            self.background_validator.stop_background_validation()
        
        if self.test_logger:
            self.test_logger.end_test("COMPLETED", {
                "simulation_data": self.simulation_data,
                "mode": self.mode,
                "environment": {
                    "obstacles": len(self.obstacles),
                    "waypoints": len(self.waypoints)
                }
            })
        
        print("🛑 Simulation stopped")
    
    def pause_simulation(self):
        """Pause/resume simulation"""
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        print(f"⏸️ Simulation {status}")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        step_count = 0
        last_status_time = time.time()
        
        print("🔄 Simulation loop started")
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Get current state
                current_state = self.simulator.state_manager.get_state()
                
                # Convert to controller state
                controller_state = ControllerState(
                    position=current_state.position,
                    quaternion=current_state.quaternion,
                    velocity=current_state.velocity,
                    angular_velocity=current_state.angular_velocity
                )
                
                reference = ControllerReference()
                
                # Determine active controller
                control_output = None
                active_controller = "none"
                
                if self.mode == "manual" and self.keyboard_controller:
                    control_output = self.keyboard_controller.update(reference, controller_state, self.dt)
                    active_controller = "keyboard"
                elif self.mode == "ai" and self.rl_controller:
                    control_output = self.rl_controller.update(reference, controller_state, self.dt)
                    active_controller = "rl"
                elif self.mode == "hybrid":
                    # Use keyboard if active, otherwise AI
                    if self.keyboard_controller and self.keyboard_controller.key_states:
                        control_output = self.keyboard_controller.update(reference, controller_state, self.dt)
                        active_controller = "keyboard"
                    elif self.rl_controller:
                        control_output = self.rl_controller.update(reference, controller_state, self.dt)
                        active_controller = "rl"
                
                # Fallback to PID
                if control_output is None:
                    control_output = self.pid_controller.update(reference, controller_state, self.dt)
                    active_controller = "pid"
                
                # Simple physics update (for demonstration)
                # In a full implementation, this would be handled by the physics engine
                self._simple_physics_update(current_state, control_output)
                
                # Log data
                current_time = step_count * self.dt
                self.simulation_data['time'].append(current_time)
                self.simulation_data['position'].append(current_state.position.copy())
                self.simulation_data['velocity'].append(current_state.velocity.copy())
                self.simulation_data['control_thrust'].append(control_output.thrust if control_output else 0.0)
                
                if self.rl_controller and hasattr(self.rl_controller, 'total_reward'):
                    self.simulation_data['rewards'].append(self.rl_controller.total_reward)
                
                # Background validation
                if self.background_validator:
                    self.background_validator.submit_test_event(
                        "Console Simulation",
                        "simulation_step",
                        {
                            'time': current_time,
                            'position': current_state.position.tolist(),
                            'velocity': current_state.velocity.tolist(),
                            'control_thrust': control_output.thrust if control_output else 0.0,
                            'active_controller': active_controller
                        },
                        {'step': step_count}
                    )
                
                # Periodic status update
                if time.time() - last_status_time > 2.0:
                    self._print_status(current_time, current_state, active_controller)
                    last_status_time = time.time()
                
                step_count += 1
                time.sleep(self.dt / self.real_time_factor)
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                break
        
        print("🔄 Simulation loop ended")
    
    def _simple_physics_update(self, state, control_output):
        """Simple physics update for demonstration"""
        if control_output:
            # Simple vertical control
            thrust_acceleration = (control_output.thrust - 9.81 * self.mass) / self.mass
            state.velocity[2] += thrust_acceleration * self.dt
            
            # Simple horizontal control from moments (simplified)
            if hasattr(control_output, 'moment') and control_output.moment is not None:
                state.velocity[0] += control_output.moment[1] * self.dt * 0.1  # Pitch to forward
                state.velocity[1] -= control_output.moment[0] * self.dt * 0.1  # Roll to right
        
        # Update position
        state.position += state.velocity * self.dt
        
        # Apply drag
        state.velocity *= 0.99
    
    def _print_status(self, time_val, state, controller):
        """Print simulation status"""
        pos = state.position
        vel = state.velocity
        
        print(f"\r🔄 T:{time_val:.1f}s | Pos:({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) | "
              f"Vel:({vel[0]:.1f},{vel[1]:.1f},{vel[2]:.1f}) | Ctrl:{controller}", end="")
        
        # Check waypoint progress
        if self.rl_controller and self.waypoints:
            wp_idx = self.rl_controller.current_waypoint_idx
            total_wp = len(self.waypoints)
            if wp_idx < total_wp:
                wp_pos = self.waypoints[wp_idx].position
                dist = np.linalg.norm(pos - wp_pos)
                print(f" | WP:{wp_idx+1}/{total_wp} Dist:{dist:.1f}m", end="")
    
    def _print_controls(self):
        """Print control instructions"""
        print("\n" + "="*60)
        print("🎮 CONTROLS:")
        if self.mode in ["manual", "hybrid"]:
            print("  WASD: Move horizontally")
            print("  Space/C: Up/Down")
            print("  QE: Yaw left/right")
            print("  H: Hover, L: Land, ESC: Emergency stop")
        
        print("\n📋 COMMANDS:")
        print("  'p' - Pause/Resume")
        print("  's' - Stop simulation")
        print("  'm <mode>' - Change mode (manual/ai/hybrid)")
        print("  'status' - Show detailed status")
        print("  'save' - Save simulation data")
        print("  'quit' - Exit")
        print("="*60)
    
    def change_mode(self, new_mode):
        """Change simulation mode"""
        if new_mode not in ["manual", "ai", "hybrid"]:
            print(f"❌ Invalid mode: {new_mode}")
            return
        
        old_mode = self.mode
        self.mode = new_mode
        print(f"🎮 Mode changed: {old_mode} → {new_mode}")
        
        # Reinitialize controllers if needed
        if new_mode in ["manual", "hybrid"] and not self.keyboard_controller:
            self.keyboard_controller = KeyboardController()
            self.keyboard_controller.start_input_monitoring()
        
        if new_mode in ["ai", "hybrid"] and not self.rl_controller:
            self.rl_controller = RLController()
            self.rl_controller.set_obstacles(self.obstacles)
            self.rl_controller.set_waypoints(self.waypoints)
    
    def show_status(self):
        """Show detailed status"""
        print("\n" + "="*60)
        print("📊 SIMULATION STATUS")
        print("="*60)
        
        if self.simulator:
            state = self.simulator.state_manager.get_state()
            print(f"Position: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}) m")
            print(f"Velocity: ({state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}) m/s")
        
        print(f"Mode: {self.mode}")
        print(f"Running: {self.running}")
        print(f"Paused: {self.paused}")
        print(f"Environment: {len(self.obstacles)} obstacles, {len(self.waypoints)} waypoints")
        
        if self.rl_controller:
            stats = self.rl_controller.get_learning_stats()
            print(f"RL Stats: Episode {stats['episode_count']}, Reward {stats['average_reward']:.2f}")
        
        if self.background_validator:
            val_stats = self.background_validator.get_real_time_stats()
            print(f"Validation: {val_stats['stats']['tests_monitored']} tests, "
                  f"{val_stats['stats']['anomalies_detected']} anomalies")
        
        print("="*60)
    
    def save_data(self, filename=None):
        """Save simulation data"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.json"
        
        data = {
            'metadata': {
                'mode': self.mode,
                'dt': self.dt,
                'real_time_factor': self.real_time_factor,
                'mass': self.mass,
                'obstacles': len(self.obstacles),
                'waypoints': len(self.waypoints),
                'timestamp': time.time()
            },
            'simulation_data': {
                'time': self.simulation_data['time'],
                'position': [pos.tolist() for pos in self.simulation_data['position']],
                'velocity': [vel.tolist() for vel in self.simulation_data['velocity']],
                'control_thrust': self.simulation_data['control_thrust'],
                'rewards': self.simulation_data['rewards']
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Data saved to {filename}")
    
    def run_console_interface(self):
        """Run the console interface"""
        print("🚀 Console Real-time Drone Simulation")
        print("Type 'help' for commands")
        
        while True:
            try:
                command = input("\nsim> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "help":
                    self._print_controls()
                elif command == "start":
                    self.start_simulation()
                elif command == "stop" or command == "s":
                    self.stop_simulation()
                elif command == "pause" or command == "p":
                    self.pause_simulation()
                elif command == "status":
                    self.show_status()
                elif command == "save":
                    self.save_data()
                elif command.startswith("m "):
                    mode = command.split()[1]
                    self.change_mode(mode)
                elif command.startswith("env "):
                    env_type = command.split()[1]
                    self.setup_environment(env_type)
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        self.stop_simulation()
        print("\n👋 Goodbye!")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Console Real-time Drone Simulation')
    parser.add_argument('--mode', choices=['manual', 'ai', 'hybrid'], default='manual',
                       help='Simulation mode (default: manual)')
    parser.add_argument('--environment', choices=['default', 'challenging', 'empty'], default='default',
                       help='Environment preset (default: default)')
    parser.add_argument('--mass', type=float, default=1.5, help='Drone mass in kg')
    parser.add_argument('--rtf', type=float, default=1.0, help='Real-time factor')
    parser.add_argument('--dt', type=float, default=0.002, help='Time step in seconds')
    parser.add_argument('--auto-start', action='store_true', help='Start simulation automatically')
    
    args = parser.parse_args()
    
    # Create simulation
    sim = ConsoleSimulation()
    sim.mode = args.mode
    sim.dt = args.dt
    sim.real_time_factor = args.rtf
    sim.mass = args.mass
    
    # Setup environment
    sim.setup_environment(args.environment)
    
    # Initialize simulation
    sim.initialize_simulation()
    
    # Auto-start if requested
    if args.auto_start:
        sim.start_simulation()
    
    # Run console interface
    sim.run_console_interface()

if __name__ == "__main__":
    main() 