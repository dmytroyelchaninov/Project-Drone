#!/usr/bin/env python3
"""
Real-time Drone Simulation
Main entry point for real-time simulation with manual and AI modes
"""

import sys
import os
import numpy as np
import time
import argparse
import json
import threading
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_sim.ui.realtime_interface import RealTimeInterface, SimulationParameters
from drone_sim.control.rl_controller import Obstacle, Waypoint
from drone_sim.utils.test_logger import TestLogger

# Check for tkinter availability
try:
    import tkinter
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

class ConsoleLogger:
    """Enhanced console logger for real-time simulation with detailed logging"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = time.time()
        self.step_count = 0
        self.last_status_time = 0
        self.last_ai_time = 0
        
        # Enhanced logging data
        self.performance_metrics = {
            'fps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'physics_violations': 0,
            'control_commands': 0,
            'ai_updates': 0
        }
        
        # Detailed state history for analysis
        self.detailed_history = {
            'timestamps': [],
            'positions': [],
            'velocities': [],
            'accelerations': [],
            'control_inputs': [],
            'ai_decisions': [],
            'system_states': []
        }
        
        if self.enabled:
            print("🔍 Enhanced console logging enabled")
            print("📊 Capturing: Status, Controls, AI, Performance, Physics, System Metrics")
            print("=" * 60)
    
    def log_status(self, position, velocity, mode_info):
        """Enhanced status logging with performance metrics"""
        if not self.enabled:
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.step_count += 1
        
        # Calculate FPS
        if current_time - self.last_status_time > 0:
            fps = 1.0 / (current_time - self.last_status_time) if self.last_status_time > 0 else 0
            self.performance_metrics['fps'].append(fps)
        
        # Get system metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            self.performance_metrics['cpu_usage'].append(cpu_percent)
            self.performance_metrics['memory_usage'].append(memory_percent)
        except ImportError:
            cpu_percent = 0
            memory_percent = 0
        
        # Store detailed history
        self.detailed_history['timestamps'].append(current_time)
        self.detailed_history['positions'].append(position.tolist() if hasattr(position, 'tolist') else list(position))
        self.detailed_history['velocities'].append(velocity.tolist() if hasattr(velocity, 'tolist') else list(velocity))
        
        # Calculate acceleration (if we have previous velocity)
        if len(self.detailed_history['velocities']) > 1:
            prev_vel = np.array(self.detailed_history['velocities'][-2])
            curr_vel = np.array(velocity)
            dt = current_time - self.detailed_history['timestamps'][-2]
            acceleration = (curr_vel - prev_vel) / dt if dt > 0 else np.zeros(3)
            self.detailed_history['accelerations'].append(acceleration.tolist())
        else:
            self.detailed_history['accelerations'].append([0, 0, 0])
        
        # Enhanced status display (throttled to avoid spam)
        if current_time - self.last_status_time >= 1.0:  # Every 1 second
            pos_str = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            vel_str = f"({velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f})"
            
            # Calculate speed
            speed = np.linalg.norm(velocity)
            
            # Get mode-specific info
            mode = mode_info.get('mode', 'unknown')
            thrust = mode_info.get('thrust', 0.0)
            
            print(f"📍 Step {self.step_count}: Pos={pos_str} Vel={vel_str} Speed={speed:.2f}m/s Mode={mode}")
            print(f"   ⚡ Thrust: {thrust:.2f}N | FPS: {fps:.1f} | CPU: {cpu_percent:.1f}% | Mem: {memory_percent:.1f}%")
            print(f"   📈 Metrics: {len(self.performance_metrics['fps'])} samples, Avg FPS: {np.mean(self.performance_metrics['fps'][-10:]):.1f}")
            
            # Physics validation status
            if hasattr(mode_info, 'physics_valid'):
                status = "✅" if mode_info.get('physics_valid', True) else "⚠️"
                print(f"   🔬 Physics: {status} | Violations: {self.performance_metrics['physics_violations']}")
            
            self.last_status_time = current_time
    
    def log_control(self, thrust, moment):
        """Enhanced control logging with input analysis"""
        if not self.enabled:
            return
        
        self.performance_metrics['control_commands'] += 1
        
        # Store detailed control history
        control_data = {
            'timestamp': time.time(),
            'thrust': float(thrust),
            'moment': moment.tolist() if hasattr(moment, 'tolist') else list(moment)
        }
        self.detailed_history['control_inputs'].append(control_data)
        
        # Calculate control magnitude
        moment_magnitude = np.linalg.norm(moment)
        
        print(f"🎮 Control: Thrust={thrust:.2f}N, Moment=({moment[0]:.2f}, {moment[1]:.2f}, {moment[2]:.2f})")
        print(f"   📊 Control Stats: Magnitude={moment_magnitude:.2f}, Total Commands={self.performance_metrics['control_commands']}")
    
    def log_ai(self, stats):
        """Enhanced AI logging with learning analytics"""
        if not self.enabled:
            return
        
        current_time = time.time()
        self.performance_metrics['ai_updates'] += 1
        
        # Store AI decision history
        ai_data = {
            'timestamp': current_time,
            'stats': stats.copy()
        }
        self.detailed_history['ai_decisions'].append(ai_data)
        
        # Enhanced AI display (throttled)
        if current_time - self.last_ai_time >= 2.0:  # Every 2 seconds
            episodes = stats.get('episodes', 0)
            reward = stats.get('reward', 0.0)
            success_rate = stats.get('success_rate', 0.0)
            learning_rate = stats.get('learning_rate', 0.0)
            exploration = stats.get('exploration', 0.0)
            
            print(f"🤖 AI Progress: Episodes={episodes}, Reward={reward:.2f}, Success={success_rate:.1f}%")
            
            # Advanced AI metrics
            if 'loss' in stats:
                print(f"   🧠 Learning: Loss={stats['loss']:.4f}, LR={learning_rate:.6f}, Explore={exploration:.2f}")
            
            # Performance trends
            if len(self.detailed_history['ai_decisions']) > 5:
                recent_rewards = [d['stats'].get('reward', 0) for d in self.detailed_history['ai_decisions'][-5:]]
                trend = "📈" if recent_rewards[-1] > recent_rewards[0] else "📉"
                print(f"   📊 Trend: {trend} Recent avg reward: {np.mean(recent_rewards):.2f}")
            
            self.last_ai_time = current_time
    
    def log_event(self, event_type, message):
        """Enhanced event logging with system context"""
        if not self.enabled:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        
        # Get system context for important events
        if event_type in ['START', 'STOP', 'ERROR', 'INIT']:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                system_context = f" [CPU: {cpu_percent:.1f}%, Mem: {memory_info.percent:.1f}%]"
            except ImportError:
                system_context = ""
        else:
            system_context = ""
        
        # Store system state for critical events
        if event_type in ['ERROR', 'START', 'STOP']:
            system_state = self._capture_system_state()
            self.detailed_history['system_states'].append({
                'timestamp': time.time(),
                'event_type': event_type,
                'message': message,
                'system_state': system_state
            })
        
        # Event type icons
        icons = {
            'START': '🚀',
            'STOP': '🛑',
            'ERROR': '❌',
            'WARNING': '⚠️',
            'INFO': 'ℹ️',
            'INIT': '⚙️',
            'SETUP': '🔧',
            'SHUTDOWN': '🔌'
        }
        
        icon = icons.get(event_type, '📝')
        print(f"[{timestamp}] {icon} {event_type}: {message}{system_context}")
    
    def _capture_system_state(self):
        """Capture detailed system state for analysis"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory': {
                    'percent': psutil.virtual_memory().percent,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used
                },
                'disk': {
                    'percent': psutil.disk_usage('/').percent,
                    'free': psutil.disk_usage('/').free
                },
                'process_count': len(psutil.pids()),
                'boot_time': psutil.boot_time()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.enabled or not self.performance_metrics['fps']:
            return {}
        
        return {
            'simulation_duration': time.time() - self.start_time,
            'total_steps': self.step_count,
            'fps_stats': {
                'mean': np.mean(self.performance_metrics['fps']),
                'min': np.min(self.performance_metrics['fps']),
                'max': np.max(self.performance_metrics['fps']),
                'std': np.std(self.performance_metrics['fps'])
            },
            'cpu_stats': {
                'mean': np.mean(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0,
                'max': np.max(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0
            },
            'memory_stats': {
                'mean': np.mean(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
                'max': np.max(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0
            },
            'activity_counts': {
                'control_commands': self.performance_metrics['control_commands'],
                'ai_updates': self.performance_metrics['ai_updates'],
                'physics_violations': self.performance_metrics['physics_violations']
            }
        }
    
    def save_detailed_log(self, filename=None):
        """Save detailed logging data to file for analysis"""
        if not self.enabled:
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_simulation_log_{timestamp}.json"
        
        log_data = {
            'session_info': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration': time.time() - self.start_time,
                'total_steps': self.step_count
            },
            'performance_summary': self.get_performance_summary(),
            'detailed_history': self.detailed_history,
            'performance_metrics': self.performance_metrics
        }
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            print(f"💾 Detailed log saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save detailed log: {e}")

def create_default_environment():
    """Create a default environment with obstacles and waypoints"""
    obstacles = [
        Obstacle(
            position=np.array([5.0, 0.0, -2.0]),
            size=np.array([1.0, 1.0, 2.0]),
            shape="box"
        ),
        Obstacle(
            position=np.array([10.0, 3.0, -1.5]),
            size=np.array([1.5, 1.5, 1.5]),
            shape="box"
        ),
        Obstacle(
            position=np.array([15.0, -2.0, -2.5]),
            size=np.array([2.0, 1.0, 3.0]),
            shape="box"
        )
    ]
    
    waypoints = [
        Waypoint(position=np.array([3.0, 0.0, -2.0]), tolerance=0.5),
        Waypoint(position=np.array([8.0, 2.0, -2.0]), tolerance=0.5),
        Waypoint(position=np.array([12.0, -1.0, -2.0]), tolerance=0.5),
        Waypoint(position=np.array([18.0, 0.0, -2.0]), tolerance=0.5),
        Waypoint(position=np.array([20.0, 0.0, -2.0]), tolerance=0.5)
    ]
    
    return obstacles, waypoints

def create_challenging_environment():
    """Create a challenging environment for AI training"""
    obstacles = []
    waypoints = []
    
    # Create a maze-like environment
    for i in range(5):
        for j in range(3):
            if np.random.random() > 0.3:  # 70% chance of obstacle
                obstacles.append(Obstacle(
                    position=np.array([i * 4.0 + 2.0, j * 3.0 - 3.0, -2.0 + np.random.uniform(-0.5, 0.5)]),
                    size=np.array([1.0 + np.random.uniform(0, 0.5), 1.0 + np.random.uniform(0, 0.5), 2.0]),
                    shape="box"
                ))
    
    # Create waypoints through the maze
    for i in range(6):
        waypoints.append(Waypoint(
            position=np.array([i * 3.5, np.random.uniform(-2, 2), -1.5 + np.random.uniform(-0.5, 0.5)]),
            tolerance=0.8
        ))
    
    return obstacles, waypoints

def load_environment_from_file(filename):
    """Load environment from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        obstacles = []
        for obs_data in data.get('obstacles', []):
            obstacles.append(Obstacle(
                position=np.array(obs_data['position']),
                size=np.array(obs_data['size']),
                shape=obs_data.get('shape', 'box')
            ))
        
        waypoints = []
        for wp_data in data.get('waypoints', []):
            waypoints.append(Waypoint(
                position=np.array(wp_data['position']),
                tolerance=wp_data.get('tolerance', 0.5)
            ))
        
        return obstacles, waypoints
        
    except Exception as e:
        print(f"❌ Failed to load environment from {filename}: {e}")
        return [], []

def save_environment_to_file(obstacles, waypoints, filename):
    """Save environment to JSON file"""
    try:
        data = {
            'obstacles': [
                {
                    'position': obs.position.tolist(),
                    'size': obs.size.tolist(),
                    'shape': obs.shape
                }
                for obs in obstacles
            ],
            'waypoints': [
                {
                    'position': wp.position.tolist(),
                    'tolerance': wp.tolerance
                }
                for wp in waypoints
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Environment saved to {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save environment to {filename}: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real-time Drone Simulation with Enhanced Logging")
    parser.add_argument("--mode", choices=["manual", "ai", "hybrid"], default="manual",
                       help="Simulation mode")
    parser.add_argument("--environment", choices=["default", "challenging", "custom"], default="default",
                       help="Environment preset")
    parser.add_argument("--mass", type=float, default=1.5, help="Drone mass in kg")
    parser.add_argument("--rtf", type=float, default=1.0, help="Real-time factor")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step in seconds")
    parser.add_argument("--log", action="store_true", default=True, help="Enable console logging (default: True)")
    parser.add_argument("--no-log", action="store_true", help="Disable console logging")
    parser.add_argument("--detailed-log", action="store_true", help="Save detailed log file at end")
    parser.add_argument("--log-interval", type=float, default=1.0, help="Status logging interval in seconds")
    
    args = parser.parse_args()
    
    # Handle logging flags
    console_logging = args.log and not args.no_log
    
    print("🚀 Starting Real-time Drone Simulation")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Environment: {args.environment}")
    print(f"Mass: {args.mass} kg")
    print(f"Real-time factor: {args.rtf}x")
    print(f"Time step: {args.dt} s")
    print(f"Console logging: {'enabled' if console_logging else 'disabled'}")
    print(f"Detailed logging: {'enabled' if args.detailed_log else 'disabled'}")
    if TKINTER_AVAILABLE:
        print(f"GUI mode: enabled")
    else:
        print(f"Console mode: enabled")
    print("=" * 50)
    
    # Initialize enhanced console logger
    console_logger = ConsoleLogger(enabled=console_logging)
    
    # Create simulation parameters
    sim_params = SimulationParameters(
        mass=args.mass,
        dt=args.dt,
        real_time_factor=args.rtf,
        physics_validation=True
    )
    
    # Create environment
    if args.environment == 'default':
        obstacles, waypoints = create_default_environment()
    elif args.environment == 'challenging':
        obstacles, waypoints = create_challenging_environment()
    else:  # custom or empty
        obstacles, waypoints = [], []
    
    print(f"Environment: {len(obstacles)} obstacles, {len(waypoints)} waypoints")
    console_logger.log_event("SETUP", f"Environment loaded: {len(obstacles)} obstacles, {len(waypoints)} waypoints")
    
    try:
        
        # Create and configure interface
        interface = RealTimeInterface()
        interface.sim_params = sim_params
        interface.obstacles = obstacles
        interface.waypoints = waypoints
        
        # Set simulation mode
        from drone_sim.ui.realtime_interface import SimulationMode
        interface.simulation_mode = SimulationMode(args.mode)
        
        console_logger.log_event("INIT", f"Interface initialized in {args.mode} mode")
        
        # Enhanced callback setup with error handling
        def on_status_update(*args, **kwargs):
            try:
                if len(args) == 3:
                    # New format: position, velocity, mode_info
                    position, velocity, mode_info = args
                    console_logger.log_status(position, velocity, mode_info)
                elif len(args) == 1:
                    # Old format: just message - convert to event
                    message = args[0]
                    console_logger.log_event("STATUS", message)
                else:
                    # Unknown format
                    console_logger.log_event("STATUS", f"Unknown status format: {args}")
            except Exception as e:
                print(f"Status callback error: {e}")
        
        def on_control_update(thrust, moment):
            try:
                console_logger.log_control(thrust, moment)
            except Exception as e:
                print(f"Control callback error: {e}")
        
        def on_ai_update(stats):
            try:
                console_logger.log_ai(stats)
            except Exception as e:
                print(f"AI callback error: {e}")
        
        def on_event_update(event_type, message):
            try:
                console_logger.log_event(event_type, message)
            except Exception as e:
                print(f"Event callback error: {e}")
        
        # Set callbacks
        interface.set_status_callback(on_status_update)
        interface.set_control_callback(on_control_update)
        interface.set_ai_callback(on_ai_update)
        interface.set_event_callback(on_event_update)
        
        console_logger.log_event("START", "Starting simulation interface")
        
        # Start the interface
        interface.run()
        
    except KeyboardInterrupt:
        console_logger.log_event("STOP", "Simulation interrupted by user")
    except Exception as e:
        console_logger.log_event("ERROR", f"Simulation error: {e}")
        raise
    finally:
        # Save detailed log if requested
        if args.detailed_log:
            console_logger.save_detailed_log()
        
        # Print performance summary
        if console_logging:
            summary = console_logger.get_performance_summary()
            if summary:
                print("\n" + "=" * 50)
                print("📊 SIMULATION PERFORMANCE SUMMARY")
                print("=" * 50)
                print(f"Duration: {summary['simulation_duration']:.1f}s")
                print(f"Total Steps: {summary['total_steps']}")
                print(f"Average FPS: {summary['fps_stats']['mean']:.1f}")
                print(f"CPU Usage: {summary['cpu_stats']['mean']:.1f}% (max: {summary['cpu_stats']['max']:.1f}%)")
                print(f"Memory Usage: {summary['memory_stats']['mean']:.1f}% (max: {summary['memory_stats']['max']:.1f}%)")
                print(f"Control Commands: {summary['activity_counts']['control_commands']}")
                print(f"AI Updates: {summary['activity_counts']['ai_updates']}")
                print("=" * 50)
        
        console_logger.log_event("SHUTDOWN", "Simulation shutdown complete")

if __name__ == "__main__":
    main() 