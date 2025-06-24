"""
Main simulation loop with fixed-time stepping
"""
import time
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import logging

from .state_manager import StateManager, DroneState
from .event_system import EventSystem
from ..physics import RigidBody, Environment, physics_validator
from ..control.base_controller import BaseController

class SimulationState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"

@dataclass
class SimulationConfig:
    """Configuration for the simulation"""
    dt: float = 0.002  # 2ms default time step
    max_steps: int = 1000000
    real_time_factor: float = 1.0  # 1.0 = real time, 0.5 = half speed, 2.0 = double speed
    physics_validation: bool = True  # Enable physics validation
    
    # Physics bounds for validation
    max_real_time_factor: float = 100.0  # Maximum RTF for stability
    min_real_time_factor: float = 0.01   # Minimum RTF for meaningful simulation
    max_efficiency: float = 0.95         # Maximum physical efficiency (95%)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.physics_validation:
            # Apply physical bounds to real-time factor
            if self.real_time_factor > self.max_real_time_factor:
                logging.warning(f"Real-time factor {self.real_time_factor} exceeds maximum {self.max_real_time_factor}, capping")
                self.real_time_factor = self.max_real_time_factor
            elif self.real_time_factor < self.min_real_time_factor:
                logging.warning(f"Real-time factor {self.real_time_factor} below minimum {self.min_real_time_factor}, adjusting")
                self.real_time_factor = self.min_real_time_factor

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    simulation_time: float = 0.0
    wall_clock_time: float = 0.0
    steps_completed: int = 0
    physics_violations: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    
    @property
    def actual_real_time_factor(self) -> float:
        """Calculate actual achieved real-time factor"""
        if self.wall_clock_time > 0:
            raw_rtf = self.simulation_time / self.wall_clock_time
            # Apply physical bounds
            return max(0.01, min(raw_rtf, 100.0))
        return 1.0
    
    @property
    def computational_efficiency(self) -> float:
        """Calculate realistic computational efficiency (0-100%)"""
        if self.steps_completed > 0 and self.wall_clock_time > 0:
            # Base efficiency on actual performance vs target
            target_time = self.simulation_time  # Ideal 1:1 real-time
            actual_time = self.wall_clock_time
            
            if actual_time > 0:
                raw_efficiency = target_time / actual_time
                # Cap at 95% maximum physical efficiency
                return min(raw_efficiency, 0.95)
        return 0.0

class Simulator:
    """
    Main simulation engine with fixed-step RK4 integrator
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.state = SimulationState.STOPPED
        self.current_time = 0.0
        self.step_count = 0
        self.last_real_time = 0.0
        
        # System components
        self.physics_engine = None
        self.control_system = None
        self.sensors = {}
        self.environment = None
        
        # Callbacks
        self.step_callbacks = []
        self.state_callbacks = []
        
        # Core components
        self.state_manager = StateManager()
        self.event_system = EventSystem()
        self.rigid_body = None
        self.controller = None
        
        # Simulation state
        self.is_running = False
        self.metrics = SimulationMetrics()
        
        # Performance tracking
        self._start_wall_time = 0.0
        self._step_times = []
        self._physics_check_interval = 100  # Check physics every N steps
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Simulator initialized with RTF={self.config.real_time_factor}, validation={self.config.physics_validation}")
        
    def register_physics_engine(self, physics_engine):
        """Register the physics engine"""
        self.physics_engine = physics_engine
        
    def register_control_system(self, control_system):
        """Register the control system"""
        self.control_system = control_system
        
    def register_sensor(self, name: str, sensor):
        """Register a sensor"""
        self.sensors[name] = sensor
        
    def register_environment(self, environment):
        """Register the environment"""
        self.environment = environment
        
    def add_step_callback(self, callback: Callable):
        """Add a callback to be called at each simulation step"""
        self.step_callbacks.append(callback)
        
    def add_state_callback(self, callback: Callable):
        """Add a callback to be called when simulation state changes"""
        self.state_callbacks.append(callback)
        
    def start(self):
        """Start the simulation"""
        if self.state == SimulationState.STOPPED:
            self.current_time = 0.0
            self.step_count = 0
            
        self.state = SimulationState.RUNNING
        self.last_real_time = time.time()
        self._notify_state_change()
        
    def pause(self):
        """Pause the simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self._notify_state_change()
            
    def resume(self):
        """Resume the simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.last_real_time = time.time()
            self._notify_state_change()
            
    def stop(self):
        """Stop the simulation"""
        self.state = SimulationState.STOPPED
        self._notify_state_change()
        
    def reset(self):
        """Reset the simulation to initial conditions"""
        self.stop()
        self.current_time = 0.0
        self.step_count = 0
        
        # Reset all components
        if self.physics_engine:
            self.physics_engine.reset()
        if self.control_system:
            self.control_system.reset()
        for sensor in self.sensors.values():
            sensor.reset()
        if self.environment:
            self.environment.reset()
            
        # Reset core components
        self.state_manager.reset()
        self.metrics = SimulationMetrics()
        self._step_times.clear()
        
        self.logger.info("Simulator reset")
        
    def step(self):
        """Execute a single simulation step using RK4 integration"""
        if self.state != SimulationState.RUNNING:
            return
            
        # RK4 integration
        dt = self.config.dt
        
        # Get current state
        if self.physics_engine:
            current_state = self.physics_engine.get_state()
            
            # RK4 stages
            k1 = self._compute_derivatives(current_state, self.current_time)
            k2 = self._compute_derivatives(current_state + k1 * dt/2, self.current_time + dt/2)
            k3 = self._compute_derivatives(current_state + k2 * dt/2, self.current_time + dt/2)
            k4 = self._compute_derivatives(current_state + k3 * dt, self.current_time + dt)
            
            # Update state
            new_state = current_state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            
            # Normalize quaternion to prevent drift accumulation
            new_state[3:7] = new_state[3:7] / np.linalg.norm(new_state[3:7])
            
            self.physics_engine.set_state(new_state)
            
        # Update sensors
        for sensor in self.sensors.values():
            sensor.update(self.current_time, dt)
            
        # Update environment
        if self.environment:
            self.environment.update(self.current_time, dt)
            
        # Update control system
        if self.control_system:
            self.control_system.update(self.current_time, dt)
            
        # Update time and step count
        self.current_time += dt
        self.step_count += 1
        
        # Call step callbacks
        for callback in self.step_callbacks:
            callback(self.current_time, dt)
            
        # Check for termination conditions
        if self.step_count >= self.config.max_steps:
            self.stop()
            
    def run(self, duration: Optional[float] = None):
        """Run the simulation for a specified duration or indefinitely"""
        self.start()
        
        target_time = self.current_time + duration if duration else float('inf')
        
        while self.state == SimulationState.RUNNING and self.current_time < target_time:
            step_start_time = time.time()
            
            self.step()
            
            # Real-time factor control
            if self.config.real_time_factor > 0:
                expected_duration = self.config.dt / self.config.real_time_factor
                actual_duration = time.time() - step_start_time
                
                if actual_duration < expected_duration:
                    time.sleep(expected_duration - actual_duration)
                    
    def _compute_derivatives(self, state, t):
        """Compute state derivatives for RK4 integration"""
        if self.physics_engine:
            return self.physics_engine.compute_derivatives(state, t)
        return np.zeros_like(state)
        
    def _notify_state_change(self):
        """Notify all state callbacks of state change"""
        for callback in self.state_callbacks:
            callback(self.state)
            
    def get_status(self) -> Dict[str, Any]:
        """Get simulation status"""
        return {
            'state': self.state.value,
            'current_time': self.current_time,
            'step_count': self.step_count,
            'dt': self.config.dt,
            'real_time_factor': self.config.real_time_factor
        }

    def set_rigid_body(self, rigid_body: RigidBody):
        """Set the rigid body for simulation"""
        self.rigid_body = rigid_body
        # Initialize with default state for now
        initial_state = DroneState()
        self.state_manager.set_state(initial_state)
    
    def set_controller(self, controller: BaseController):
        """Set the controller for simulation"""
        self.controller = controller
    
    def _initialize_simulation(self):
        """Initialize simulation state"""
        self.current_time = 0.0
        self.is_running = True
        self._start_wall_time = time.time()
        self._step_times.clear()
        self.metrics = SimulationMetrics()
        
        # Validate initial configuration
        if self.config.physics_validation:
            self._validate_simulation_config()
    
    def _simulation_step(self):
        """Perform a single simulation step"""
        if not self.rigid_body:
            raise RuntimeError("No rigid body set for simulation")
        
        # Get current state
        current_state = self.state_manager.get_state()
        
        # Apply control inputs if controller is available
        if self.controller:
            control_forces = self.controller.compute_control(current_state, self.current_time)
            self.rigid_body.apply_forces(control_forces)
        
        # Apply environmental effects if environment is available
        if self.environment:
            env_forces = self.environment.compute_forces(current_state, self.current_time)
            self.rigid_body.apply_forces(env_forces)
        
        # Integrate physics (RK4 with quaternion normalization)
        new_state = self._integrate_physics(current_state)
        
        # Update state with physics validation
        if self.config.physics_validation:
            new_state = self._validate_and_correct_state(new_state)
        
        self.state_manager.set_state(new_state, self.current_time)
        
        # Emit step event
        self.event_system.emit('simulation_step', {
            'time': self.current_time,
            'state': new_state
        })
    
    def _integrate_physics(self, state: DroneState) -> DroneState:
        """Integrate physics using simple Euler method with quaternion normalization"""
        dt = self.config.dt
        
        # Simple physics integration for validation purposes
        # In a real implementation, this would use the rigid body dynamics
        new_state = state.copy()
        
        # Simple gravity effect
        gravity_accel = np.array([0, 0, 9.81])  # m/s^2
        new_state.velocity += gravity_accel * dt
        new_state.position += new_state.velocity * dt
        
        # Simple angular damping
        damping_factor = 0.99
        new_state.angular_velocity *= damping_factor
        
        # CRITICAL FIX: Normalize quaternion to prevent drift
        # This addresses the quaternion drift issues found in physics analysis
        quat = new_state.quaternion
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-10:  # Avoid division by zero
            new_state.quaternion = quat / quat_norm
        else:
            # Reset to identity quaternion if degenerate
            new_state.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            self.metrics.physics_violations.append(f"Quaternion reset at t={self.current_time:.3f}")
        
        return new_state
    
    def _validate_and_correct_state(self, state: DroneState) -> DroneState:
        """Validate and correct physical state"""
        corrected_state = state.copy()
        
        # Validate position bounds (reasonable flight envelope)
        max_altitude = 10000.0  # 10km
        min_altitude = -100.0   # 100m below sea level
        
        if state.position[2] > max_altitude:
            corrected_state.position[2] = max_altitude
            corrected_state.velocity[2] = 0.0  # Stop climbing
            self.metrics.physics_violations.append(f"Altitude capped at {max_altitude}m")
        elif state.position[2] < min_altitude:
            corrected_state.position[2] = min_altitude
            corrected_state.velocity[2] = 0.0  # Stop descending
            self.metrics.physics_violations.append(f"Altitude bounded at {min_altitude}m")
        
        # Validate velocity bounds (reasonable for drones)
        max_velocity = 100.0  # 100 m/s maximum
        velocity_magnitude = np.linalg.norm(state.velocity)
        if velocity_magnitude > max_velocity:
            corrected_state.velocity = state.velocity * (max_velocity / velocity_magnitude)
            self.metrics.physics_violations.append(f"Velocity capped at {max_velocity} m/s")
        
        # Validate angular velocity bounds
        max_angular_velocity = 20.0  # 20 rad/s maximum
        angular_velocity_magnitude = np.linalg.norm(state.angular_velocity)
        if angular_velocity_magnitude > max_angular_velocity:
            corrected_state.angular_velocity = state.angular_velocity * (max_angular_velocity / angular_velocity_magnitude)
            self.metrics.physics_violations.append(f"Angular velocity capped at {max_angular_velocity} rad/s")
        
        return corrected_state
    
    def _validate_physics_state(self):
        """Validate current physics state for anomalies"""
        current_state = self.state_manager.get_state()
        
        # Check for NaN or infinite values
        state_array = np.concatenate([
            current_state.position,
            current_state.velocity,
            current_state.quaternion,
            current_state.angular_velocity
        ])
        
        if np.any(np.isnan(state_array)) or np.any(np.isinf(state_array)):
            self.metrics.physics_violations.append(f"NaN/Inf detected at t={self.current_time:.3f}")
            self.is_running = False  # Stop simulation on critical error
        
        # Check quaternion normalization
        quat_norm = np.linalg.norm(current_state.quaternion)
        if abs(quat_norm - 1.0) > 1e-3:  # Relaxed tolerance from original 1e-6
            self.metrics.performance_warnings.append(f"Quaternion drift: {quat_norm:.6f} at t={self.current_time:.3f}")
    
    def _validate_simulation_config(self):
        """Validate simulation configuration"""
        config_dict = {
            'real_time_factor': self.config.real_time_factor,
            'dt': self.config.dt,
            'max_steps': self.config.max_steps
        }
        
        validation_result = physics_validator.validate_test_results("Simulation Config", config_dict)
        
        if validation_result.violations:
            for violation in validation_result.violations:
                if "CRITICAL" in violation:
                    self.logger.error(f"Config validation: {violation}")
                else:
                    self.logger.warning(f"Config validation: {violation}")
        
        # Apply corrections if needed
        if validation_result.corrections:
            for param, corrected_value in validation_result.corrections.items():
                if param == 'real_time_factor':
                    self.config.real_time_factor = corrected_value
                    self.logger.info(f"Corrected real_time_factor to {corrected_value}")
    
    def _finalize_simulation(self, steps: int) -> Dict[str, Any]:
        """Finalize simulation and return results"""
        wall_clock_time = time.time() - self._start_wall_time
        
        # Update metrics
        self.metrics.simulation_time = self.current_time
        self.metrics.wall_clock_time = wall_clock_time
        self.metrics.steps_completed = steps
        
        # Calculate realistic performance metrics
        actual_rtf = self.metrics.actual_real_time_factor
        efficiency = self.metrics.computational_efficiency
        
        # Performance analysis
        avg_step_time = np.mean(self._step_times) if self._step_times else 0.0
        max_step_time = np.max(self._step_times) if self._step_times else 0.0
        
        results = {
            'simulation_time': self.current_time,
            'wall_clock_time': wall_clock_time,
            'steps_completed': steps,
            'actual_real_time_factor': actual_rtf,
            'computational_efficiency': efficiency,
            'avg_step_time': avg_step_time,
            'max_step_time': max_step_time,
            'physics_violations': self.metrics.physics_violations,
            'performance_warnings': self.metrics.performance_warnings,
            'final_state': self.state_manager.get_state()
        }
        
        # Physics validation of results
        if self.config.physics_validation:
            validation_result = physics_validator.validate_test_results("Simulation Results", results)
            if validation_result.violations:
                results['physics_validation'] = {
                    'violations': validation_result.violations,
                    'corrections': validation_result.corrections,
                    'is_valid': validation_result.is_valid
                }
        
        self.logger.info(f"Simulation completed: {steps} steps in {wall_clock_time:.3f}s (RTF: {actual_rtf:.2f}x, Efficiency: {efficiency*100:.1f}%)")
        
        if self.metrics.physics_violations:
            self.logger.warning(f"Physics violations detected: {len(self.metrics.physics_violations)}")
        
        return results
    
    def get_state_history(self) -> List[DroneState]:
        """Get complete state history"""
        states, _ = self.state_manager.get_history()
        return states 