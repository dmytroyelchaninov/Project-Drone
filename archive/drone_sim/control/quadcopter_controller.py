"""
Physics-based quadcopter controller with individual engine thrust control
"""
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from .base_controller import BaseController, ControllerState, ControllerReference, ControllerOutput

@dataclass
class QuadcopterConfig:
    """Configuration for quadcopter physics"""
    # Engine positions in body frame (relative to center of mass)
    # Standard quadcopter configuration (+ shape)
    #   1
    # 4 + 2  
    #   3
    engine_positions: np.ndarray = None  # [4, 3] array of engine positions
    engine_arm_length: float = 0.225  # meters (typical for 450mm frame)
    
    # Engine specifications
    max_thrust_per_engine: float = 10.0  # Newtons per engine
    min_thrust_per_engine: float = 0.0   # Minimum thrust
    thrust_response_time: float = 0.05   # Engine response time constant
    
    # Mass and gravity
    mass: float = 1.5  # kg
    gravity: float = 9.81  # m/s^2
    
    # Control sensitivity
    thrust_sensitivity: float = 1.0  # Thrust change per control input
    rotation_sensitivity: float = 0.5  # Rotation moment per control input
    
    def __post_init__(self):
        if self.engine_positions is None:
            # Standard quadcopter layout (+ configuration)
            arm = self.engine_arm_length
            self.engine_positions = np.array([
                [arm, 0.0, 0.0],    # Engine 1: Front (+X)
                [0.0, arm, 0.0],    # Engine 2: Right (+Y) 
                [-arm, 0.0, 0.0],   # Engine 3: Back (-X)
                [0.0, -arm, 0.0]    # Engine 4: Left (-Y)
            ])

@dataclass
class EngineStates:
    """Current state of all engines"""
    thrusts: np.ndarray = None  # [4] array of engine thrusts in Newtons
    target_thrusts: np.ndarray = None  # Target thrusts for smooth control
    
    def __post_init__(self):
        if self.thrusts is None:
            self.thrusts = np.zeros(4)
        if self.target_thrusts is None:
            self.target_thrusts = np.zeros(4)

class QuadcopterController(BaseController):
    """
    Physics-based quadcopter controller with individual engine control
    """
    
    def __init__(self, config: QuadcopterConfig = None):
        super().__init__("QuadcopterController")
        
        self.config = config if config else QuadcopterConfig()
        self.engine_states = EngineStates()
        
        # Calculate hover thrust (thrust needed to counteract gravity)
        self.hover_thrust_total = self.config.mass * self.config.gravity
        self.hover_thrust_per_engine = self.hover_thrust_total / 4.0
        
        # Initialize engines at hover
        self.engine_states.thrusts.fill(self.hover_thrust_per_engine)
        self.engine_states.target_thrusts.fill(self.hover_thrust_per_engine)
        
        # Control state
        self.control_inputs = {
            'throttle': 0.0,    # -1 to 1 (down to up)
            'roll': 0.0,        # -1 to 1 (left to right)
            'pitch': 0.0,       # -1 to 1 (backward to forward)
            'yaw': 0.0          # -1 to 1 (CCW to CW)
        }
        
    def update(self, reference: ControllerReference, 
               current_state: ControllerState, 
               dt: float) -> ControllerOutput:
        """Update quadcopter controller"""
        if not self.enabled:
            return ControllerOutput()
        
        # Update engine thrust targets based on control inputs
        self._update_engine_targets()
        
        # Apply engine response dynamics (first-order lag)
        self._apply_engine_dynamics(dt)
        
        # Calculate total thrust and moments from individual engines
        total_thrust, total_moment = self._calculate_forces_and_moments()
        
        # Create output with individual engine commands
        output = ControllerOutput(
            thrust=total_thrust,
            moment=total_moment,
            motor_commands=self.engine_states.thrusts.copy()
        )
        
        # Update metrics
        self._update_metrics(reference, current_state, output)
        
        return output
    
    def set_control_inputs(self, throttle: float = 0.0, roll: float = 0.0, 
                          pitch: float = 0.0, yaw: float = 0.0):
        """
        Set control inputs directly
        
        Args:
            throttle: Vertical thrust control (-1 to 1)
            roll: Roll control (-1 to 1, positive = right)
            pitch: Pitch control (-1 to 1, positive = forward)
            yaw: Yaw control (-1 to 1, positive = clockwise)
        """
        self.control_inputs['throttle'] = np.clip(throttle, -1.0, 1.0)
        self.control_inputs['roll'] = np.clip(roll, -1.0, 1.0)
        self.control_inputs['pitch'] = np.clip(pitch, -1.0, 1.0)
        self.control_inputs['yaw'] = np.clip(yaw, -1.0, 1.0)
    
    def _update_engine_targets(self):
        """Update target thrusts for each engine based on control inputs"""
        # Start with hover thrust for all engines
        base_thrust = self.hover_thrust_per_engine
        
        # Calculate thrust modifications for each control input
        throttle_delta = self.control_inputs['throttle'] * self.config.thrust_sensitivity * base_thrust
        roll_delta = self.control_inputs['roll'] * self.config.rotation_sensitivity * base_thrust
        pitch_delta = self.control_inputs['pitch'] * self.config.rotation_sensitivity * base_thrust
        yaw_delta = self.control_inputs['yaw'] * self.config.rotation_sensitivity * base_thrust * 0.5
        
        # Apply control mixing for quadcopter (+) configuration
        # Engine layout:
        #   1 (Front)
        # 4 + 2 (Left/Right)
        #   3 (Back)
        
        self.engine_states.target_thrusts[0] = base_thrust + throttle_delta + pitch_delta - yaw_delta  # Front
        self.engine_states.target_thrusts[1] = base_thrust + throttle_delta + roll_delta + yaw_delta   # Right
        self.engine_states.target_thrusts[2] = base_thrust + throttle_delta - pitch_delta - yaw_delta  # Back
        self.engine_states.target_thrusts[3] = base_thrust + throttle_delta - roll_delta + yaw_delta   # Left
        
        # Clamp to engine limits
        self.engine_states.target_thrusts = np.clip(
            self.engine_states.target_thrusts,
            self.config.min_thrust_per_engine,
            self.config.max_thrust_per_engine
        )
    
    def _apply_engine_dynamics(self, dt: float):
        """Apply first-order engine response dynamics"""
        # Simple first-order lag: thrust(t+dt) = thrust(t) + (target - thrust(t)) * (dt / tau)
        tau = self.config.thrust_response_time
        alpha = dt / (tau + dt)  # Low-pass filter coefficient
        
        self.engine_states.thrusts = (
            self.engine_states.thrusts * (1 - alpha) + 
            self.engine_states.target_thrusts * alpha
        )
    
    def _calculate_forces_and_moments(self) -> tuple:
        """Calculate total thrust and moments from individual engine thrusts"""
        # Total thrust (all engines point in +Z body direction)
        total_thrust = np.sum(self.engine_states.thrusts)
        
        # Calculate moments about center of mass
        # Each engine creates a force vector [0, 0, thrust] at its position
        moments = np.zeros(3)
        
        for i, (pos, thrust) in enumerate(zip(self.config.engine_positions, self.engine_states.thrusts)):
            # Force vector (all engines point up in body frame)
            force_vector = np.array([0.0, 0.0, thrust])
            
            # Moment = position × force
            moment = np.cross(pos, force_vector)
            moments += moment
        
        return total_thrust, moments
    
    def get_engine_thrusts(self) -> np.ndarray:
        """Get current engine thrusts"""
        return self.engine_states.thrusts.copy()
    
    def get_hover_thrust(self) -> float:
        """Get hover thrust (total thrust needed to counteract gravity)"""
        return self.hover_thrust_total
    
    def reset(self):
        """Reset controller to hover state"""
        self.engine_states.thrusts.fill(self.hover_thrust_per_engine)
        self.engine_states.target_thrusts.fill(self.hover_thrust_per_engine)
        self.control_inputs = {
            'throttle': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }
        self.update_count = 0
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information"""
        return {
            'engine_thrusts': self.engine_states.thrusts.tolist(),
            'target_thrusts': self.engine_states.target_thrusts.tolist(),
            'control_inputs': self.control_inputs.copy(),
            'hover_thrust': self.hover_thrust_total,
            'total_thrust': np.sum(self.engine_states.thrusts)
        } 