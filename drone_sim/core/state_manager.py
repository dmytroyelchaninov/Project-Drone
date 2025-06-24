"""
Handles drone state transitions, validation, and history management
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings

@dataclass
class DroneState:
    """
    13-DOF drone state representation
    """
    # Position (3 DOF) - North, East, Down in meters
    position: np.ndarray = None  # [x, y, z]
    
    # Orientation (4 DOF) - Quaternion [w, x, y, z]
    quaternion: np.ndarray = None  # [qw, qx, qy, qz]
    
    # Linear velocity (3 DOF) - Body frame in m/s
    velocity: np.ndarray = None  # [vx, vy, vz]
    
    # Angular velocity (3 DOF) - Body frame in rad/s
    angular_velocity: np.ndarray = None  # [wx, wy, wz]
    
    def __post_init__(self):
        """Initialize arrays if not provided"""
        if self.position is None:
            self.position = np.zeros(3)
        if self.quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)
            
    def to_vector(self) -> np.ndarray:
        """Convert state to a 13-element vector"""
        return np.concatenate([
            self.position,
            self.quaternion,
            self.velocity,
            self.angular_velocity
        ])
        
    @classmethod
    def from_vector(cls, state_vector: np.ndarray) -> 'DroneState':
        """Create DroneState from 13-element vector"""
        if len(state_vector) != 13:
            raise ValueError(f"State vector must have 13 elements, got {len(state_vector)}")
            
        return cls(
            position=state_vector[0:3].copy(),
            quaternion=state_vector[3:7].copy(),
            velocity=state_vector[7:10].copy(),
            angular_velocity=state_vector[10:13].copy()
        )
        
    def copy(self) -> 'DroneState':
        """Create a deep copy of the state"""
        return DroneState(
            position=self.position.copy(),
            quaternion=self.quaternion.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy()
        )

class StateValidator:
    """Validates drone state for physical consistency"""
    
    def __init__(self):
        # Physical bounds
        self.max_position = 1000.0  # meters
        self.max_velocity = 50.0    # m/s
        self.max_angular_velocity = 10.0  # rad/s
        self.quaternion_tolerance = 1e-4
        
    def validate(self, state: DroneState) -> Tuple[bool, List[str]]:
        """
        Validate state and return (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for NaN values
        if np.any(np.isnan(state.position)):
            errors.append("Position contains NaN values")
        if np.any(np.isnan(state.quaternion)):
            errors.append("Quaternion contains NaN values")
        if np.any(np.isnan(state.velocity)):
            errors.append("Velocity contains NaN values")
        if np.any(np.isnan(state.angular_velocity)):
            errors.append("Angular velocity contains NaN values")
            
        # Check for infinite values
        if np.any(np.isinf(state.position)):
            errors.append("Position contains infinite values")
        if np.any(np.isinf(state.quaternion)):
            errors.append("Quaternion contains infinite values")
        if np.any(np.isinf(state.velocity)):
            errors.append("Velocity contains infinite values")
        if np.any(np.isinf(state.angular_velocity)):
            errors.append("Angular velocity contains infinite values")
            
        # Check physical bounds
        if np.linalg.norm(state.position) > self.max_position:
            errors.append(f"Position magnitude exceeds limit ({self.max_position} m)")
            
        if np.linalg.norm(state.velocity) > self.max_velocity:
            errors.append(f"Velocity magnitude exceeds limit ({self.max_velocity} m/s)")
            
        if np.linalg.norm(state.angular_velocity) > self.max_angular_velocity:
            errors.append(f"Angular velocity exceeds limit ({self.max_angular_velocity} rad/s)")
            
        # Validate quaternion normalization
        quat_norm = np.linalg.norm(state.quaternion)
        if abs(quat_norm - 1.0) > self.quaternion_tolerance:
            errors.append(f"Quaternion not normalized: norm = {quat_norm}")
            
        return len(errors) == 0, errors
        
    def sanitize(self, state: DroneState) -> DroneState:
        """
        Sanitize state by fixing common issues
        """
        sanitized = state.copy()
        
        # Replace NaN with zeros
        sanitized.position = np.nan_to_num(sanitized.position)
        sanitized.velocity = np.nan_to_num(sanitized.velocity)
        sanitized.angular_velocity = np.nan_to_num(sanitized.angular_velocity)
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(sanitized.quaternion)
        if quat_norm > 1e-12:
            sanitized.quaternion /= quat_norm
        else:
            sanitized.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            
        # Clamp values to physical bounds
        pos_norm = np.linalg.norm(sanitized.position)
        if pos_norm > self.max_position:
            sanitized.position *= self.max_position / pos_norm
            
        vel_norm = np.linalg.norm(sanitized.velocity)
        if vel_norm > self.max_velocity:
            sanitized.velocity *= self.max_velocity / vel_norm
            
        omega_norm = np.linalg.norm(sanitized.angular_velocity)
        if omega_norm > self.max_angular_velocity:
            sanitized.angular_velocity *= self.max_angular_velocity / omega_norm
            
        return sanitized

class StateManager:
    """
    Manages drone state transitions, validation, and history
    """
    
    def __init__(self, history_size: int = 1000):
        self.current_state = DroneState()
        self.validator = StateValidator()
        
        # Rolling history buffer
        self.history_size = history_size
        self.state_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
        # State change callbacks
        self.state_callbacks = []
        
        # Statistics
        self.validation_failures = 0
        self.sanitization_count = 0
        
    def set_state(self, new_state: DroneState, timestamp: float = 0.0, validate: bool = True):
        """
        Set the current state with optional validation
        """
        if validate:
            is_valid, errors = self.validator.validate(new_state)
            
            if not is_valid:
                self.validation_failures += 1
                warnings.warn(f"State validation failed: {errors}")
                
                # Attempt to sanitize
                new_state = self.validator.sanitize(new_state)
                self.sanitization_count += 1
                
                # Re-validate after sanitization
                is_valid, errors = self.validator.validate(new_state)
                if not is_valid:
                    raise ValueError(f"State could not be sanitized: {errors}")
                    
        # Update current state
        self.current_state = new_state.copy()
        
        # Add to history
        self.state_history.append(self.current_state.copy())
        self.time_history.append(timestamp)
        
        # Notify callbacks
        self._notify_state_change(timestamp)
        
    def get_state(self) -> DroneState:
        """Get the current state"""
        return self.current_state.copy()
        
    def get_state_vector(self) -> np.ndarray:
        """Get current state as a vector"""
        return self.current_state.to_vector()
        
    def set_state_vector(self, state_vector: np.ndarray, timestamp: float = 0.0, validate: bool = True):
        """Set state from a vector"""
        new_state = DroneState.from_vector(state_vector)
        self.set_state(new_state, timestamp, validate)
        
    def get_history(self, n_samples: int = None) -> Tuple[List[DroneState], List[float]]:
        """
        Get state history
        
        Args:
            n_samples: Number of recent samples to return, or None for all
            
        Returns:
            Tuple of (states, timestamps)
        """
        if n_samples is None:
            return list(self.state_history), list(self.time_history)
        else:
            n_samples = min(n_samples, len(self.state_history))
            return (list(self.state_history)[-n_samples:], 
                   list(self.time_history)[-n_samples:])
            
    def reset(self, initial_state: DroneState = None):
        """Reset to initial state and clear history"""
        if initial_state is None:
            initial_state = DroneState()
            
        self.current_state = initial_state.copy()
        self.state_history.clear()
        self.time_history.clear()
        
        # Reset statistics
        self.validation_failures = 0
        self.sanitization_count = 0
        
    def add_state_callback(self, callback):
        """Add a callback for state changes"""
        self.state_callbacks.append(callback)
        
    def remove_state_callback(self, callback):
        """Remove a state change callback"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)
            
    def _notify_state_change(self, timestamp: float):
        """Notify all callbacks of state change"""
        for callback in self.state_callbacks:
            try:
                callback(self.current_state, timestamp)
            except Exception as e:
                warnings.warn(f"State callback failed: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        return {
            'validation_failures': self.validation_failures,
            'sanitization_count': self.sanitization_count,
            'history_size': len(self.state_history),
            'max_history_size': self.history_size
        }
        
    def get_euler_angles(self) -> np.ndarray:
        """Convert current quaternion to Euler angles (roll, pitch, yaw)"""
        q = self.current_state.quaternion
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        # Roll (x-axis rotation)
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        
        # Pitch (y-axis rotation)
        sin_pitch = 2*(qw*qy - qz*qx)
        sin_pitch = np.clip(sin_pitch, -1, 1)  # Clamp to avoid numerical issues
        pitch = np.arcsin(sin_pitch)
        
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        return np.array([roll, pitch, yaw])
        
    def set_euler_angles(self, roll: float, pitch: float, yaw: float):
        """Set orientation using Euler angles"""
        # Convert to quaternion
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        self.current_state.quaternion = np.array([qw, qx, qy, qz]) 