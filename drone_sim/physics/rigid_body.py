"""
6DOF rigid body dynamics with quaternion-based attitude representation
"""
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class RigidBodyConfig:
    """Configuration for rigid body properties"""
    mass: float  # kg
    inertia: np.ndarray  # 3x3 inertia matrix in kg⋅m²
    
    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        if self.inertia.shape != (3, 3):
            raise ValueError("Inertia must be 3x3 matrix")

class RigidBody:
    """
    6DOF rigid body dynamics implementation
    """
    
    def __init__(self, config: RigidBodyConfig):
        self.config = config
        
        # State variables (13 DOF)
        self.position = np.zeros(3)     # [x, y, z] in inertial frame
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.velocity = np.zeros(3)     # [vx, vy, vz] in body frame
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz] in body frame
        
        # Force and moment accumulators
        self.forces = np.zeros(3)       # Forces in body frame
        self.moments = np.zeros(3)      # Moments in body frame
        
        # Gravity vector in inertial frame
        self.gravity = np.array([0.0, 0.0, 9.81])
        
    def compute_derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute state derivatives for integration
        
        Args:
            state: 13-element state vector [pos, quat, vel, omega]
            t: Current time
            
        Returns:
            State derivatives
        """
        # Extract state components
        position = state[0:3]
        quaternion = state[3:7]
        velocity = state[7:10]
        angular_velocity = state[10:13]
        
        # Normalize quaternion to prevent drift
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        # Position derivative (velocity in inertial frame)
        R_body_to_inertial = self.quaternion_to_rotation_matrix(quaternion)
        position_dot = R_body_to_inertial @ velocity
        
        # Quaternion derivative
        omega_quat = np.array([0.0, angular_velocity[0], angular_velocity[1], angular_velocity[2]])
        quaternion_dot = 0.5 * self.quaternion_multiply(quaternion, omega_quat)
        
        # Linear velocity derivative (Newton's second law in body frame)
        gravity_body = R_body_to_inertial.T @ self.gravity
        linear_acceleration = (self.forces / self.config.mass + 
                             gravity_body - 
                             np.cross(angular_velocity, velocity))
        velocity_dot = linear_acceleration
        
        # Angular velocity derivative (Euler's equation)
        inertia_omega = self.config.inertia @ angular_velocity
        angular_acceleration = np.linalg.solve(
            self.config.inertia,
            self.moments - np.cross(angular_velocity, inertia_omega)
        )
        angular_velocity_dot = angular_acceleration
        
        # Combine derivatives
        derivatives = np.concatenate([
            position_dot,
            quaternion_dot,
            velocity_dot,
            angular_velocity_dot
        ])
        
        return derivatives
        
    def apply_force(self, force: np.ndarray, position: np.ndarray = None):
        """
        Apply force at a position
        
        Args:
            force: Force vector in body frame
            position: Position relative to CG (if None, applies at CG)
        """
        self.forces += force
        
        if position is not None:
            # Add moment due to force offset
            moment = np.cross(position, force)
            self.moments += moment
            
    def apply_moment(self, moment: np.ndarray):
        """
        Apply moment directly
        
        Args:
            moment: Moment vector in body frame
        """
        self.moments += moment
        
    def clear_forces_and_moments(self):
        """Clear accumulated forces and moments"""
        self.forces.fill(0.0)
        self.moments.fill(0.0)
        
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix (body to inertial)
        """
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
        ])
        
        return R
        
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
            
        Returns:
            Product quaternion
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        result = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        return result
        
    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector"""
        return np.concatenate([
            self.position,
            self.quaternion,
            self.velocity,
            self.angular_velocity
        ])
        
    def set_state_vector(self, state: np.ndarray):
        """Set state from vector"""
        self.position = state[0:3].copy()
        self.quaternion = state[3:7].copy()
        self.velocity = state[7:10].copy()
        self.angular_velocity = state[10:13].copy()
        
        # Normalize quaternion
        self.quaternion /= np.linalg.norm(self.quaternion)
        
    def get_euler_angles(self) -> np.ndarray:
        """Get Euler angles from quaternion"""
        q = self.quaternion
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Roll (x-axis rotation)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        
        # Pitch (y-axis rotation)
        sin_pitch = 2*(w*y - z*x)
        sin_pitch = np.clip(sin_pitch, -1, 1)
        pitch = np.arcsin(sin_pitch)
        
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return np.array([roll, pitch, yaw])
        
    def update_inertia_tensor(self, new_inertia: np.ndarray):
        """Update inertia tensor for configuration changes"""
        self.config.inertia = new_inertia.copy()
        
    def reset(self):
        """Reset to initial conditions"""
        self.position.fill(0.0)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.velocity.fill(0.0)
        self.angular_velocity.fill(0.0)
        self.clear_forces_and_moments() 