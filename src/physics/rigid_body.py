"""
Rigid Body Physics Module
Handles 3D rigid body dynamics with quaternion rotations
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RigidBodyConfig:
    """Configuration for rigid body physics"""
    mass: float = 1.0                    # kg
    inertia_matrix: np.ndarray = None    # 3x3 inertia tensor [kg*m^2]
    center_of_mass: np.ndarray = None    # [x, y, z] offset from origin
    
    def __post_init__(self):
        if self.inertia_matrix is None:
            # Default to unit sphere inertia
            I = 2.0/5.0 * self.mass * 0.1**2  # Sphere with radius 0.1m
            self.inertia_matrix = np.array([
                [I, 0, 0],
                [0, I, 0],
                [0, 0, I]
            ])
        
        if self.center_of_mass is None:
            self.center_of_mass = np.zeros(3)

@dataclass
class RigidBodyState:
    """Complete state of a rigid body"""
    # Linear motion
    position: np.ndarray = None          # [x, y, z] in world frame
    velocity: np.ndarray = None          # [vx, vy, vz] in world frame
    acceleration: np.ndarray = None      # [ax, ay, az] in world frame
    
    # Angular motion
    quaternion: np.ndarray = None        # [w, x, y, z] orientation
    angular_velocity: np.ndarray = None  # [wx, wy, wz] in body frame
    angular_acceleration: np.ndarray = None  # [αx, αy, αz] in body frame
    
    # Forces and moments
    force_world: np.ndarray = None       # Total external force in world frame
    moment_body: np.ndarray = None       # Total external moment in body frame
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.acceleration is None:
            self.acceleration = np.zeros(3)
        if self.quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)
        if self.angular_acceleration is None:
            self.angular_acceleration = np.zeros(3)
        if self.force_world is None:
            self.force_world = np.zeros(3)
        if self.moment_body is None:
            self.moment_body = np.zeros(3)

class RigidBody:
    """
    3D rigid body with quaternion-based orientation
    Handles general rigid body dynamics
    """
    
    def __init__(self, config: RigidBodyConfig = None):
        self.config = config if config else RigidBodyConfig()
        self.state = RigidBodyState()
        
        # Derived properties
        self._rotation_matrix = np.eye(3)
        self._inertia_world = self.config.inertia_matrix.copy()
        
        self._update_derived_properties()
    
    def apply_force_at_point(self, force_world: np.ndarray, point_world: np.ndarray):
        """Apply force at a specific point in world coordinates"""
        # Add force to total force
        self.state.force_world += force_world
        
        # Calculate moment about center of mass
        com_world = self.state.position + self._rotate_vector_to_world(self.config.center_of_mass)
        moment_arm = point_world - com_world
        moment_world = np.cross(moment_arm, force_world)
        
        # Convert moment to body frame and add to total moment
        moment_body = self._rotate_vector_to_body(moment_world)
        self.state.moment_body += moment_body
    
    def apply_force_at_com(self, force_world: np.ndarray):
        """Apply force at center of mass (no resulting moment)"""
        self.state.force_world += force_world
    
    def apply_moment_body(self, moment_body: np.ndarray):
        """Apply moment in body frame"""
        self.state.moment_body += moment_body
    
    def clear_forces(self):
        """Clear all applied forces and moments"""
        self.state.force_world.fill(0.0)
        self.state.moment_body.fill(0.0)
    
    def update(self, dt: float):
        """Update rigid body dynamics for one time step"""
        # Calculate accelerations
        self._calculate_accelerations()
        
        # Integrate linear motion
        self._integrate_linear_motion(dt)
        
        # Integrate angular motion
        self._integrate_angular_motion(dt)
        
        # Update derived properties
        self._update_derived_properties()
        
        # Clear forces for next iteration
        self.clear_forces()
    
    def _calculate_accelerations(self):
        """Calculate linear and angular accelerations from forces/moments"""
        # Linear acceleration: a = F / m
        self.state.acceleration = self.state.force_world / self.config.mass
        
        # Angular acceleration: α = I⁻¹ * (τ - ω × (I * ω))
        inertia = self.config.inertia_matrix
        angular_momentum = inertia @ self.state.angular_velocity
        gyroscopic_term = np.cross(self.state.angular_velocity, angular_momentum)
        
        net_moment = self.state.moment_body - gyroscopic_term
        self.state.angular_acceleration = np.linalg.solve(inertia, net_moment)
    
    def _integrate_linear_motion(self, dt: float):
        """Integrate linear motion using Euler method"""
        # Update velocity: v = v₀ + a * dt
        self.state.velocity += self.state.acceleration * dt
        
        # Update position: x = x₀ + v * dt
        self.state.position += self.state.velocity * dt
    
    def _integrate_angular_motion(self, dt: float):
        """Integrate angular motion with quaternions"""
        # Update angular velocity: ω = ω₀ + α * dt
        self.state.angular_velocity += self.state.angular_acceleration * dt
        
        # Integrate quaternion
        self._integrate_quaternion(dt)
        
        # Normalize quaternion to prevent drift
        self._normalize_quaternion()
    
    def _integrate_quaternion(self, dt: float):
        """Integrate quaternion from angular velocity"""
        # Quaternion derivative: q̇ = 0.5 * q * ω_quat
        omega = self.state.angular_velocity
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        
        q_dot = 0.5 * self._quaternion_multiply(self.state.quaternion, omega_quat)
        self.state.quaternion += q_dot * dt
    
    def _update_derived_properties(self):
        """Update rotation matrix and world-frame inertia"""
        # Update rotation matrix from quaternion
        self._rotation_matrix = self._quaternion_to_rotation_matrix(self.state.quaternion)
        
        # Update world-frame inertia tensor: I_world = R * I_body * R^T
        R = self._rotation_matrix
        I_body = self.config.inertia_matrix
        self._inertia_world = R @ I_body @ R.T
    
    def _rotate_vector_to_world(self, vector_body: np.ndarray) -> np.ndarray:
        """Rotate vector from body frame to world frame"""
        return self._rotation_matrix @ vector_body
    
    def _rotate_vector_to_body(self, vector_world: np.ndarray) -> np.ndarray:
        """Rotate vector from world frame to body frame"""
        return self._rotation_matrix.T @ vector_world
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _normalize_quaternion(self):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(self.state.quaternion)
        if norm > 0:
            self.state.quaternion /= norm
    
    def set_position(self, position: np.ndarray):
        """Set position in world frame"""
        self.state.position = position.copy()
    
    def set_velocity(self, velocity: np.ndarray):
        """Set velocity in world frame"""
        self.state.velocity = velocity.copy()
    
    def set_orientation_euler(self, roll: float, pitch: float, yaw: float):
        """Set orientation from Euler angles (radians)"""
        self.state.quaternion = self._euler_to_quaternion(roll, pitch, yaw)
        self._update_derived_properties()
    
    def set_angular_velocity(self, angular_velocity: np.ndarray):
        """Set angular velocity in body frame"""
        self.state.angular_velocity = angular_velocity.copy()
    
    def get_euler_angles(self) -> np.ndarray:
        """Get orientation as Euler angles [roll, pitch, yaw] in radians"""
        q = self.state.quaternion
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Roll (x-axis rotation)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        
        # Pitch (y-axis rotation)
        sin_pitch = 2*(w*y - z*x)
        pitch = np.arcsin(np.clip(sin_pitch, -1, 1))
        
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return np.array([roll, pitch, yaw])
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion"""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def get_kinetic_energy(self) -> Dict[str, float]:
        """Calculate kinetic energies"""
        # Translational kinetic energy
        ke_trans = 0.5 * self.config.mass * np.dot(self.state.velocity, self.state.velocity)
        
        # Rotational kinetic energy
        omega = self.state.angular_velocity
        ke_rot = 0.5 * omega.T @ self.config.inertia_matrix @ omega
        
        return {
            'translational': ke_trans,
            'rotational': ke_rot,
            'total': ke_trans + ke_rot
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state as dictionary"""
        return {
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'acceleration': self.state.acceleration.copy(),
            'quaternion': self.state.quaternion.copy(),
            'angular_velocity': self.state.angular_velocity.copy(),
            'angular_acceleration': self.state.angular_acceleration.copy(),
            'euler_angles': self.get_euler_angles(),
            'kinetic_energy': self.get_kinetic_energy()
        } 