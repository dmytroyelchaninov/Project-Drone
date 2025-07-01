"""
Quadcopter Physics Simulation
Handles the physical simulation of quadcopter dynamics
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class QuadcopterPhysicsConfig:
    """Configuration for quadcopter physics simulation"""
    # Mass properties
    mass: float = 1.5  # kg
    inertia_matrix: np.ndarray = None  # 3x3 inertia matrix [kg*m^2]
    
    # Geometry
    arm_length: float = 0.225  # meters (distance from center to engine)
    
    # Engine configuration
    num_engines: int = 4
    max_thrust_per_engine: float = 10.0  # Newtons
    min_thrust_per_engine: float = 0.0   # Newtons
    
    # Environmental constants
    gravity: float = 9.81  # m/s^2
    air_density: float = 1.225  # kg/m^3
    drag_coefficient: float = 0.1
    
    # Engine positions (body frame, relative to center of mass)
    engine_positions: np.ndarray = None
    
    def __post_init__(self):
        # Default inertia matrix for symmetric quadcopter
        if self.inertia_matrix is None:
            Ixx = Iyy = 0.01  # kg*m^2 (roll and pitch)
            Izz = 0.02        # kg*m^2 (yaw)
            self.inertia_matrix = np.array([
                [Ixx, 0.0, 0.0],
                [0.0, Iyy, 0.0], 
                [0.0, 0.0, Izz]
            ])
        
        # Default engine positions (+ configuration)
        if self.engine_positions is None:
            arm = self.arm_length
            self.engine_positions = np.array([
                [arm, 0.0, 0.0],    # Engine 0: Front (+X)
                [0.0, arm, 0.0],    # Engine 1: Right (+Y)
                [-arm, 0.0, 0.0],   # Engine 2: Back (-X)
                [0.0, -arm, 0.0]    # Engine 3: Left (-Y)
            ])

@dataclass
class QuadcopterState:
    """Complete state of quadcopter"""
    # Position and orientation
    position: np.ndarray = None          # [x, y, z] in world frame
    quaternion: np.ndarray = None        # [w, x, y, z] orientation
    velocity: np.ndarray = None          # [vx, vy, vz] in world frame
    angular_velocity: np.ndarray = None  # [wx, wy, wz] in body frame
    
    # Engine states
    engine_thrusts: np.ndarray = None    # [4] individual engine thrusts
    
    # Forces and moments
    total_force_world: np.ndarray = None    # Total force in world frame
    total_moment_body: np.ndarray = None    # Total moment in body frame
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 1.0])  # Start 1 meter above ground
        if self.quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)
        if self.engine_thrusts is None:
            self.engine_thrusts = np.zeros(4)
        if self.total_force_world is None:
            self.total_force_world = np.zeros(3)
        if self.total_moment_body is None:
            self.total_moment_body = np.zeros(3)

class QuadcopterPhysics:
    """
    Physics simulation for quadcopter with individual engine control
    SINGLETON class that manages the physics state
    """
    _instance = None
    
    def __new__(cls, config: QuadcopterPhysicsConfig = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance
    
    def __init__(self, config: QuadcopterPhysicsConfig = None):
        if not self._initialized:
            self.config = config if config else QuadcopterPhysicsConfig()
            self.state = QuadcopterState()
            
            # Initialize engines at hover
            hover_thrust_per_engine = self.config.mass * self.config.gravity / 4.0
            self.state.engine_thrusts.fill(hover_thrust_per_engine)
            
            self._initialized = True
            logger.info("QuadcopterPhysics singleton initialized")
    
    def set_engine_thrusts(self, thrusts: np.ndarray):
        """Set individual engine thrusts"""
        # Clamp to engine limits
        self.state.engine_thrusts = np.clip(
            thrusts[:4],  # Only use first 4 values
            self.config.min_thrust_per_engine,
            self.config.max_thrust_per_engine
        )
    
    def update(self, dt: float):
        """Update physics simulation for one time step"""
        # Calculate forces and moments from engine thrusts
        self._calculate_forces_and_moments()
        
        # Apply aerodynamic effects
        self._apply_aerodynamics()
        
        # Integrate equations of motion
        self._integrate_motion(dt)
        
        # Normalize quaternion to prevent drift
        self._normalize_quaternion()
    
    def _calculate_forces_and_moments(self):
        """Calculate total forces and moments from individual engine thrusts"""
        # All engines point in +Z direction in body frame
        engine_force_direction = np.array([0.0, 0.0, 1.0])
        
        # Total thrust force in body frame
        total_thrust_body = np.sum(self.state.engine_thrusts) * engine_force_direction
        
        # Transform thrust force from body to world frame
        self.state.total_force_world = self._body_to_world_force(total_thrust_body)
        
        # Add gravity (world frame)
        gravity_force = np.array([0.0, 0.0, -self.config.mass * self.config.gravity])
        self.state.total_force_world += gravity_force
        
        # Calculate moments about center of mass (body frame)
        total_moment = np.zeros(3)
        
        for i, (pos, thrust) in enumerate(zip(self.config.engine_positions, self.state.engine_thrusts)):
            # Force vector at engine position (body frame)
            force_at_engine = thrust * engine_force_direction
            
            # Moment = r × F (cross product)
            moment = np.cross(pos, force_at_engine)
            total_moment += moment
        
        self.state.total_moment_body = total_moment
    
    def _apply_aerodynamics(self):
        """Apply aerodynamic effects (drag)"""
        # Simple drag model
        velocity_magnitude = np.linalg.norm(self.state.velocity)
        if velocity_magnitude > 0:
            drag_force = -0.5 * self.config.air_density * self.config.drag_coefficient * \
                        velocity_magnitude * self.state.velocity
            self.state.total_force_world += drag_force
    
    def _integrate_motion(self, dt: float):
        """Integrate equations of motion using Euler integration"""
        # Linear motion (Newton's second law)
        acceleration = self.state.total_force_world / self.config.mass
        
        # Update velocity and position
        self.state.velocity += acceleration * dt
        new_position = self.state.position + self.state.velocity * dt
        
        # Ground collision detection and handling
        ground_height = 0.0  # Ground at z = 0
        if new_position[2] < ground_height:
            # Hit the ground - handle collision
            # logger.info(f"Ground collision detected at altitude {new_position[2]:.2f}m")
            
            # Clamp position to ground level
            new_position[2] = ground_height
            
            # Set vertical velocity to zero (no bouncing)
            if self.state.velocity[2] < 0:
                self.state.velocity[2] = 0.0
            
            # Apply some friction to horizontal movement when on ground
            friction_factor = 0.8
            self.state.velocity[0] *= friction_factor
            self.state.velocity[1] *= friction_factor
            
            # logger.debug(f"Ground collision resolved: position={new_position}, velocity={self.state.velocity}")
        
        # Update position
        self.state.position = new_position
        
        # Angular motion (Euler's equation)
        # τ = I * α + ω × (I * ω)
        inertia = self.config.inertia_matrix
        angular_momentum = inertia @ self.state.angular_velocity
        gyroscopic_term = np.cross(self.state.angular_velocity, angular_momentum)
        angular_acceleration = np.linalg.solve(inertia, self.state.total_moment_body - gyroscopic_term)
        
        # Update angular velocity
        self.state.angular_velocity += angular_acceleration * dt
        
        # Update orientation (quaternion integration)
        self._integrate_quaternion(dt)
    
    def _integrate_quaternion(self, dt: float):
        """Integrate quaternion from angular velocity"""
        # Quaternion derivative: q̇ = 0.5 * q * ω_quat
        omega = self.state.angular_velocity
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        
        q_dot = 0.5 * self._quaternion_multiply(self.state.quaternion, omega_quat)
        self.state.quaternion += q_dot * dt
    
    def _body_to_world_force(self, force_body: np.ndarray) -> np.ndarray:
        """Transform force from body frame to world frame using quaternion"""
        # Rotate vector using quaternion: v_world = q * v_body * q*
        return self._rotate_vector_by_quaternion(force_body, self.state.quaternion)
    
    def _rotate_vector_by_quaternion(self, vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion"""
        # Convert vector to quaternion
        v_quat = np.array([0.0, vector[0], vector[1], vector[2]])
        
        # Rotation: q * v * q_conjugate
        q_conj = self._quaternion_conjugate(quat)
        rotated_quat = self._quaternion_multiply(
            self._quaternion_multiply(quat, v_quat), q_conj
        )
        
        # Extract vector part
        return rotated_quat[1:4]
    
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
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _normalize_quaternion(self):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(self.state.quaternion)
        if norm > 0:
            self.state.quaternion /= norm
    
    def reset(self, position: np.ndarray = None, velocity: np.ndarray = None):
        """Reset quadcopter state"""
        self.state = QuadcopterState()
        
        if position is not None:
            self.state.position = position.copy()
        if velocity is not None:
            self.state.velocity = velocity.copy()
        
        # Initialize at hover
        hover_thrust = self.config.mass * self.config.gravity / 4.0
        self.state.engine_thrusts.fill(hover_thrust)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary"""
        return {
            'position': self.state.position.copy(),
            'quaternion': self.state.quaternion.copy(),
            'velocity': self.state.velocity.copy(),
            'angular_velocity': self.state.angular_velocity.copy(),
            'engine_thrusts': self.state.engine_thrusts.copy(),
            'total_force_world': self.state.total_force_world.copy(),
            'total_moment_body': self.state.total_moment_body.copy()
        }
    
    def get_euler_angles(self) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        q = self.state.quaternion
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Roll (rotation about x-axis)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        
        # Pitch (rotation about y-axis)
        sin_pitch = 2*(w*y - z*x)
        pitch = np.arcsin(np.clip(sin_pitch, -1, 1))
        
        # Yaw (rotation about z-axis)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return np.array([roll, pitch, yaw])
    
    def get_hover_thrust_per_engine(self) -> float:
        """Get thrust per engine needed for hover"""
        return self.config.mass * self.config.gravity / 4.0 