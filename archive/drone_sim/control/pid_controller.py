"""
Classical PID controller implementation with anti-windup
"""
import numpy as np
from typing import Dict, Any
from .base_controller import BaseController, ControllerState, ControllerReference, ControllerOutput

class PIDController(BaseController):
    """
    Three-layer PID controller for drone control
    - Position PID (NED frame)
    - Attitude PID (Quaternion error)
    - Rate PID (Body frame)
    """
    
    def __init__(self, controller_name: str = "PIDController"):
        super().__init__(controller_name)
        
        # Default PID gains
        self.set_parameters({
            # Position controller gains
            'pos_kp': [1.0, 1.0, 2.0],  # [x, y, z]
            'pos_ki': [0.1, 0.1, 0.2],
            'pos_kd': [0.5, 0.5, 1.0],
            
            # Attitude controller gains
            'att_kp': [6.0, 6.0, 3.0],  # [roll, pitch, yaw]
            'att_ki': [0.1, 0.1, 0.1],
            'att_kd': [0.3, 0.3, 0.3],
            
            # Rate controller gains
            'rate_kp': [150.0, 150.0, 50.0],  # [p, q, r]
            'rate_ki': [50.0, 50.0, 20.0],
            'rate_kd': [5.0, 5.0, 2.0],
            
            # Anti-windup limits
            'pos_integral_limit': [5.0, 5.0, 5.0],
            'att_integral_limit': [1.0, 1.0, 1.0],
            'rate_integral_limit': [100.0, 100.0, 50.0],
            
            # Output limits
            'max_tilt_angle': 0.5,  # radians (about 30 degrees)
            'max_thrust': 20.0,     # N
            'min_thrust': 0.0,      # N
            'max_angular_rate': 5.0  # rad/s
        })
        
        # Internal state
        self.reset()
        
    def reset(self):
        """Reset controller internal state"""
        self.internal_state = {
            'pos_integral': np.zeros(3),
            'pos_error_prev': np.zeros(3),
            'att_integral': np.zeros(3),
            'att_error_prev': np.zeros(3),
            'rate_integral': np.zeros(3),
            'rate_error_prev': np.zeros(3),
            'initialized': False
        }
        
    def update(self, reference: ControllerReference, 
               current_state: ControllerState, 
               dt: float) -> ControllerOutput:
        """
        Update the PID controller
        
        Args:
            reference: Reference/desired state
            current_state: Current drone state
            dt: Time step in seconds
            
        Returns:
            Control commands
        """
        if not self.enabled:
            return ControllerOutput()
            
        # Get parameters as numpy arrays
        pos_kp = np.array(self.get_parameter('pos_kp'))
        pos_ki = np.array(self.get_parameter('pos_ki'))
        pos_kd = np.array(self.get_parameter('pos_kd'))
        
        att_kp = np.array(self.get_parameter('att_kp'))
        att_ki = np.array(self.get_parameter('att_ki'))
        att_kd = np.array(self.get_parameter('att_kd'))
        
        rate_kp = np.array(self.get_parameter('rate_kp'))
        rate_ki = np.array(self.get_parameter('rate_ki'))
        rate_kd = np.array(self.get_parameter('rate_kd'))
        
        # Position control
        desired_acceleration = self._position_control(
            reference, current_state, dt, pos_kp, pos_ki, pos_kd
        )
        
        # Convert desired acceleration to desired attitude and thrust
        desired_attitude, desired_thrust = self._acceleration_to_attitude_thrust(
            desired_acceleration, current_state
        )
        
        # Attitude control
        desired_angular_velocity = self._attitude_control(
            desired_attitude, current_state, dt, att_kp, att_ki, att_kd
        )
        
        # Rate control
        desired_moment = self._rate_control(
            desired_angular_velocity, current_state, dt, rate_kp, rate_ki, rate_kd
        )
        
        # Create output
        output = ControllerOutput(
            thrust=desired_thrust,
            moment=desired_moment
        )
        
        # Update metrics
        self._update_metrics(reference, current_state, output)
        
        return output
        
    def _position_control(self, reference: ControllerReference, 
                         current_state: ControllerState, 
                         dt: float, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> np.ndarray:
        """Position PID control loop"""
        
        # Position error
        position_error = reference.position - current_state.position
        
        # Velocity error (if reference velocity is provided)
        if hasattr(reference, 'velocity') and reference.velocity is not None:
            velocity_error = reference.velocity - current_state.velocity
        else:
            velocity_error = np.zeros(3)
            
        # Initialize previous error if needed
        if not self.internal_state['initialized']:
            self.internal_state['pos_error_prev'] = position_error.copy()
            self.internal_state['initialized'] = True
            
        # Integral term
        self.internal_state['pos_integral'] += position_error * dt
        
        # Apply integral windup protection
        integral_limits = np.array(self.get_parameter('pos_integral_limit'))
        self.internal_state['pos_integral'] = np.clip(
            self.internal_state['pos_integral'], -integral_limits, integral_limits
        )
        
        # Derivative term
        if dt > 0:
            position_derivative = (position_error - self.internal_state['pos_error_prev']) / dt
        else:
            position_derivative = np.zeros(3)
            
        self.internal_state['pos_error_prev'] = position_error.copy()
        
        # PID output (desired acceleration)
        desired_acceleration = (kp * position_error + 
                              ki * self.internal_state['pos_integral'] + 
                              kd * position_derivative)
        
        # Add feedforward acceleration if provided
        if hasattr(reference, 'acceleration') and reference.acceleration is not None:
            desired_acceleration += reference.acceleration
            
        return desired_acceleration
        
    def _acceleration_to_attitude_thrust(self, desired_acceleration: np.ndarray, 
                                       current_state: ControllerState) -> tuple:
        """Convert desired acceleration to attitude and thrust commands"""
        
        # Add gravity compensation
        gravity = np.array([0.0, 0.0, 9.81])
        desired_acceleration += gravity
        
        # Total desired force magnitude
        desired_thrust = np.linalg.norm(desired_acceleration)
        
        # Limit thrust
        max_thrust = self.get_parameter('max_thrust')
        min_thrust = self.get_parameter('min_thrust')
        desired_thrust = np.clip(desired_thrust, min_thrust, max_thrust)
        
        # Desired thrust direction (normalized)
        if desired_thrust > 0:
            thrust_direction = desired_acceleration / np.linalg.norm(desired_acceleration)
        else:
            thrust_direction = np.array([0.0, 0.0, -1.0])  # Default downward
            
        # Convert thrust direction to desired attitude (quaternion)
        # Assume thrust is along negative z-axis in body frame
        body_z = -thrust_direction  # Negative because thrust is upward
        
        # Create a rotation matrix from current z-axis to desired z-axis
        current_rotation = self._quaternion_to_rotation_matrix(current_state.quaternion)
        
        # Simplified attitude calculation - assume no yaw preference
        # Roll and pitch to align z-axis with thrust direction
        max_tilt = self.get_parameter('max_tilt_angle')
        
        # Limit tilt angle
        tilt_angle = np.arccos(np.clip(-body_z[2], -1, 1))
        if tilt_angle > max_tilt:
            # Scale down the tilt to maximum allowed
            scale_factor = max_tilt / tilt_angle
            body_z[0] *= scale_factor
            body_z[1] *= scale_factor
            body_z[2] = -np.sqrt(1 - body_z[0]**2 - body_z[1]**2)
            
        # Convert to Euler angles (simplified)
        roll = np.arctan2(body_z[1], body_z[2])
        pitch = -np.arcsin(body_z[0])
        yaw = 0.0  # No yaw preference for now
        
        # Convert to quaternion
        desired_attitude = self._euler_to_quaternion(roll, pitch, yaw)
        
        return desired_attitude, desired_thrust
        
    def _attitude_control(self, desired_attitude: np.ndarray, 
                         current_state: ControllerState, 
                         dt: float, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> np.ndarray:
        """Attitude PID control loop"""
        
        # Quaternion error
        q_error = self._quaternion_error(desired_attitude, current_state.quaternion)
        
        # Convert to Euler angle error for control
        attitude_error = self._quaternion_to_euler(q_error)
        
        # Integral term
        self.internal_state['att_integral'] += attitude_error * dt
        
        # Apply integral windup protection
        integral_limits = np.array(self.get_parameter('att_integral_limit'))
        self.internal_state['att_integral'] = np.clip(
            self.internal_state['att_integral'], -integral_limits, integral_limits
        )
        
        # Derivative term
        if dt > 0:
            attitude_derivative = (attitude_error - self.internal_state['att_error_prev']) / dt
        else:
            attitude_derivative = np.zeros(3)
            
        self.internal_state['att_error_prev'] = attitude_error.copy()
        
        # PID output (desired angular velocity)
        desired_angular_velocity = (kp * attitude_error + 
                                  ki * self.internal_state['att_integral'] + 
                                  kd * attitude_derivative)
        
        # Limit angular velocity
        max_rate = self.get_parameter('max_angular_rate')
        desired_angular_velocity = np.clip(desired_angular_velocity, -max_rate, max_rate)
        
        return desired_angular_velocity
        
    def _rate_control(self, desired_angular_velocity: np.ndarray, 
                     current_state: ControllerState, 
                     dt: float, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray) -> np.ndarray:
        """Rate PID control loop"""
        
        # Angular velocity error
        rate_error = desired_angular_velocity - current_state.angular_velocity
        
        # Integral term
        self.internal_state['rate_integral'] += rate_error * dt
        
        # Apply integral windup protection
        integral_limits = np.array(self.get_parameter('rate_integral_limit'))
        self.internal_state['rate_integral'] = np.clip(
            self.internal_state['rate_integral'], -integral_limits, integral_limits
        )
        
        # Derivative term
        if dt > 0:
            rate_derivative = (rate_error - self.internal_state['rate_error_prev']) / dt
        else:
            rate_derivative = np.zeros(3)
            
        self.internal_state['rate_error_prev'] = rate_error.copy()
        
        # PID output (desired moment)
        desired_moment = (kp * rate_error + 
                         ki * self.internal_state['rate_integral'] + 
                         kd * rate_derivative)
        
        return desired_moment
        
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
        ])
        
        return R
        
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion"""
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
        
        return np.array([qw, qx, qy, qz])
        
    def _quaternion_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles"""
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
        
    def _quaternion_error(self, q_desired: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Compute quaternion error"""
        # Normalize quaternions
        q_desired = q_desired / np.linalg.norm(q_desired)
        q_current = q_current / np.linalg.norm(q_current)
        
        # Compute error quaternion
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        q_error = self._quaternion_multiply(q_desired, q_current_conj)
        
        return q_error
        
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        
        result = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        return result 