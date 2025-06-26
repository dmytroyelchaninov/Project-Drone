"""
Environment module for gravity, wind, and external disturbances
"""
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class WindModel(Enum):
    """Types of wind models"""
    CONSTANT = "constant"
    TURBULENT = "turbulent"
    GUST = "gust"
    CUSTOM = "custom"

@dataclass
class EnvironmentConfig:
    """Configuration for environmental conditions"""
    gravity_magnitude: float = 9.81  # m/s²
    gravity_direction: np.ndarray = None  # Default is [0, 0, 1] (downward), normalized vector
    air_density: float = 1.225  # kg/m³
    temperature: float = 288.15  # K
    pressure: float = 101325.0  # Pa
    
    def __post_init__(self):
        if self.gravity_direction is None:
            self.gravity_direction = np.array([0.0, 0.0, 1.0])
        self.gravity_direction = self.gravity_direction / np.linalg.norm(self.gravity_direction)

class Environment:
    """
    Environmental conditions and disturbances
    """
    
    def __init__(self, config: EnvironmentConfig = None):
        self.config = config or EnvironmentConfig()
        
        # Current time
        self.time = 0.0
        
        # Wind parameters
        self.wind_velocity = np.zeros(3)  # Wind velocity in inertial frame
        self.wind_model_type = WindModel.CONSTANT
        self.turbulence_intensity = 0.0
        self.gust_parameters = {}
        
        # Custom disturbance function
        self.custom_disturbance_fn: Optional[Callable] = None
        
        # Gravity vector
        self.gravity_vector = (self.config.gravity_magnitude * 
                              self.config.gravity_direction)
        
    def update(self, time: float, dt: float):
        """
        Update environmental conditions
        
        Args:
            time: Current simulation time
            dt: Time step
        """
        self.time = time
        
        # Update wind based on model
        if self.wind_model_type == WindModel.CONSTANT:
            pass  # Wind velocity remains constant
        elif self.wind_model_type == WindModel.TURBULENT:
            self._update_turbulent_wind(dt)
        elif self.wind_model_type == WindModel.GUST:
            self._update_gust_wind()
        elif self.wind_model_type == WindModel.CUSTOM:
            self._update_custom_wind()
            
    def get_gravity_force(self, mass: float) -> np.ndarray:
        """
        Get gravity force for given mass
        
        Args:
            mass: Mass in kg
            
        Returns:
            Gravity force vector in inertial frame
        """
        return mass * self.gravity_vector
        
    def get_wind_velocity(self, position: np.ndarray = None) -> np.ndarray:
        """
        Get wind velocity at position
        
        Args:
            position: Position vector (for spatially varying wind)
            
        Returns:
            Wind velocity vector in inertial frame
        """
        # For now, assume uniform wind field
        return self.wind_velocity.copy()
        
    def get_air_density(self, altitude: float = 0.0) -> float:
        """
        Get air density at altitude
        
        Args:
            altitude: Altitude above sea level in meters
            
        Returns:
            Air density in kg/m³
        """
        # Simple atmospheric model
        # ρ = ρ₀ * exp(-altitude / scale_height)
        scale_height = 8000.0  # meters
        return self.config.air_density * np.exp(-altitude / scale_height)
        
    def set_constant_wind(self, wind_velocity: np.ndarray):
        """
        Set constant wind
        
        Args:
            wind_velocity: Wind velocity vector [vx, vy, vz] in m/s
        """
        self.wind_model_type = WindModel.CONSTANT
        self.wind_velocity = wind_velocity.copy()
        
    def set_turbulent_wind(self, mean_wind: np.ndarray, turbulence_intensity: float):
        """
        Set turbulent wind model
        
        Args:
            mean_wind: Mean wind velocity vector
            turbulence_intensity: Turbulence intensity (0-1)
        """
        self.wind_model_type = WindModel.TURBULENT
        self.wind_velocity = mean_wind.copy()
        self.turbulence_intensity = turbulence_intensity
        
    def set_wind_gust(self, base_wind: np.ndarray, gust_amplitude: float, 
                     gust_duration: float, gust_start_time: float):
        """
        Set wind gust parameters
        
        Args:
            base_wind: Base wind velocity
            gust_amplitude: Gust amplitude in m/s
            gust_duration: Gust duration in seconds
            gust_start_time: When gust starts in seconds
        """
        self.wind_model_type = WindModel.GUST
        self.wind_velocity = base_wind.copy()
        self.gust_parameters = {
            'amplitude': gust_amplitude,
            'duration': gust_duration,
            'start_time': gust_start_time,
            'base_wind': base_wind.copy()
        }
        
    def set_custom_wind(self, custom_function: Callable[[float], np.ndarray]):
        """
        Set custom wind function
        
        Args:
            custom_function: Function that takes time and returns wind velocity
        """
        self.wind_model_type = WindModel.CUSTOM
        self.custom_disturbance_fn = custom_function
        
    def apply_payload_drop(self, mass_change: float):
        """
        Apply sudden mass change (payload drop)
        
        Args:
            mass_change: Change in mass (negative for drop)
        """
        # This would typically be handled by the calling system
        # that manages the rigid body mass
        pass
        
    def _update_turbulent_wind(self, dt: float):
        """Update turbulent wind model"""
        if self.turbulence_intensity > 0:
            # Simple turbulence model using random walk
            turbulence_scale = self.turbulence_intensity * 5.0  # m/s
            correlation_time = 2.0  # seconds
            
            # Ornstein-Uhlenbeck process for each component
            alpha = dt / correlation_time
            noise = np.random.normal(0, 1, 3) * np.sqrt(2 * alpha) * turbulence_scale
            
            # Update wind with correlated noise
            self.wind_velocity += -alpha * self.wind_velocity + noise
            
    def _update_gust_wind(self):
        """Update wind gust model"""
        params = self.gust_parameters
        
        if (self.time >= params['start_time'] and 
            self.time <= params['start_time'] + params['duration']):
            
            # Gust profile (1-cosine shape)
            t_rel = (self.time - params['start_time']) / params['duration']
            gust_factor = 0.5 * (1 - np.cos(2 * np.pi * t_rel))
            
            # Apply gust in x-direction (can be modified)
            gust_vector = np.array([params['amplitude'], 0.0, 0.0])
            self.wind_velocity = params['base_wind'] + gust_factor * gust_vector
        else:
            self.wind_velocity = params['base_wind'].copy()
            
    def _update_custom_wind(self):
        """Update custom wind model"""
        if self.custom_disturbance_fn:
            self.wind_velocity = self.custom_disturbance_fn(self.time)
            
    def reset(self):
        """Reset environment to initial conditions"""
        self.time = 0.0
        self.wind_velocity.fill(0.0)
        self.wind_model_type = WindModel.CONSTANT
        self.turbulence_intensity = 0.0
        self.gust_parameters.clear()
        self.custom_disturbance_fn = None 