"""
Environment Physics Module
Handles environmental effects like wind, ground effects, and atmospheric conditions
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass 
class EnvironmentConfig:
    """Configuration for environmental effects"""
    # Atmospheric properties
    gravity: float = 9.81           # m/s^2
    air_density: float = 1.225      # kg/m^3 at sea level
    temperature: float = 288.15     # K (15°C)
    pressure: float = 101325.0      # Pa (sea level)
    
    # Wind configuration
    wind_enabled: bool = False
    wind_speed: float = 0.0         # m/s
    wind_direction: float = 0.0     # degrees (0 = North)
    wind_turbulence: float = 0.1    # turbulence intensity (0-1)
    
    # Ground effects
    ground_effect_enabled: bool = True
    ground_effect_height: float = 2.0  # meters (height where ground effect starts)
    
    # Boundaries
    max_altitude: float = 1000.0    # meters
    boundary_size: float = 1000.0   # meters (square boundary)

class Environment:
    """
    Environment singleton that manages atmospheric and environmental effects
    """
    _instance = None
    
    def __new__(cls, config: EnvironmentConfig = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance
    
    def __init__(self, config: EnvironmentConfig = None):
        if not self._initialized:
            self.config = config if config else EnvironmentConfig()
            self._wind_velocity = np.zeros(3)  # [vx, vy, vz] in world frame
            self._ground_height = 0.0  # Ground level (can be variable in future)
            
            # Initialize wind if enabled
            if self.config.wind_enabled:
                self._update_wind()
            
            self._initialized = True
            logger.info("Environment singleton initialized")
    
    def update(self, dt: float):
        """Update environmental conditions"""
        if self.config.wind_enabled:
            self._update_wind_turbulence(dt)
    
    def get_atmospheric_properties(self, altitude: float) -> Dict[str, float]:
        """Get atmospheric properties at given altitude"""
        # Simple atmospheric model (could be enhanced with ISA model)
        if altitude < 0:
            altitude = 0
        
        # Temperature lapse rate: 6.5°C per 1000m
        temp = self.config.temperature - 0.0065 * altitude
        
        # Pressure (barometric formula)
        pressure = self.config.pressure * (temp / self.config.temperature) ** 5.255
        
        # Air density
        air_density = pressure / (287.058 * temp)  # R = 287.058 J/(kg·K)
        
        return {
            'temperature': temp,
            'pressure': pressure,
            'air_density': air_density,
            'altitude': altitude
        }
    
    def get_wind_velocity(self, position: np.ndarray) -> np.ndarray:
        """Get wind velocity at given position"""
        if not self.config.wind_enabled:
            return np.zeros(3)
        
        # Add some spatial variation to wind (simple model)
        x, y, z = position
        
        # Base wind vector
        wind_base = self._wind_velocity.copy()
        
        # Add altitude variation (wind increases with height)
        altitude_factor = 1.0 + 0.1 * z  # 10% increase per meter
        wind_base *= altitude_factor
        
        # Add turbulence based on position
        if self.config.wind_turbulence > 0:
            turbulence = self.config.wind_turbulence * np.array([
                np.sin(0.1 * x) * np.cos(0.1 * y),
                np.cos(0.1 * x) * np.sin(0.1 * y),
                0.5 * np.sin(0.05 * (x + y))
            ])
            wind_base += turbulence
        
        return wind_base
    
    def get_ground_effect_factor(self, position: np.ndarray) -> float:
        """Calculate ground effect factor (increases thrust near ground)"""
        if not self.config.ground_effect_enabled:
            return 1.0
        
        height_above_ground = position[2] - self._ground_height
        
        if height_above_ground <= 0:
            return 1.0  # On or below ground
        
        if height_above_ground >= self.config.ground_effect_height:
            return 1.0  # Above ground effect zone
        
        # Ground effect increases thrust when close to ground
        # Effect decreases linearly with height
        effect_strength = 1.0 - (height_above_ground / self.config.ground_effect_height)
        ground_effect_factor = 1.0 + 0.1 * effect_strength  # Up to 10% thrust increase
        
        return ground_effect_factor
    
    def check_boundaries(self, position: np.ndarray) -> Dict[str, bool]:
        """Check if position is within environment boundaries"""
        x, y, z = position
        
        return {
            'altitude_exceeded': z > self.config.max_altitude,
            'ground_collision': z < self._ground_height,
            'boundary_exceeded': (abs(x) > self.config.boundary_size or 
                                abs(y) > self.config.boundary_size),
            'within_bounds': (z >= self._ground_height and 
                            z <= self.config.max_altitude and
                            abs(x) <= self.config.boundary_size and
                            abs(y) <= self.config.boundary_size)
        }
    
    def get_aerodynamic_forces(self, velocity: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Calculate aerodynamic forces (wind effects, etc.)"""
        forces = np.zeros(3)
        
        # Wind effects
        if self.config.wind_enabled:
            wind_velocity = self.get_wind_velocity(position)
            relative_velocity = velocity - wind_velocity
            
            # Simple wind resistance model
            # F_wind = -0.5 * ρ * C_d * A * |v_rel| * v_rel
            atm_props = self.get_atmospheric_properties(position[2])
            air_density = atm_props['air_density']
            
            drag_coefficient = 0.05  # Simple drag model
            reference_area = 0.1     # m^2 (approximate quadcopter frontal area)
            
            relative_speed = np.linalg.norm(relative_velocity)
            if relative_speed > 0:
                drag_force = -0.5 * air_density * drag_coefficient * reference_area * \
                           relative_speed * relative_velocity
                forces += drag_force
        
        return forces
    
    def _update_wind(self):
        """Update wind velocity vector from speed and direction"""
        # Convert wind direction to velocity vector
        wind_dir_rad = np.radians(self.config.wind_direction)
        
        # Wind direction: 0° = North (+Y), 90° = East (+X)
        self._wind_velocity = self.config.wind_speed * np.array([
            np.sin(wind_dir_rad),   # X component
            np.cos(wind_dir_rad),   # Y component  
            0.0                     # Z component (no vertical wind)
        ])
    
    def _update_wind_turbulence(self, dt: float):
        """Update wind turbulence over time"""
        if self.config.wind_turbulence > 0:
            # Add time-varying turbulence
            time_scale = 0.1  # Hz
            turbulence_amplitude = self.config.wind_turbulence * self.config.wind_speed
            
            # Simple sinusoidal turbulence
            import time
            t = time.time()
            
            turbulence = turbulence_amplitude * np.array([
                0.5 * np.sin(time_scale * t),
                0.3 * np.cos(time_scale * t * 1.3),
                0.1 * np.sin(time_scale * t * 0.7)
            ])
            
            # Update wind velocity with turbulence
            base_wind = self.config.wind_speed * np.array([
                np.sin(np.radians(self.config.wind_direction)),
                np.cos(np.radians(self.config.wind_direction)),
                0.0
            ])
            
            self._wind_velocity = base_wind + turbulence
    
    def set_wind(self, speed: float, direction: float):
        """Set wind conditions"""
        self.config.wind_speed = speed
        self.config.wind_direction = direction
        self._update_wind()
    
    def set_ground_height(self, height: float):
        """Set ground level height"""
        self._ground_height = height
    
    def get_ground_height(self, position: Optional[np.ndarray] = None) -> float:
        """Get ground height at position (or current ground height if no position given)"""
        # For now, return uniform ground height
        # In future could implement terrain height based on position
        return self._ground_height
    
    def get_wind_at_position(self, position: np.ndarray) -> np.ndarray:
        """Get wind velocity at given position (alias for get_wind_velocity)"""
        return self.get_wind_velocity(position)
    
    def get_air_density(self, altitude: float) -> float:
        """Get air density at given altitude"""
        atm_props = self.get_atmospheric_properties(altitude)
        return atm_props['air_density']
    
    def get_environment_info(self, position: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive environment information at position"""
        atm_props = self.get_atmospheric_properties(position[2])
        wind_vel = self.get_wind_velocity(position)
        ground_effect = self.get_ground_effect_factor(position)
        boundaries = self.check_boundaries(position)
        
        return {
            'atmospheric': atm_props,
            'wind_velocity': wind_vel,
            'wind_speed': np.linalg.norm(wind_vel),
            'ground_effect_factor': ground_effect,
            'height_above_ground': position[2] - self._ground_height,
            'boundaries': boundaries
        }
