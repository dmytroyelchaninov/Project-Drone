"""
Configurable propeller models with thrust and torque calculations per motor
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class PropellerType(Enum):
    """Types of propeller models"""
    SIMPLE = "simple"  # Basic momentum theory
    DETAILED = "detailed"  # Blade element momentum theory
    LOOKUP_TABLE = "lookup_table"  # Experimental data

@dataclass
class PropellerConfig:
    """Configuration for a propeller"""
    diameter: float  # meters
    pitch: float    # meters (theoretical advance per revolution)
    blades: int     # number of blades
    chord: float = 0.02  # average chord length in meters
    material: str = "carbon_fiber"
    
    # Aerodynamic coefficients
    ct_coefficient: float = 1.1e-5  # thrust coefficient
    cq_coefficient: float = 1.9e-6  # torque coefficient
    
    # Advanced parameters
    blade_twist: float = 0.0  # radians
    airfoil_type: str = "NACA2412"
    
    def __post_init__(self):
        """Validate configuration"""
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.blades < 2:
            raise ValueError("Must have at least 2 blades")
        if self.ct_coefficient <= 0 or self.cq_coefficient <= 0:
            raise ValueError("Coefficients must be positive")

class Propeller:
    """
    Configurable propeller model with thrust and torque calculations
    """
    
    def __init__(self, config: PropellerConfig, propeller_type: PropellerType = PropellerType.SIMPLE):
        self.config = config
        self.propeller_type = propeller_type
        
        # Operating parameters
        self.rpm = 0.0
        self.thrust = 0.0
        self.torque = 0.0
        self.power = 0.0
        
        # Environmental parameters
        self.air_density = 1.225  # kg/m³ at sea level
        self.temperature = 288.15  # K (15°C)
        self.pressure = 101325.0  # Pa
        
        # Performance data
        self.efficiency = 0.0
        self.figure_of_merit = 0.0
        
    def calculate_thrust(self, rpm: float, air_density: float = None) -> float:
        """
        Calculate thrust based on RPM and air density
        
        Args:
            rpm: Rotations per minute
            air_density: Air density in kg/m³ (uses default if None)
            
        Returns:
            Thrust in Newtons
        """
        self.rpm = rpm
        if air_density is not None:
            self.air_density = air_density
            
        if self.propeller_type == PropellerType.SIMPLE:
            thrust = self._calculate_thrust_simple()
        elif self.propeller_type == PropellerType.DETAILED:
            thrust = self._calculate_thrust_detailed()
        elif self.propeller_type == PropellerType.LOOKUP_TABLE:
            thrust = self._calculate_thrust_lookup()
        else:
            raise ValueError(f"Unknown propeller type: {self.propeller_type}")
            
        self.thrust = thrust
        return thrust
        
    def calculate_torque(self, rpm: float, air_density: float = None) -> float:
        """
        Calculate required motor torque
        
        Args:
            rpm: Rotations per minute
            air_density: Air density in kg/m³ (uses default if None)
            
        Returns:
            Torque in N⋅m
        """
        self.rpm = rpm
        if air_density is not None:
            self.air_density = air_density
            
        if self.propeller_type == PropellerType.SIMPLE:
            torque = self._calculate_torque_simple()
        elif self.propeller_type == PropellerType.DETAILED:
            torque = self._calculate_torque_detailed()
        elif self.propeller_type == PropellerType.LOOKUP_TABLE:
            torque = self._calculate_torque_lookup()
        else:
            raise ValueError(f"Unknown propeller type: {self.propeller_type}")
            
        self.torque = torque
        self.power = torque * (rpm * 2 * np.pi / 60)  # Convert RPM to rad/s
        
        # Calculate efficiency
        if self.power > 0:
            ideal_power = self.thrust * np.sqrt(self.thrust / (2 * self.air_density * np.pi * (self.config.diameter/2)**2))
            self.efficiency = ideal_power / self.power
            self.figure_of_merit = self.efficiency  # Simplified
        else:
            self.efficiency = 0.0
            self.figure_of_merit = 0.0
            
        return torque
        
    def calculate_forces_and_moments(self, rpm: float, air_density: float = None) -> Dict[str, float]:
        """
        Calculate all forces and moments
        
        Args:
            rpm: Rotations per minute
            air_density: Air density in kg/m³
            
        Returns:
            Dictionary with thrust, torque, power, efficiency
        """
        thrust = self.calculate_thrust(rpm, air_density)
        torque = self.calculate_torque(rpm, air_density)
        
        return {
            'thrust': thrust,
            'torque': torque,
            'power': self.power,
            'efficiency': self.efficiency,
            'figure_of_merit': self.figure_of_merit
        }
        
    def _calculate_thrust_simple(self) -> float:
        """Simple momentum theory thrust calculation"""
        # T = CT * ρ * n² * D⁴
        # where n is revolutions per second
        n = self.rpm / 60.0  # Convert to rev/s
        diameter = self.config.diameter
        
        thrust = (self.config.ct_coefficient * 
                 self.air_density * 
                 n**2 * 
                 diameter**4)
                 
        return thrust
        
    def _calculate_torque_simple(self) -> float:
        """Simple momentum theory torque calculation"""
        # Q = CQ * ρ * n² * D⁵
        n = self.rpm / 60.0  # Convert to rev/s
        diameter = self.config.diameter
        
        torque = (self.config.cq_coefficient * 
                 self.air_density * 
                 n**2 * 
                 diameter**5)
                 
        return torque
        
    def _calculate_thrust_detailed(self) -> float:
        """Blade element momentum theory thrust calculation"""
        # More detailed calculation considering blade geometry
        n = self.rpm / 60.0
        diameter = self.config.diameter
        radius = diameter / 2
        
        # Account for blade count and geometry
        blade_factor = self.config.blades / 2  # Normalized to 2-blade
        
        # Modified coefficient based on geometry
        ct_modified = self.config.ct_coefficient * blade_factor
        
        # Include pitch effects
        advance_ratio = self.config.pitch / diameter
        pitch_factor = 1.0 + 0.1 * advance_ratio  # Simplified pitch effect
        
        thrust = (ct_modified * 
                 self.air_density * 
                 n**2 * 
                 diameter**4 * 
                 pitch_factor)
                 
        return thrust
        
    def _calculate_torque_detailed(self) -> float:
        """Blade element momentum theory torque calculation"""
        n = self.rpm / 60.0
        diameter = self.config.diameter
        
        # Account for blade count and geometry
        blade_factor = self.config.blades / 2
        
        # Modified coefficient
        cq_modified = self.config.cq_coefficient * blade_factor
        
        # Include pitch effects
        advance_ratio = self.config.pitch / diameter
        pitch_factor = 1.0 + 0.15 * advance_ratio
        
        torque = (cq_modified * 
                 self.air_density * 
                 n**2 * 
                 diameter**5 * 
                 pitch_factor)
                 
        return torque
        
    def _calculate_thrust_lookup(self) -> float:
        """Lookup table based thrust calculation (placeholder)"""
        # In a real implementation, this would use experimental data
        # For now, fall back to simple calculation
        return self._calculate_thrust_simple()
        
    def _calculate_torque_lookup(self) -> float:
        """Lookup table based torque calculation (placeholder)"""
        # In a real implementation, this would use experimental data
        # For now, fall back to simple calculation
        return self._calculate_torque_simple()
        
    def update_environmental_conditions(self, temperature: float = None, 
                                      pressure: float = None, 
                                      humidity: float = None):
        """
        Update environmental conditions and recalculate air density
        
        Args:
            temperature: Temperature in Kelvin
            pressure: Pressure in Pa
            humidity: Relative humidity (0-1)
        """
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure
            
        # Calculate air density using ideal gas law with corrections
        R_specific = 287.0  # J/(kg⋅K) for dry air
        
        if humidity is not None:
            # Correct for humidity (simplified)
            self.air_density = self.pressure / (R_specific * self.temperature) * (1 - 0.378 * humidity)
        else:
            self.air_density = self.pressure / (R_specific * self.temperature)
            
    def get_performance_data(self) -> Dict[str, Any]:
        """Get current performance data"""
        return {
            'rpm': self.rpm,
            'thrust': self.thrust,
            'torque': self.torque,
            'power': self.power,
            'efficiency': self.efficiency,
            'figure_of_merit': self.figure_of_merit,
            'air_density': self.air_density,
            'temperature': self.temperature,
            'pressure': self.pressure
        }
        
    def get_config(self) -> PropellerConfig:
        """Get propeller configuration"""
        return self.config
        
    def update_config(self, new_config: PropellerConfig):
        """Update propeller configuration"""
        self.config = new_config
        
    def get_disk_area(self) -> float:
        """Get propeller disk area in m²"""
        return np.pi * (self.config.diameter / 2)**2
        
    def get_tip_speed(self) -> float:
        """Get tip speed in m/s"""
        omega = self.rpm * 2 * np.pi / 60  # rad/s
        return omega * (self.config.diameter / 2)
        
    def get_advance_ratio(self, forward_velocity: float = 0.0) -> float:
        """
        Get advance ratio (J = V / nD)
        
        Args:
            forward_velocity: Forward velocity in m/s
            
        Returns:
            Advance ratio (dimensionless)
        """
        if self.rpm == 0:
            return 0.0
        n = self.rpm / 60.0  # rev/s
        return forward_velocity / (n * self.config.diameter)

class PropellerArray:
    """
    Manages multiple propellers for multirotor configurations
    """
    
    def __init__(self):
        self.propellers: Dict[int, Propeller] = {}
        self.positions: Dict[int, np.ndarray] = {}  # Relative positions from CG
        self.orientations: Dict[int, np.ndarray] = {}  # Thrust directions
        
    def add_propeller(self, prop_id: int, propeller: Propeller, 
                     position: np.ndarray, orientation: np.ndarray = None):
        """
        Add a propeller to the array
        
        Args:
            prop_id: Unique identifier for the propeller
            propeller: Propeller instance
            position: Position relative to center of gravity [x, y, z]
            orientation: Thrust direction vector (default is [0, 0, -1])
        """
        if orientation is None:
            orientation = np.array([0.0, 0.0, -1.0])  # Default downward thrust
            
        self.propellers[prop_id] = propeller
        self.positions[prop_id] = position.copy()
        self.orientations[prop_id] = orientation / np.linalg.norm(orientation)
        
    def remove_propeller(self, prop_id: int):
        """Remove a propeller from the array"""
        if prop_id in self.propellers:
            del self.propellers[prop_id]
            del self.positions[prop_id]
            del self.orientations[prop_id]
            
    def calculate_total_forces_and_moments(self, rpms: Dict[int, float]) -> Dict[str, np.ndarray]:
        """
        Calculate total forces and moments from all propellers
        
        Args:
            rpms: Dictionary of RPM values for each propeller
            
        Returns:
            Dictionary with total force and moment vectors
        """
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        
        for prop_id in self.propellers:
            if prop_id not in rpms:
                continue
                
            propeller = self.propellers[prop_id]
            rpm = rpms[prop_id]
            
            # Calculate forces
            thrust = propeller.calculate_thrust(rpm)
            torque = propeller.calculate_torque(rpm)
            
            # Force in body frame
            thrust_vector = self.orientations[prop_id] * thrust
            total_force += thrust_vector
            
            # Moment from position (r × F)
            moment_from_position = np.cross(self.positions[prop_id], thrust_vector)
            
            # Moment from propeller torque (around thrust axis)
            moment_from_torque = self.orientations[prop_id] * torque
            
            total_moment += moment_from_position + moment_from_torque
            
        return {
            'force': total_force,
            'moment': total_moment
        }
        
    def get_propeller_count(self) -> int:
        """Get number of propellers in the array"""
        return len(self.propellers)
        
    def get_propeller_ids(self) -> list:
        """Get list of propeller IDs"""
        return list(self.propellers.keys()) 