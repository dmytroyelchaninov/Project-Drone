"""
Propeller Physics Module
Handles propeller aerodynamics and thrust/torque calculations
"""
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PropellerConfig:
    """Configuration for propeller aerodynamics"""
    # Physical properties
    diameter: float = 0.24      # meters
    pitch: float = 0.12         # meters (theoretical advance per revolution)
    num_blades: int = 2         # number of blades
    
    # Aerodynamic coefficients
    ct_coefficient: float = 1.1e-5    # Thrust coefficient
    cq_coefficient: float = 1.9e-6    # Torque coefficient
    
    # Performance characteristics
    max_rpm: float = 8000       # maximum RPM
    min_rpm: float = 0          # minimum RPM
    
    # Motor characteristics
    rpm_per_volt: float = 800.0  # RPM per volt
    motor_resistance: float = 0.5  # Ohms
    motor_constant: float = 0.01   # Nm/A (torque constant)

class Propeller:
    """
    Propeller aerodynamics model
    Calculates thrust and torque from RPM and air conditions
    """
    
    def __init__(self, config: PropellerConfig = None):
        self.config = config if config else PropellerConfig()
        self._last_thrust = 0.0
        self._last_torque = 0.0
        self._last_power = 0.0
    
    def calculate_thrust_from_rpm(self, rpm: float, air_density: float = 1.225) -> float:
        """Calculate thrust from RPM and air density"""
        # Clamp RPM to valid range
        rpm = np.clip(rpm, self.config.min_rpm, self.config.max_rpm)
        
        if rpm <= 0:
            return 0.0
        
        # Thrust formula: T = Ct * ρ * n² * D⁴
        # Where: Ct = thrust coefficient, ρ = air density, n = RPS, D = diameter
        rps = rpm / 60.0  # Convert RPM to RPS
        diameter = self.config.diameter
        
        thrust = (self.config.ct_coefficient * air_density * 
                 (rps ** 2) * (diameter ** 4))
        
        self._last_thrust = thrust
        return thrust
    
    def calculate_torque_from_rpm(self, rpm: float, air_density: float = 1.225) -> float:
        """Calculate torque from RPM and air density"""
        # Clamp RPM to valid range
        rpm = np.clip(rpm, self.config.min_rpm, self.config.max_rpm)
        
        if rpm <= 0:
            return 0.0
        
        # Torque formula: Q = Cq * ρ * n² * D⁵
        # Where: Cq = torque coefficient, ρ = air density, n = RPS, D = diameter
        rps = rpm / 60.0  # Convert RPM to RPS
        diameter = self.config.diameter
        
        torque = (self.config.cq_coefficient * air_density * 
                 (rps ** 2) * (diameter ** 5))
        
        self._last_torque = torque
        return torque
    
    def calculate_power_from_rpm(self, rpm: float, air_density: float = 1.225) -> float:
        """Calculate power consumption from RPM"""
        if rpm <= 0:
            return 0.0
        
        torque = self.calculate_torque_from_rpm(rpm, air_density)
        angular_velocity = rpm * 2 * np.pi / 60.0  # rad/s
        
        power = torque * angular_velocity
        self._last_power = power
        return power
    
    def calculate_rpm_from_voltage(self, voltage: float) -> float:
        """Calculate RPM from applied voltage"""
        # Simple linear model: RPM = K * V
        # In reality, this would depend on load, but this is a good approximation
        rpm = voltage * self.config.rpm_per_volt
        return np.clip(rpm, self.config.min_rpm, self.config.max_rpm)
    
    def calculate_voltage_from_rpm(self, rpm: float) -> float:
        """Calculate required voltage for target RPM"""
        # Inverse of rpm_from_voltage
        voltage = rpm / self.config.rpm_per_volt
        return max(0.0, voltage)
    
    def calculate_thrust_from_voltage(self, voltage: float, air_density: float = 1.225) -> float:
        """Calculate thrust directly from voltage"""
        rpm = self.calculate_rpm_from_voltage(voltage)
        return self.calculate_thrust_from_rpm(rpm, air_density)
    
    def calculate_efficiency(self, rpm: float, air_density: float = 1.225) -> float:
        """Calculate propeller efficiency"""
        if rpm <= 0:
            return 0.0
        
        thrust = self.calculate_thrust_from_rpm(rpm, air_density)
        power = self.calculate_power_from_rpm(rpm, air_density)
        
        if power <= 0:
            return 0.0
        
        # Ideal efficiency calculation
        # This is simplified - real efficiency depends on advance ratio
        velocity = 0.0  # Assume static thrust for now
        
        # Power loading (thrust per unit power)
        power_loading = thrust / power if power > 0 else 0
        
        # Simplified efficiency model
        efficiency = min(0.85, power_loading * 0.1)  # Cap at 85% max efficiency
        return max(0.0, efficiency)
    
    def get_performance_info(self, rpm: float, air_density: float = 1.225) -> Dict[str, float]:
        """Get comprehensive performance information"""
        thrust = self.calculate_thrust_from_rpm(rpm, air_density)
        torque = self.calculate_torque_from_rpm(rpm, air_density)
        power = self.calculate_power_from_rpm(rpm, air_density)
        efficiency = self.calculate_efficiency(rpm, air_density)
        voltage = self.calculate_voltage_from_rpm(rpm)
        
        return {
            'rpm': rpm,
            'thrust': thrust,
            'torque': torque,
            'power': power,
            'efficiency': efficiency,
            'voltage': voltage,
            'thrust_to_weight_ratio': thrust / 9.81,  # Assume 1kg reference
            'power_loading': thrust / power if power > 0 else 0,
            'disc_loading': thrust / (np.pi * (self.config.diameter/2)**2)
        }
    
    def get_hover_specs(self, total_weight: float, num_engines: int = 4) -> Dict[str, float]:
        """Calculate hover requirements for given weight"""
        thrust_per_engine = total_weight * 9.81 / num_engines
        
        # Find RPM needed for hover thrust
        # This requires solving: thrust = Ct * ρ * n² * D⁴
        # n = sqrt(thrust / (Ct * ρ * D⁴))
        air_density = 1.225  # Sea level
        
        required_rps_squared = thrust_per_engine / (
            self.config.ct_coefficient * air_density * (self.config.diameter ** 4)
        )
        
        if required_rps_squared <= 0:
            return {
                'hover_rpm': 0,
                'hover_voltage': 0,
                'hover_power': 0,
                'hover_current': 0,
                'margin_to_max': 0
            }
        
        required_rps = np.sqrt(required_rps_squared)
        hover_rpm = required_rps * 60.0  # Convert to RPM
        
        # Clamp to max RPM
        hover_rpm = min(hover_rpm, self.config.max_rpm)
        
        hover_voltage = self.calculate_voltage_from_rpm(hover_rpm)
        hover_power = self.calculate_power_from_rpm(hover_rpm)
        hover_current = hover_power / hover_voltage if hover_voltage > 0 else 0
        
        margin_to_max = (self.config.max_rpm - hover_rpm) / self.config.max_rpm
        
        return {
            'hover_rpm': hover_rpm,
            'hover_voltage': hover_voltage,
            'hover_power': hover_power,
            'hover_current': hover_current,
            'margin_to_max': margin_to_max,
            'thrust_per_engine': thrust_per_engine
        }
    
    @property
    def last_performance(self) -> Dict[str, float]:
        """Get last calculated performance values"""
        return {
            'thrust': self._last_thrust,
            'torque': self._last_torque,
            'power': self._last_power
        }
    
    def calculate_rpm_from_thrust(self, thrust: float, air_density: float = 1.225) -> float:
        """Calculate RPM required to produce given thrust"""
        if thrust <= 0:
            return 0.0
        
        # Solve: thrust = Ct * ρ * n² * D⁴
        # n = sqrt(thrust / (Ct * ρ * D⁴))
        diameter = self.config.diameter
        
        required_rps_squared = thrust / (
            self.config.ct_coefficient * air_density * (diameter ** 4)
        )
        
        if required_rps_squared <= 0:
            return 0.0
        
        required_rps = np.sqrt(required_rps_squared)
        rpm = required_rps * 60.0  # Convert to RPM
        
        # Clamp to valid range
        return np.clip(rpm, self.config.min_rpm, self.config.max_rpm) 