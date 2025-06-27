"""
Voltage-based engine controller for drone manual control
Handles voltage inputs for individual engines and converts to physics-compatible signals
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

from .base_device import BaseDevice, DeviceConfig, DeviceStatus

class VoltageControllerConfig(DeviceConfig):
    """Configuration for voltage controller"""
    def __init__(self, name: str, poll_rate: float = 100.0, timeout: float = 1.0,
                 retry_attempts: int = 3, auto_reconnect: bool = True,
                 validation_enabled: bool = True, num_engines: int = 4,
                 min_voltage: float = 0.0, max_voltage: float = 12.0,
                 hover_voltage: float = 6.0, max_rpm_per_volt: float = 800.0,
                 voltage_response_time: float = 0.05, voltage_ramp_rate: float = 24.0,
                 emergency_cutoff_enabled: bool = True, voltage_offsets: Optional[np.ndarray] = None,
                 voltage_gains: Optional[np.ndarray] = None):
        super().__init__(name, poll_rate, timeout, retry_attempts, auto_reconnect, validation_enabled)
        
        self.num_engines = num_engines
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.hover_voltage = hover_voltage
        self.max_rpm_per_volt = max_rpm_per_volt
        self.voltage_response_time = voltage_response_time
        self.voltage_ramp_rate = voltage_ramp_rate
        self.emergency_cutoff_enabled = emergency_cutoff_enabled
        
        if voltage_offsets is None:
            self.voltage_offsets = np.zeros(self.num_engines)
        else:
            self.voltage_offsets = voltage_offsets
            
        if voltage_gains is None:
            self.voltage_gains = np.ones(self.num_engines)
        else:
            self.voltage_gains = voltage_gains

class VoltageController(BaseDevice):
    """
    Voltage-based engine controller
    Receives voltage commands and converts to RPM/thrust for physics simulation
    """
    
    def __init__(self, config: VoltageControllerConfig):
        super().__init__(config)
        self.voltage_config = config
        
        # Engine states
        self.target_voltages = np.zeros(config.num_engines)
        self.current_voltages = np.zeros(config.num_engines)
        self.engine_rpms = np.zeros(config.num_engines)
        self.engine_thrusts = np.zeros(config.num_engines)
        
        # Initialize at hover
        self.target_voltages.fill(config.hover_voltage)
        self.current_voltages.fill(config.hover_voltage)
        
        # Timing
        self.last_update_time = time.time()
        
        # Emergency state
        self.emergency_stop = False
        
        # Import propeller physics for thrust calculation
        try:
            from drone_sim.physics.aerodynamics.propeller import Propeller, PropellerConfig
            from drone_sim.configs.drone_presets.quadcopter_default import get_default_propeller_config
            
            # Create propellers for each engine
            self.propellers = []
            for i in range(config.num_engines):
                prop_config = PropellerConfig(
                    diameter=0.24,
                    pitch=0.12,
                    blades=2,
                    ct_coefficient=1.1e-5,
                    cq_coefficient=1.9e-6
                )
                self.propellers.append(Propeller(prop_config))
        except ImportError:
            # Fallback to simple thrust calculation
            self.propellers = None
    
    def set_engine_voltages(self, voltages: np.ndarray):
        """
        Set target voltages for all engines
        
        Args:
            voltages: Array of voltages [V] for each engine
        """
        if self.emergency_stop:
            return
        
        # Apply safety limits
        voltages = np.clip(voltages, self.voltage_config.min_voltage, self.voltage_config.max_voltage)
        
        # Apply calibration
        calibrated_voltages = voltages * self.voltage_config.voltage_gains + self.voltage_config.voltage_offsets
        
        self.target_voltages = calibrated_voltages
    
    def set_single_engine_voltage(self, engine_id: int, voltage: float):
        """Set voltage for a single engine"""
        if 0 <= engine_id < self.voltage_config.num_engines:
            voltages = self.target_voltages.copy()
            voltages[engine_id] = voltage
            self.set_engine_voltages(voltages)
    
    def emergency_stop_all(self):
        """Emergency stop - cut all voltages to zero"""
        self.emergency_stop = True
        self.target_voltages.fill(0.0)
        self.current_voltages.fill(0.0)
    
    def reset_emergency_stop(self):
        """Reset emergency stop and return to hover"""
        self.emergency_stop = False
        self.target_voltages.fill(self.voltage_config.hover_voltage)
    
    def get_engine_rpms(self) -> np.ndarray:
        """Get current RPM for all engines"""
        return self.engine_rpms.copy()
    
    def get_engine_thrusts(self) -> np.ndarray:
        """Get current thrust for all engines"""
        return self.engine_thrusts.copy()
    
    def get_engine_voltages(self) -> np.ndarray:
        """Get current voltages for all engines"""
        return self.current_voltages.copy()
    
    def get_total_thrust(self) -> float:
        """Get total thrust from all engines"""
        return np.sum(self.engine_thrusts)
    
    # BaseDevice implementation
    def _connect_device(self) -> bool:
        """Connect to voltage controller (always succeeds for simulation)"""
        return True
    
    def _disconnect_device(self):
        """Disconnect voltage controller"""
        self.emergency_stop_all()
    
    def _poll_data(self) -> Dict[str, Any]:
        """Update engine states and return current data"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update voltage with ramping
        self._update_voltages(dt)
        
        # Convert voltage to RPM
        self._voltage_to_rpm()
        
        # Convert RPM to thrust
        self._rpm_to_thrust()
        
        return {
            'voltages': self.current_voltages.copy(),
            'rpms': self.engine_rpms.copy(),
            'thrusts': self.engine_thrusts.copy(),
            'total_thrust': self.get_total_thrust(),
            'emergency_stop': self.emergency_stop,
            'timestamp': current_time
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate engine data"""
        try:
            # Check data structure
            required_keys = ['voltages', 'rpms', 'thrusts', 'total_thrust']
            if not all(key in data for key in required_keys):
                return False
            
            # Check array sizes
            if len(data['voltages']) != self.voltage_config.num_engines:
                return False
            if len(data['rpms']) != self.voltage_config.num_engines:
                return False
            if len(data['thrusts']) != self.voltage_config.num_engines:
                return False
            
            # Check for reasonable values
            voltages = data['voltages']
            if np.any(voltages < 0) or np.any(voltages > self.voltage_config.max_voltage * 1.1):
                return False
            
            rpms = data['rpms']
            if np.any(rpms < 0) or np.any(rpms > 20000):  # Reasonable RPM limit
                return False
            
            thrusts = data['thrusts']
            if np.any(thrusts < 0) or np.any(thrusts > 50):  # Reasonable thrust limit
                return False
            
            return True
            
        except Exception:
            return False
    
    def _update_voltages(self, dt: float):
        """Update current voltages with ramping"""
        # Calculate maximum voltage change this timestep
        max_change = self.voltage_config.voltage_ramp_rate * dt
        
        # Apply voltage ramping
        voltage_diff = self.target_voltages - self.current_voltages
        voltage_change = np.clip(voltage_diff, -max_change, max_change)
        
        self.current_voltages += voltage_change
        
        # Apply final safety limits
        self.current_voltages = np.clip(
            self.current_voltages,
            self.voltage_config.min_voltage,
            self.voltage_config.max_voltage
        )
    
    def _voltage_to_rpm(self):
        """Convert voltage to RPM for each engine"""
        # Simple linear relationship: RPM = voltage * max_rpm_per_volt
        self.engine_rpms = self.current_voltages * self.voltage_config.max_rpm_per_volt
        
        # Ensure non-negative RPM
        self.engine_rpms = np.maximum(self.engine_rpms, 0.0)
    
    def _rpm_to_thrust(self):
        """Convert RPM to thrust using propeller physics"""
        if self.propellers:
            # Use propeller physics
            for i, propeller in enumerate(self.propellers):
                rpm = self.engine_rpms[i]
                thrust = propeller.calculate_thrust(rpm)
                self.engine_thrusts[i] = thrust
        else:
            # Fallback: simple quadratic relationship
            # Thrust = k * RPM^2 (momentum theory approximation)
            k = 1.0e-8  # Thrust coefficient (N⋅s²/rev²)
            self.engine_thrusts = k * self.engine_rpms**2
    
    def get_engine_diagnostics(self) -> Dict[str, Any]:
        """Get detailed engine diagnostics"""
        return {
            'target_voltages': self.target_voltages.tolist(),
            'current_voltages': self.current_voltages.tolist(),
            'engine_rpms': self.engine_rpms.tolist(),
            'engine_thrusts': self.engine_thrusts.tolist(),
            'total_thrust': self.get_total_thrust(),
            'hover_voltage': self.voltage_config.hover_voltage,
            'emergency_stop': self.emergency_stop,
            'voltage_ramp_rate': self.voltage_config.voltage_ramp_rate,
            'max_voltage': self.voltage_config.max_voltage,
            'min_voltage': self.voltage_config.min_voltage
        }
    
    def calibrate_engine(self, engine_id: int, test_voltage: float, expected_thrust: float):
        """
        Calibrate a single engine by adjusting gain/offset
        
        Args:
            engine_id: Engine to calibrate (0-3)
            test_voltage: Voltage to apply for calibration
            expected_thrust: Expected thrust output
        """
        if not (0 <= engine_id < self.voltage_config.num_engines):
            return
        
        # Apply test voltage
        old_voltage = self.target_voltages[engine_id]
        self.set_single_engine_voltage(engine_id, test_voltage)
        
        # Wait for voltage to settle
        time.sleep(0.2)
        
        # Get actual thrust
        data = self.poll()
        if data:
            actual_thrust = data['thrusts'][engine_id]
            
            # Calculate gain adjustment
            if actual_thrust > 0:
                gain_adjustment = expected_thrust / actual_thrust
                self.voltage_config.voltage_gains[engine_id] *= gain_adjustment
        
        # Restore original voltage
        self.set_single_engine_voltage(engine_id, old_voltage) 