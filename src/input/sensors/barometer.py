"""
Barometer Sensor Implementation
Provides altitude data based on atmospheric pressure
"""
import numpy as np
from typing import Dict, Any
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class Barometer(BaseSensor):
    
    def _connect_device(self) -> bool:
        return self.fake or False
    
    def _disconnect_device(self):
        pass
    
    def _read_sensor(self) -> float:
        if self.fake:
            return self.get_pressure()
        return 101325.0  # Standard sea level pressure
    
    def _convert_units(self, raw_value: float) -> float:
        return raw_value
    """
    Barometric pressure sensor for altitude measurement
    More accurate than GPS altitude for relative altitude changes
    """
    
    def __init__(self, sensor_id: str = "barometer", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 50.0),
            measurement_range=(80000.0, 120000.0)  # Pressure range in Pa
        )
        super().__init__(config)
        self.fake = fake
        
        # Barometer-specific configuration
        self.accuracy = kwargs.get('accuracy', 0.1)  # meters
        self.update_rate = kwargs.get('update_rate', 50)  # Hz
        
        # Atmospheric constants
        self.sea_level_pressure = 101325.0  # Pa
        self.temperature = 288.15  # K (15°C)
        self.gas_constant = 287.058  # J/(kg·K)
        self.gravity = 9.81  # m/s^2
        
        # Sensor state
        self.pressure = self.sea_level_pressure  # Pa
        self.altitude = 0.0  # meters
        self.temperature_reading = self.temperature  # K
        self.reference_pressure = self.sea_level_pressure
        
        # Calibration
        self.pressure_offset = 0.0  # Pa
        self.altitude_offset = 0.0  # meters
        
        # For simulation
        self.sim_noise_std = kwargs.get('noise_std', 0.05)  # meters
        
        logger.info(f"Barometer sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real barometer hardware"""
        # This would interface with actual barometer hardware (e.g., MS5611, BMP280)
        logger.warning("Real barometer hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated barometer data"""
        # Get altitude from physics simulation if available
        try:
            from ...physics import QuadcopterPhysics
            physics = QuadcopterPhysics()
            actual_altitude = physics.state.position[2]
            
            # Calculate pressure from altitude using barometric formula
            # p = p₀ * (1 - (L*h)/(T₀))^(g*M/(R*L))
            # Simplified: p = p₀ * exp(-h/H) where H is scale height
            scale_height = 8400  # meters (approximate)
            self.pressure = self.reference_pressure * np.exp(-actual_altitude / scale_height)
            
            # Calculate altitude from pressure
            self.altitude = -scale_height * np.log(self.pressure / self.reference_pressure)
            
        except ImportError:
            # No physics available, use static values
            pass
        
        # Add noise
        if self.sim_noise_std > 0:
            altitude_noise = np.random.normal(0, self.sim_noise_std)
            noisy_altitude = self.altitude + altitude_noise
            
            # Convert altitude noise back to pressure noise
            pressure_noise = self.reference_pressure * np.exp(-noisy_altitude / 8400) - self.pressure
            noisy_pressure = self.pressure + pressure_noise
        else:
            noisy_altitude = self.altitude
            noisy_pressure = self.pressure
        
        # Apply calibration offsets
        calibrated_pressure = noisy_pressure + self.pressure_offset
        calibrated_altitude = noisy_altitude + self.altitude_offset
        
        # Simulate temperature reading (affects pressure calculation)
        temp_variation = np.random.normal(0, 1.0)  # ±1K variation
        self.temperature_reading = self.temperature + temp_variation
        
        return {
            'pressure': calibrated_pressure,
            'altitude': calibrated_altitude,
            'temperature': self.temperature_reading,
            'pressure_raw': noisy_pressure,
            'altitude_msl': calibrated_altitude,  # Mean sea level
            'altitude_agl': max(0, calibrated_altitude),  # Above ground level
            'vertical_speed': self._calculate_vertical_speed(),
            'timestamp': time.time()
        }
    
    def _calculate_vertical_speed(self) -> float:
        """Calculate vertical speed from pressure changes"""
        if len(self.data_history) < 2:
            return 0.0
        
        # Get last two altitude readings
        current_alt = self.data_history[-1].get('altitude', 0)
        previous_alt = self.data_history[-2].get('altitude', 0)
        
        # Calculate time difference
        current_time = self.data_history[-1].get('timestamp', time.time())
        previous_time = self.data_history[-2].get('timestamp', time.time())
        dt = current_time - previous_time
        
        if dt > 0:
            return (current_alt - previous_alt) / dt
        return 0.0
    
    def set_reference_pressure(self, pressure: float):
        """Set reference pressure for altitude calculations"""
        self.reference_pressure = pressure
        logger.info(f"Barometer reference pressure set to {pressure} Pa")
    
    def set_sea_level_pressure(self, pressure: float):
        """Set sea level pressure for absolute altitude"""
        self.sea_level_pressure = pressure
        self.reference_pressure = pressure
    
    def calibrate_altitude(self, known_altitude: float):
        """Calibrate sensor to known altitude"""
        if self.last_data:
            current_reading = self.last_data['altitude']
            self.altitude_offset = known_altitude - current_reading
            logger.info(f"Barometer calibrated: offset = {self.altitude_offset:.2f} m")
    
    def get_altitude_msl(self) -> float:
        """Get altitude above mean sea level"""
        if self.last_data:
            return self.last_data['altitude_msl']
        return 0.0
    
    def get_altitude_agl(self, ground_elevation: float = 0.0) -> float:
        """Get altitude above ground level"""
        msl_altitude = self.get_altitude_msl()
        return max(0, msl_altitude - ground_elevation)
    
    def get_pressure(self) -> float:
        """Get current pressure reading"""
        if self.last_data:
            return self.last_data['pressure']
        return self.sea_level_pressure
    
    def get_vertical_speed(self) -> float:
        """Get current vertical speed"""
        if self.last_data:
            return self.last_data.get('vertical_speed', 0.0)
        return 0.0
    
    def get_density_altitude(self) -> float:
        """Calculate density altitude for performance calculations"""
        if not self.last_data:
            return 0.0
        
        pressure = self.last_data['pressure']
        temperature = self.last_data['temperature']
        
        # Density altitude calculation
        standard_temp = 288.15  # K
        standard_pressure = 101325.0  # Pa
        
        # ISA temperature at altitude
        altitude = self.last_data['altitude']
        isa_temp = standard_temp - 0.0065 * altitude
        
        # Density altitude formula
        density_altitude = altitude + (isa_temp / 0.0065) * (
            1 - (pressure / standard_pressure) ** 0.190284
        )
        
        return density_altitude
