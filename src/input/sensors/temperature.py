"""
Temperature Sensor Implementation
Provides ambient temperature measurements
"""
import numpy as np
from typing import Dict, Any
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class Temperature(BaseSensor):
    
    def _connect_device(self) -> bool:
        return self.fake or False
    
    def _disconnect_device(self):
        pass
    
    def _read_sensor(self) -> float:
        if self.fake:
            return self.get_temperature()
        return 15.0  # Default temperature
    
    def _convert_units(self, raw_value: float) -> float:
        return raw_value
    """
    Temperature sensor for environmental monitoring
    Used for calibration and environmental awareness
    """
    
    def __init__(self, sensor_id: str = "temperature", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 10.0),
            measurement_range=(0.0, 50.0)  # Temperature range in Celsius
        )
        super().__init__(config)
        self.fake = fake
        
        # Temperature-specific configuration
        self.accuracy = kwargs.get('accuracy', 0.5)  # Kelvin
        self.update_rate = kwargs.get('update_rate', 1)  # Hz
        
        # Sensor state
        self.temperature = 288.15  # K (15°C)
        self.humidity = 50.0      # % relative humidity (if available)
        
        # For simulation
        self.sim_noise_std = kwargs.get('noise_std', 0.2)  # Kelvin
        self.base_temperature = 288.15  # K
        
        logger.info(f"Temperature sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real temperature hardware"""
        logger.warning("Real temperature hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated temperature data"""
        # Get altitude from physics if available for temperature lapse
        try:
            from ...physics import QuadcopterPhysics, Environment
            physics = QuadcopterPhysics()
            environment = Environment()
            
            altitude = physics.state.position[2]
            env_info = environment.get_atmospheric_properties(altitude)
            self.temperature = env_info['temperature']
            
        except ImportError:
            # Use standard atmosphere model
            # Temperature decreases by 6.5°C per 1000m
            altitude = 0  # Default ground level
            self.temperature = self.base_temperature - 0.0065 * altitude
        
        # Add noise
        if self.sim_noise_std > 0:
            noise = np.random.normal(0, self.sim_noise_std)
            noisy_temp = self.temperature + noise
        else:
            noisy_temp = self.temperature
        
        # Simulate humidity (simple model)
        self.humidity = 50.0 + np.random.normal(0, 10.0)
        self.humidity = np.clip(self.humidity, 0, 100)
        
        return {
            'temperature_k': noisy_temp,
            'temperature_c': noisy_temp - 273.15,
            'temperature_f': (noisy_temp - 273.15) * 9/5 + 32,
            'humidity': self.humidity,
            'timestamp': time.time()
        }
    
    def get_temperature_celsius(self) -> float:
        """Get temperature in Celsius"""
        if self.last_data:
            return self.last_data['temperature_c']
        return 15.0
    
    def get_temperature_kelvin(self) -> float:
        """Get temperature in Kelvin"""
        if self.last_data:
            return self.last_data['temperature_k']
        return 288.15
    
    def get_humidity(self) -> float:
        """Get relative humidity percentage"""
        if self.last_data:
            return self.last_data.get('humidity', 50.0)
        return 50.0
