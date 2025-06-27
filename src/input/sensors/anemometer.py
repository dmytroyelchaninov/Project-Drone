"""
Anemometer Sensor Implementation
Provides wind speed and direction measurements
"""
import numpy as np
from typing import Dict, Any
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class Anemometer(BaseSensor):
    
    def _connect_device(self) -> bool:
        return self.fake or False
    
    def _disconnect_device(self):
        pass
    
    def _read_sensor(self) -> float:
        if self.fake:
            data = self.get_wind_data()
            return data.get('wind_speed', 0.0)
        return 0.0
    
    def _convert_units(self, raw_value: float) -> float:
        return raw_value
    """
    Anemometer for wind speed and direction measurement
    Useful for flight planning and environmental awareness
    """
    
    def __init__(self, sensor_id: str = "anemometer", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 10.0),
            measurement_range=(0.0, 50.0)  # Wind speed range in m/s
        )
        super().__init__(config)
        self.fake = fake
        
        # Anemometer-specific configuration
        self.update_rate = kwargs.get('update_rate', 10)  # Hz
        self.accuracy = kwargs.get('accuracy', 0.5)  # m/s
        
        # Sensor state
        self.wind_speed = 0.0      # m/s
        self.wind_direction = 0.0  # degrees (0 = North)
        self.wind_velocity = np.zeros(3)  # [vx, vy, vz] m/s
        
        # For simulation
        self.sim_noise_std = kwargs.get('noise_std', 0.2)  # m/s
        
        logger.info(f"Anemometer sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real anemometer hardware"""
        logger.warning("Real anemometer hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated wind data"""
        # Get wind from environment if available
        try:
            from ...physics import Environment, QuadcopterPhysics
            environment = Environment()
            physics = QuadcopterPhysics()
            
            position = physics.state.position
            wind_velocity = environment.get_wind_velocity(position)
            
            self.wind_velocity = wind_velocity
            self.wind_speed = np.linalg.norm(wind_velocity[:2])  # Horizontal wind only
            
            # Calculate wind direction (meteorological convention)
            if self.wind_speed > 0.1:
                # Direction wind is coming FROM (meteorological convention)
                wind_from_x = -wind_velocity[0]  # Negative because "from"
                wind_from_y = -wind_velocity[1]
                self.wind_direction = np.degrees(np.arctan2(wind_from_x, wind_from_y))
                if self.wind_direction < 0:
                    self.wind_direction += 360
            
        except ImportError:
            # No environment available, use static values
            self.wind_speed = 2.0 + np.random.normal(0, 0.5)
            self.wind_direction = 180.0 + np.random.normal(0, 30.0)
        
        # Add noise
        if self.sim_noise_std > 0:
            speed_noise = np.random.normal(0, self.sim_noise_std)
            dir_noise = np.random.normal(0, 5.0)  # 5 degrees direction noise
            
            noisy_speed = max(0, self.wind_speed + speed_noise)
            noisy_direction = self.wind_direction + dir_noise
        else:
            noisy_speed = self.wind_speed
            noisy_direction = self.wind_direction
        
        # Wrap direction
        while noisy_direction < 0:
            noisy_direction += 360
        while noisy_direction >= 360:
            noisy_direction -= 360
        
        # Calculate wind components
        wind_north = noisy_speed * np.cos(np.radians(noisy_direction))
        wind_east = noisy_speed * np.sin(np.radians(noisy_direction))
        
        return {
            'wind_speed': noisy_speed,
            'wind_direction': noisy_direction,
            'wind_velocity': self.wind_velocity,
            'wind_north': wind_north,
            'wind_east': wind_east,
            'wind_vertical': self.wind_velocity[2] if len(self.wind_velocity) > 2 else 0.0,
            'beaufort_scale': self._wind_speed_to_beaufort(noisy_speed),
            'timestamp': time.time()
        }
    
    def _wind_speed_to_beaufort(self, speed: float) -> int:
        """Convert wind speed to Beaufort scale"""
        if speed < 0.3:
            return 0  # Calm
        elif speed < 1.6:
            return 1  # Light air
        elif speed < 3.4:
            return 2  # Light breeze
        elif speed < 5.5:
            return 3  # Gentle breeze
        elif speed < 8.0:
            return 4  # Moderate breeze
        elif speed < 10.8:
            return 5  # Fresh breeze
        elif speed < 13.9:
            return 6  # Strong breeze
        elif speed < 17.2:
            return 7  # Near gale
        elif speed < 20.8:
            return 8  # Gale
        elif speed < 24.5:
            return 9  # Strong gale
        elif speed < 28.5:
            return 10  # Storm
        elif speed < 32.7:
            return 11  # Violent storm
        else:
            return 12  # Hurricane
    
    def get_wind_speed(self) -> float:
        """Get wind speed in m/s"""
        if self.last_data:
            return self.last_data['wind_speed']
        return 0.0
    
    def get_wind_direction(self) -> float:
        """Get wind direction in degrees (meteorological convention)"""
        if self.last_data:
            return self.last_data['wind_direction']
        return 0.0
    
    def get_wind_velocity(self) -> np.ndarray:
        """Get 3D wind velocity vector"""
        if self.last_data:
            return self.last_data['wind_velocity']
        return np.zeros(3)
    
    def is_windy(self, threshold: float = 5.0) -> bool:
        """Check if wind speed exceeds threshold"""
        return self.get_wind_speed() > threshold
