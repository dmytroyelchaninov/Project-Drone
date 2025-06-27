"""
GPS Sensor Implementation
Provides GPS position data (real or simulated)
"""
import numpy as np
from typing import Dict, Any, Optional
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class GPS(BaseSensor):
    """
    GPS sensor for position and altitude data
    Provides latitude, longitude, and altitude information
    """
    
    def _connect_device(self) -> bool:
        """Connect to GPS device (or simulate connection)"""
        if self.fake:
            return True
        else:
            # Real GPS hardware connection would go here
            logger.warning("Real GPS hardware not implemented")
            return False
    
    def _disconnect_device(self):
        """Disconnect from GPS device"""
        pass
    
    def _read_sensor(self) -> float:
        """Read raw sensor value"""
        # For GPS, we return latitude as the primary reading
        if self.fake:
            data = self._poll_fake()
            return data.get('latitude', 0.0) if data else 0.0
        else:
            data = self._poll_real()
            return data.get('latitude', 0.0) if data else 0.0
    
    def _convert_units(self, raw_value: float) -> float:
        """Convert raw value to meaningful units"""
        return raw_value  # GPS already returns degrees
    
    def __init__(self, sensor_id: str = "gps", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 10.0),
            measurement_range=(-90.0, 90.0)  # Latitude range
        )
        super().__init__(config)
        self.fake = fake
        
        # GPS-specific configuration
        self.accuracy = kwargs.get('accuracy', 3.0)  # meters
        self.update_rate = kwargs.get('update_rate', 10)  # Hz
        
        # GPS state
        self.latitude = 0.0      # degrees
        self.longitude = 0.0     # degrees
        self.altitude = 0.0      # meters above sea level
        self.speed = 0.0         # m/s ground speed
        self.heading = 0.0       # degrees
        self.satellites = 8      # number of satellites
        self.hdop = 1.0         # horizontal dilution of precision
        
        # For simulation
        self.sim_noise_std = kwargs.get('noise_std', 0.5)  # meters
        self.last_physics_position = np.zeros(3)
        
        logger.info(f"GPS sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real GPS hardware"""
        # This would interface with actual GPS hardware
        # For now, return None to indicate no real hardware available
        logger.warning("Real GPS hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated GPS data"""
        # Get position from physics simulation if available
        try:
            from ...physics import QuadcopterPhysics
            physics = QuadcopterPhysics()
            position = physics.state.position
            velocity = physics.state.velocity
            
            # Convert physics position to GPS coordinates
            # This is a simple conversion - in reality, you'd need proper coordinate transforms
            lat_offset = position[1] / 111320.0  # Approximate meters to degrees latitude
            lon_offset = position[0] / (111320.0 * np.cos(np.radians(self.latitude)))
            
            self.latitude += lat_offset
            self.longitude += lon_offset
            self.altitude = position[2]  # Direct altitude
            
            # Calculate speed and heading
            ground_velocity = velocity[:2]  # X, Y components
            self.speed = np.linalg.norm(ground_velocity)
            
            if self.speed > 0.1:  # Only update heading if moving
                self.heading = np.degrees(np.arctan2(velocity[0], velocity[1]))
                if self.heading < 0:
                    self.heading += 360
            
            # Store for next iteration
            self.last_physics_position = position.copy()
            
        except ImportError:
            # No physics available, use static position
            pass
        
        # Add GPS noise
        if self.sim_noise_std > 0:
            noise = np.random.normal(0, self.sim_noise_std, 3)
            lat_noise = noise[0] / 111320.0
            lon_noise = noise[1] / (111320.0 * np.cos(np.radians(self.latitude)))
            alt_noise = noise[2]
            
            noisy_lat = self.latitude + lat_noise
            noisy_lon = self.longitude + lon_noise
            noisy_alt = self.altitude + alt_noise
        else:
            noisy_lat = self.latitude
            noisy_lon = self.longitude
            noisy_alt = self.altitude
        
        # Simulate GPS quality indicators
        self.satellites = max(4, min(12, self.satellites + np.random.randint(-1, 2)))
        self.hdop = max(0.5, min(5.0, self.hdop + np.random.normal(0, 0.1)))
        
        return {
            'latitude': noisy_lat,
            'longitude': noisy_lon,
            'altitude': noisy_alt,
            'speed': self.speed,
            'heading': self.heading,
            'satellites': self.satellites,
            'hdop': self.hdop,
            'fix_quality': 'GPS' if self.satellites >= 4 else 'No Fix',
            'timestamp': time.time()
        }
    
    def get_position_lla(self) -> np.ndarray:
        """Get position as [latitude, longitude, altitude]"""
        if self.last_data:
            return np.array([
                self.last_data['latitude'],
                self.last_data['longitude'],
                self.last_data['altitude']
            ])
        return np.zeros(3)
    
    def get_position_ned(self, reference_lla: np.ndarray = None) -> np.ndarray:
        """
        Get position in North-East-Down frame relative to reference
        
        Args:
            reference_lla: Reference position [lat, lon, alt] in degrees/meters
            
        Returns:
            Position in NED frame [north, east, down] in meters
        """
        if not self.last_data or reference_lla is None:
            return np.zeros(3)
        
        # Simple flat-earth approximation
        dlat = self.last_data['latitude'] - reference_lla[0]
        dlon = self.last_data['longitude'] - reference_lla[1]
        dalt = self.last_data['altitude'] - reference_lla[2]
        
        # Convert to meters
        north = dlat * 111320.0  # degrees to meters
        east = dlon * 111320.0 * np.cos(np.radians(reference_lla[0]))
        down = -dalt  # GPS altitude is positive up, NED is positive down
        
        return np.array([north, east, down])
    
    def get_velocity_ned(self) -> np.ndarray:
        """Get velocity in North-East-Down frame"""
        if not self.last_data:
            return np.zeros(3)
        
        speed = self.last_data['speed']
        heading_rad = np.radians(self.last_data['heading'])
        
        # Convert speed and heading to NED velocity
        north_vel = speed * np.cos(heading_rad)
        east_vel = speed * np.sin(heading_rad)
        down_vel = 0.0  # GPS doesn't typically provide vertical velocity
        
        return np.array([north_vel, east_vel, down_vel])
    
    def set_reference_position(self, latitude: float, longitude: float, altitude: float):
        """Set reference position for simulation"""
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
    
    def get_quality_info(self) -> Dict[str, Any]:
        """Get GPS quality and status information"""
        if not self.last_data:
            return {}
        
        return {
            'satellites': self.last_data.get('satellites', 0),
            'hdop': self.last_data.get('hdop', 99.0),
            'fix_quality': self.last_data.get('fix_quality', 'No Fix'),
            'accuracy_estimate': self.accuracy,
            'signal_strength': min(100, max(0, (self.satellites - 4) * 20))
        }
