"""
Compass/Magnetometer Sensor Implementation
Provides magnetic heading and magnetic field measurements
"""
import numpy as np
from typing import Dict, Any
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class Compass(BaseSensor):
    
    def _connect_device(self) -> bool:
        return self.fake or False
    
    def _disconnect_device(self):
        pass
    
    def _read_sensor(self) -> float:
        if self.fake:
            data = self.get_heading()
            return data.get('heading', 0.0)
        return 0.0
    
    def _convert_units(self, raw_value: float) -> float:
        return raw_value
    """
    3-axis magnetometer/compass for heading determination
    Critical for navigation and attitude reference
    """
    
    def __init__(self, sensor_id: str = "compass", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 20.0),
            measurement_range=(0.0, 360.0)  # Heading range in degrees
        )
        super().__init__(config)
        self.fake = fake
        
        # Compass-specific configuration
        self.update_rate = kwargs.get('update_rate', 50)  # Hz
        self.declination = kwargs.get('declination', 0.0)  # degrees
        
        # Sensor characteristics
        self.noise_std = kwargs.get('noise_std', 0.1)  # degrees
        self.resolution = kwargs.get('resolution', 0.1)  # degrees
        
        # Current measurements
        self.magnetic_field = np.array([0.5, 0.0, 0.5])  # Normalized magnetic field vector
        self.heading = 0.0  # degrees (0 = North)
        self.inclination = 60.0  # degrees (typical for northern hemisphere)
        
        # Calibration parameters
        self.hard_iron_offset = np.zeros(3)  # Offset calibration
        self.soft_iron_matrix = np.eye(3)    # Scale/rotation calibration
        
        # Earth's magnetic field (typical values)
        self.earth_field_strength = 50000  # nanoTesla
        self.earth_field_inclination = 60.0  # degrees
        
        logger.info(f"Compass sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real magnetometer hardware"""
        logger.warning("Real magnetometer hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated compass data"""
        # Get orientation from physics simulation if available
        try:
            from ...physics import QuadcopterPhysics
            physics = QuadcopterPhysics()
            euler_angles = physics.get_euler_angles()
            roll, pitch, yaw = euler_angles
            
            # Calculate expected magnetic field in body frame
            # Earth's magnetic field vector in NED frame
            mag_ned = np.array([
                np.cos(np.radians(self.earth_field_inclination)),
                0.0,
                np.sin(np.radians(self.earth_field_inclination))
            ]) * self.earth_field_strength
            
            # Rotate to body frame using rotation matrix
            R_body_to_ned = self._euler_to_rotation_matrix(roll, pitch, yaw)
            mag_body = R_body_to_ned.T @ mag_ned
            
            # Normalize
            self.magnetic_field = mag_body / np.linalg.norm(mag_body)
            
            # Calculate heading from magnetic field
            self.heading = np.degrees(np.arctan2(-mag_body[1], mag_body[0]))
            if self.heading < 0:
                self.heading += 360
                
            # Apply declination correction
            self.heading += self.declination
            if self.heading >= 360:
                self.heading -= 360
                
        except ImportError:
            # No physics available, use fixed heading
            pass
        
        # Apply calibration
        calibrated_field = self._apply_calibration(self.magnetic_field)
        
        # Add noise
        if self.noise_std > 0:
            heading_noise = np.random.normal(0, self.noise_std)
            field_noise = np.random.normal(0, 0.01, 3)  # Small field noise
            
            noisy_heading = self.heading + heading_noise
            noisy_field = calibrated_field + field_noise
        else:
            noisy_heading = self.heading
            noisy_field = calibrated_field
        
        # Wrap heading
        while noisy_heading < 0:
            noisy_heading += 360
        while noisy_heading >= 360:
            noisy_heading -= 360
        
        # Calculate field strength
        field_strength = np.linalg.norm(noisy_field) * self.earth_field_strength
        
        return {
            'heading': noisy_heading,
            'magnetic_field': noisy_field,
            'magnetic_field_raw': self.magnetic_field,
            'field_strength': field_strength,
            'declination': self.declination,
            'inclination': self.inclination,
            'timestamp': time.time()
        }
    
    def _apply_calibration(self, raw_field: np.ndarray) -> np.ndarray:
        """Apply hard and soft iron calibration"""
        # Remove hard iron offset
        corrected = raw_field - self.hard_iron_offset
        
        # Apply soft iron correction matrix
        calibrated = self.soft_iron_matrix @ corrected
        
        return calibrated
    
    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        return np.array([
            [cp*cy, -cp*sy, sp],
            [sr*sp*cy + cr*sy, -sr*sp*sy + cr*cy, -sr*cp],
            [-cr*sp*cy + sr*sy, cr*sp*sy + sr*cy, cr*cp]
        ])
    
    def get_heading(self) -> float:
        """Get current magnetic heading in degrees"""
        if self.last_data:
            return self.last_data['heading']
        return 0.0
    
    def get_magnetic_field(self) -> np.ndarray:
        """Get calibrated magnetic field vector"""
        if self.last_data:
            return self.last_data['magnetic_field']
        return np.array([1.0, 0.0, 0.0])
    
    def set_declination(self, declination: float):
        """Set magnetic declination for location"""
        self.declination = declination
        logger.info(f"Compass declination set to {declination}°")
    
    def calibrate_hard_iron(self, num_samples: int = 1000):
        """
        Calibrate hard iron offset by rotating sensor in all directions
        """
        logger.info(f"Starting compass hard iron calibration...")
        logger.info("Rotate the drone slowly in all directions for best results")
        
        samples = []
        start_time = time.time()
        
        while len(samples) < num_samples:
            data = self.poll()
            if data:
                samples.append(data['magnetic_field_raw'])
            time.sleep(0.01)
        
        if samples:
            field_samples = np.array(samples)
            
            # Hard iron offset is the center of the sphere
            self.hard_iron_offset = np.mean(field_samples, axis=0)
            
            logger.info(f"Hard iron calibration complete:")
            logger.info(f"  Offset: {self.hard_iron_offset}")
            
            return {
                'hard_iron_offset': self.hard_iron_offset,
                'samples_used': len(samples)
            }
        else:
            logger.error("Compass calibration failed")
            return None 