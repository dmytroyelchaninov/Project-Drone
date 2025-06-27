"""
Gyroscope Sensor Implementation
Provides angular velocity measurements for attitude control
"""
import numpy as np
from typing import Dict, Any
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class Gyroscope(BaseSensor):
    
    def _connect_device(self) -> bool:
        return self.fake or False
    
    def _disconnect_device(self):
        pass
    
    def _read_sensor(self) -> float:
        if self.fake:
            rates = self.get_angular_rates()
            return np.linalg.norm(rates)  # Return magnitude as primary reading
        return 0.0
    
    def _convert_units(self, raw_value: float) -> float:
        return raw_value
    """
    3-axis gyroscope for measuring angular velocity
    Critical for flight control and attitude estimation
    """
    
    def __init__(self, sensor_id: str = "gyroscope", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 100.0),
            measurement_range=(-1000.0, 1000.0)  # Angular rate range in deg/s
        )
        super().__init__(config)
        self.fake = fake
        
        # Gyroscope-specific configuration
        self.update_rate = kwargs.get('update_rate', 100)  # Hz
        self.full_scale_range = kwargs.get('full_scale', 2000)  # degrees/second
        
        # Sensor characteristics
        self.noise_std = kwargs.get('noise_std', 0.01)  # rad/s
        self.bias_stability = kwargs.get('bias_stability', 0.001)  # rad/s
        self.resolution = kwargs.get('resolution', 16)  # bits
        
        # Current measurements
        self.angular_velocity = np.zeros(3)  # [roll_rate, pitch_rate, yaw_rate] rad/s
        self.temperature = 25.0  # °C
        
        # Calibration parameters
        self.bias_offset = np.zeros(3)  # rad/s
        self.scale_factor = np.ones(3)  # correction factors
        self.cross_coupling = np.eye(3)  # cross-axis coupling matrix
        
        # Temperature compensation
        self.temp_coefficient = np.zeros(3)  # rad/s/°C
        self.reference_temperature = 25.0  # °C
        
        # Internal state for simulation
        self.sim_bias_drift = np.zeros(3)
        self.last_update_time = time.time()
        
        logger.info(f"Gyroscope sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real gyroscope hardware"""
        # This would interface with actual gyroscope hardware (e.g., MPU6050, LSM6DS3)
        logger.warning("Real gyroscope hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated gyroscope data"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Get angular velocity from physics simulation if available
        try:
            from ...physics import QuadcopterPhysics
            physics = QuadcopterPhysics()
            true_angular_velocity = physics.state.angular_velocity  # rad/s in body frame
            
            # Convert from physics convention to sensor convention if needed
            self.angular_velocity = true_angular_velocity.copy()
            
        except ImportError:
            # No physics available, use static values
            self.angular_velocity = np.zeros(3)
        
        # Apply sensor imperfections
        measured_rates = self._apply_sensor_model(self.angular_velocity, dt)
        
        # Convert to degrees for output
        rates_deg = np.degrees(measured_rates)
        
        # Simulate temperature reading
        self.temperature = 25.0 + np.random.normal(0, 2.0)
        
        return {
            'angular_velocity_rad': measured_rates,  # rad/s
            'angular_velocity_deg': rates_deg,       # deg/s
            'roll_rate': measured_rates[0],          # rad/s
            'pitch_rate': measured_rates[1],         # rad/s
            'yaw_rate': measured_rates[2],           # rad/s
            'roll_rate_deg': rates_deg[0],           # deg/s
            'pitch_rate_deg': rates_deg[1],          # deg/s
            'yaw_rate_deg': rates_deg[2],            # deg/s
            'temperature': self.temperature,         # °C
            'timestamp': current_time,
            'scale_factor': self.scale_factor.copy(),
            'bias_estimate': self.bias_offset.copy()
        }
    
    def _apply_sensor_model(self, true_rates: np.ndarray, dt: float) -> np.ndarray:
        """Apply realistic sensor imperfections"""
        # Start with true rates
        measured_rates = true_rates.copy()
        
        # Apply bias drift (random walk)
        if dt > 0:
            bias_drift_rate = 0.001  # rad/s per sqrt(s)
            self.sim_bias_drift += np.random.normal(0, bias_drift_rate * np.sqrt(dt), 3)
        
        # Apply temperature compensation
        temp_error = (self.temperature - self.reference_temperature) * self.temp_coefficient
        
        # Apply calibration corrections
        measured_rates = measured_rates * self.scale_factor + self.bias_offset + temp_error + self.sim_bias_drift
        
        # Apply cross-coupling
        measured_rates = self.cross_coupling @ measured_rates
        
        # Add white noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, 3)
            measured_rates += noise
        
        # Apply range limiting
        max_rate = np.radians(self.full_scale_range)
        measured_rates = np.clip(measured_rates, -max_rate, max_rate)
        
        # Apply quantization (ADC resolution)
        if self.resolution > 0:
            levels = 2 ** self.resolution
            rate_range = 2 * max_rate
            quantization = rate_range / levels
            measured_rates = np.round(measured_rates / quantization) * quantization
        
        return measured_rates
    
    def calibrate_bias(self, num_samples: int = 1000, duration: float = 10.0):
        """
        Calibrate gyroscope bias while stationary
        
        Args:
            num_samples: Number of samples to average
            duration: Calibration duration in seconds
        """
        logger.info(f"Starting gyroscope bias calibration for {duration}s...")
        
        samples = []
        start_time = time.time()
        
        while len(samples) < num_samples and (time.time() - start_time) < duration:
            data = self.poll()
            if data:
                samples.append(data['angular_velocity_rad'])
            time.sleep(0.01)  # 100 Hz sampling
        
        if samples:
            # Calculate average bias
            bias_measurements = np.array(samples)
            self.bias_offset = np.mean(bias_measurements, axis=0)
            
            # Calculate noise characteristics
            noise_std = np.std(bias_measurements, axis=0)
            
            logger.info(f"Gyroscope calibration complete:")
            logger.info(f"  Bias offset: {np.degrees(self.bias_offset)} deg/s")
            logger.info(f"  Noise std: {np.degrees(noise_std)} deg/s")
            
            return {
                'bias_offset': self.bias_offset,
                'noise_std': noise_std,
                'samples_used': len(samples)
            }
        else:
            logger.error("Gyroscope calibration failed - no data collected")
            return None
    
    def set_bias_offset(self, bias: np.ndarray):
        """Set bias offset manually"""
        self.bias_offset = bias.copy()
        logger.info(f"Gyroscope bias set to {np.degrees(bias)} deg/s")
    
    def set_scale_factor(self, scale: np.ndarray):
        """Set scale factor corrections"""
        self.scale_factor = scale.copy()
        logger.info(f"Gyroscope scale factors set to {scale}")
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get current angular velocity in rad/s"""
        if self.last_data:
            return self.last_data['angular_velocity_rad']
        return np.zeros(3)
    
    def get_angular_velocity_deg(self) -> np.ndarray:
        """Get current angular velocity in deg/s"""
        if self.last_data:
            return self.last_data['angular_velocity_deg']
        return np.zeros(3)
    
    def get_temperature(self) -> float:
        """Get sensor temperature"""
        if self.last_data:
            return self.last_data['temperature']
        return 25.0
    
    def is_rotating(self, threshold: float = 0.1) -> bool:
        """Check if significant rotation is detected"""
        rates = self.get_angular_velocity()
        return np.linalg.norm(rates) > threshold
    
    def get_rotation_magnitude(self) -> float:
        """Get magnitude of total rotation rate"""
        rates = self.get_angular_velocity()
        return np.linalg.norm(rates)
    
    def integrate_orientation(self, dt: float, initial_orientation: np.ndarray = None) -> np.ndarray:
        """
        Simple integration to estimate orientation change
        Note: This is prone to drift - use complementary filter in practice
        
        Args:
            dt: Time step
            initial_orientation: Initial Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            Updated Euler angles [roll, pitch, yaw] in radians
        """
        if initial_orientation is None:
            initial_orientation = np.zeros(3)
        
        rates = self.get_angular_velocity()
        
        # Simple Euler integration (not accurate for large angles)
        new_orientation = initial_orientation + rates * dt
        
        # Wrap angles
        new_orientation[2] = np.arctan2(np.sin(new_orientation[2]), np.cos(new_orientation[2]))
        
        return new_orientation
