"""
Base sensor classes for drone sensor management
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import abstractmethod

from ..devices.base_device import BaseDevice, DeviceConfig

class SensorConfig:
    """Base configuration for sensors"""
    def __init__(self, name: str = None, sensor_id: str = None, fake: bool = True,
                 poll_rate: float = 10.0, timeout: float = 1.0, enabled: bool = True,
                 precision: int = 2, calibration: Dict[str, float] = None,
                 measurement_range: tuple = (0.0, 100.0), resolution: float = 0.01,
                 accuracy: float = 0.05, noise_enabled: bool = True, noise_std: float = 0.01,
                 bias_drift: float = 0.001, filter_enabled: bool = True,
                 filter_cutoff_freq: float = 10.0, **kwargs):
        self.name = name or sensor_id or "unnamed_sensor"
        self.sensor_id = sensor_id or name or "unnamed_sensor"
        self.fake = fake
        self.poll_rate = poll_rate
        self.timeout = timeout
        self.enabled = enabled
        self.precision = precision
        self.calibration = calibration or {}
        self.measurement_range = measurement_range
        self.resolution = resolution
        self.accuracy = accuracy
        self.noise_enabled = noise_enabled
        self.noise_std = noise_std
        self.bias_drift = bias_drift
        self.filter_enabled = filter_enabled
        self.filter_cutoff_freq = filter_cutoff_freq
        
        # Store any additional kwargs for sensor-specific config
        for key, value in kwargs.items():
            setattr(self, key, value)

class BaseSensor(BaseDevice):
    """
    Base class for all drone sensors
    Extends BaseDevice with sensor-specific functionality
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.sensor_config = config
        
        # Sensor state
        self.raw_measurement: Optional[float] = None
        self.filtered_measurement: Optional[float] = None
        self.measurement_history: list = []
        self.bias = 0.0
        self.last_filter_time = 0.0
        
        # Filter state (simple low-pass)
        self.filter_state = 0.0
    
    @abstractmethod
    def _read_sensor(self) -> float:
        """Read raw sensor value. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _convert_units(self, raw_value: float) -> float:
        """Convert raw sensor reading to meaningful units."""
        pass
    
    def _poll_data(self) -> Dict[str, Any]:
        """Poll sensor data with filtering and noise simulation"""
        import time
        current_time = time.time()
        
        # Read raw sensor
        raw_value = self._read_sensor()
        
        # Convert to meaningful units
        converted_value = self._convert_units(raw_value)
        
        # Apply sensor characteristics
        measurement = self._apply_sensor_characteristics(converted_value, current_time)
        
        # Apply filtering
        if self.sensor_config.filter_enabled:
            filtered_value = self._apply_filter(measurement, current_time)
        else:
            filtered_value = measurement
        
        # Store measurements
        self.raw_measurement = raw_value
        self.filtered_measurement = filtered_value
        
        # Update history (keep last 100 measurements)
        self.measurement_history.append({
            'timestamp': current_time,
            'raw': raw_value,
            'converted': converted_value,
            'measurement': measurement,
            'filtered': filtered_value
        })
        if len(self.measurement_history) > 100:
            self.measurement_history.pop(0)
        
        return {
            'raw_value': raw_value,
            'measurement': measurement,
            'filtered_value': filtered_value,
            'bias': self.bias,
            'timestamp': current_time,
            'sensor_name': self.config.name
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate sensor data"""
        try:
            # Check required fields
            if 'measurement' not in data or 'filtered_value' not in data:
                return False
            
            # Check measurement range
            measurement = data['measurement']
            min_val, max_val = self.sensor_config.measurement_range
            if not (min_val <= measurement <= max_val * 2):  # Allow some overage
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(measurement):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_sensor_characteristics(self, true_value: float, current_time: float) -> float:
        """Apply sensor noise, bias, and other characteristics"""
        # Update bias drift
        if hasattr(self, '_last_bias_update'):
            dt = current_time - self._last_bias_update
            self.bias += np.random.normal(0, self.sensor_config.bias_drift * dt)
        self._last_bias_update = current_time
        
        # Apply bias
        biased_value = true_value + self.bias
        
        # Apply noise
        if self.sensor_config.noise_enabled:
            noise = np.random.normal(0, self.sensor_config.noise_std)
            noisy_value = biased_value + noise
        else:
            noisy_value = biased_value
        
        # Apply resolution quantization
        if self.sensor_config.resolution > 0:
            quantized_value = np.round(noisy_value / self.sensor_config.resolution) * self.sensor_config.resolution
        else:
            quantized_value = noisy_value
        
        return quantized_value
    
    def _apply_filter(self, measurement: float, current_time: float) -> float:
        """Apply low-pass filter to measurement"""
        if self.last_filter_time == 0:
            self.filter_state = measurement
            self.last_filter_time = current_time
            return measurement
        
        dt = current_time - self.last_filter_time
        self.last_filter_time = current_time
        
        # Simple first-order low-pass filter
        cutoff = self.sensor_config.filter_cutoff_freq
        alpha = dt / (dt + 1.0 / (2 * np.pi * cutoff))
        
        self.filter_state = alpha * measurement + (1 - alpha) * self.filter_state
        return self.filter_state
    
    def get_measurement(self) -> Optional[float]:
        """Get latest filtered measurement"""
        return self.filtered_measurement
    
    def get_raw_measurement(self) -> Optional[float]:
        """Get latest raw measurement"""
        return self.raw_measurement
    
    def calibrate(self, known_value: float, num_samples: int = 100):
        """
        Calibrate sensor against a known value
        
        Args:
            known_value: The true value being measured
            num_samples: Number of samples to average for calibration
        """
        measurements = []
        
        for _ in range(num_samples):
            data = self.poll()
            if data and 'measurement' in data:
                measurements.append(data['measurement'])
        
        if measurements:
            average_measurement = np.mean(measurements)
            self.bias = known_value - average_measurement
            return True
        
        return False
    
    def get_sensor_stats(self) -> Dict[str, Any]:
        """Get sensor statistics"""
        stats = self.get_stats()
        
        if self.measurement_history:
            measurements = [h['measurement'] for h in self.measurement_history]
            stats.update({
                'measurement_count': len(measurements),
                'mean_measurement': np.mean(measurements),
                'std_measurement': np.std(measurements),
                'min_measurement': np.min(measurements),
                'max_measurement': np.max(measurements),
                'current_bias': self.bias
            })
        
        return stats 