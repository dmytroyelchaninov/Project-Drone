"""
Hub Module
SINGLETON class that serves as the central data hub for the drone system
Manages all input/output data, device states, and system configuration
"""
import threading
import time
import logging
from typing import Dict, Any, List, Optional
from cfg import settings
from .devices import BaseDevice, KeyboardDevice, VoltageController
from .sensors import *

logger = logging.getLogger(__name__)

class Hub:
    """
    SINGLETON Hub class - central data storage and coordination
    
    Serves as the independent data storage that's the "database" of the system.
    Collects input data from external devices and sensors.
    Stores output data from drone (voltage for each engine).
    
    All updates happen "in place" and asynchronously in real-time.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # System configuration
            self.settings = settings
            self.simulation = self.settings.SIMULATION
            self.frequency = self.settings.get('GENERAL.POLL_FREQUENCY', 100)
            
            # Device configuration
            device_config = self.settings.CURRENT_DEVICE
            self.device = {
                'type': device_config.get('type', 'keyboard'),
                'obj': None  # Will be initialized later
            }
            
            # Sensors configuration
            sensors_config = self.settings.AVAILABLE_SENSORS
            self.sensors = {}
            self._sensors = {}  # Actual sensor objects
            
            # Initialize sensor configurations
            for sensor_name, sensor_config in sensors_config.items():
                if sensor_config.get('enabled', False):
                    self.sensors[sensor_name] = {
                        'fake': sensor_config.get('fake', True),
                        'obj': None  # Will be initialized later
                    }
            
            # Input data storage (our "database")
            self.input = {
                'gps': None,
                'barometer': None,
                'gyroscope': None,
                'temperature': None,
                'anemometer': None,
                'compass': None,
                'cameras': None,
                'device_voltages': None,
                'device_status': None
            }
            
            # Output data (voltage for each engine)
            self.output = [0.0, 0.0, 0.0, 0.0]  # 4 engines, initialized to 0V
            
            # Control state
            control_config = self.settings.CONTROL
            self.task = control_config.get('default_task')  # None, take_off, land, follow, back_to_base, projectile
            self.mode = control_config.get('default_mode', 'manual')  # ai, hybrid, manual
            self.go = control_config.get('default_go', 'idle')  # off, operate, idle, float
            
            # Internal state
            self._device = None
            self._last_update_time = time.time()
            self._lock = threading.RLock()  # Reentrant lock for thread safety
            
            # Initialize device and sensors
            self._initialize_device()
            self._initialize_sensors()
            
            self._initialized = True
            logger.info("Hub singleton initialized")
    
    def _initialize_device(self):
        """Initialize the current device object"""
        device_type = self.device['type']
        
        try:
            if device_type == 'keyboard':
                from .devices.keyboard_device import KeyboardDeviceConfig, KeyboardDevice
                device_config = KeyboardDeviceConfig(
                    name="keyboard_input",
                    poll_rate=self.settings.get('GENERAL.poll_frequency', 100.0)
                )
                self._device = KeyboardDevice(device_config)
                self.device['obj'] = self._device
                logger.info("Keyboard device initialized")
            elif device_type == 'joystick':
                # TODO: Implement joystick device
                logger.warning("Joystick device not implemented yet")
                self._device = None
            else:
                logger.error(f"Unknown device type: {device_type}")
                self._device = None
        except Exception as e:
            logger.error(f"Failed to initialize device {device_type}: {e}")
            self._device = None
    
    def _initialize_sensors(self):
        """Initialize all enabled sensor objects"""
        # Create a copy of sensor names to avoid dictionary size change during iteration
        sensor_names = list(self.sensors.keys())
        
        for sensor_name in sensor_names:
            try:
                # Get sensor configuration from settings
                sensor_settings = self.settings.get(f'AVAILABLE_SENSORS.{sensor_name}', {})
                
                if sensor_name == 'gps':
                    sensor_obj = GPS(sensor_id='gps', **sensor_settings)
                elif sensor_name == 'barometer':
                    sensor_obj = Barometer(sensor_id='barometer', **sensor_settings)
                elif sensor_name == 'gyroscope':
                    sensor_obj = Gyroscope(sensor_id='gyroscope', **sensor_settings)
                elif sensor_name == 'temperature':
                    sensor_obj = Temperature(sensor_id='temperature', **sensor_settings)
                elif sensor_name == 'anemometer':
                    sensor_obj = Anemometer(sensor_id='anemometer', **sensor_settings)
                elif sensor_name == 'compass':
                    sensor_obj = Compass(sensor_id='compass', **sensor_settings)
                elif sensor_name == 'cameras':
                    # Special handling for multiple cameras
                    camera_count = sensor_settings.get('camera_count', 2)
                    fake = sensor_settings.get('fake', True)
                    cameras = []
                    for i in range(camera_count):
                        cam = Camera(sensor_id=f"camera_{i}", camera_index=i, **sensor_settings)
                        cameras.append(cam)
                    
                    # Store each camera individually for polling
                    for i, cam in enumerate(cameras):
                        camera_sensor_name = f'camera_{i}'
                        self.sensors[camera_sensor_name] = {'fake': fake, 'obj': cam}
                        self._sensors[camera_sensor_name] = cam
                    
                    # Store the list for the 'cameras' entry but don't poll it directly
                    sensor_obj = cameras
                    # Mark cameras as a group sensor that shouldn't be polled directly
                    self.sensors[sensor_name]['group_sensor'] = True
                else:
                    logger.warning(f"Unknown sensor type: {sensor_name}")
                    continue
                
                self.sensors[sensor_name]['obj'] = sensor_obj
                self._sensors[sensor_name] = sensor_obj
                fake_status = sensor_settings.get('fake', True)
                logger.info(f"Sensor {sensor_name} initialized (fake={fake_status})")
                
            except Exception as e:
                logger.error(f"Failed to initialize sensor {sensor_name}: {e}")
    
    def update_output(self, voltages: List[float]):
        """
        Update output voltages for engines
        This is called by the Drone module to set engine voltages
        
        Args:
            voltages: List of voltages for each engine (typically 4 engines)
        """
        with self._lock:
            # Ensure we have the right number of engines
            num_engines = len(self.output)
            
            if len(voltages) >= num_engines:
                self.output = voltages[:num_engines]
            else:
                # Pad with zeros if not enough voltages provided
                self.output = list(voltages) + [0.0] * (num_engines - len(voltages))
            
            # Clamp voltages to safe range
            max_voltage = self.settings.get('DRONE.engines.max_voltage', 12.0)
            min_voltage = self.settings.get('DRONE.engines.min_voltage', 0.0)
            
            self.output = [max(min_voltage, min(max_voltage, v)) for v in self.output]
            
            logger.debug(f"Hub output updated: {self.output}")
    
    def get_input_data(self) -> Dict[str, Any]:
        """Get a copy of all current input data"""
        with self._lock:
            return self.input.copy()
    
    def get_output_data(self) -> List[float]:
        """Get a copy of current output voltages"""
        with self._lock:
            return self.output.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete hub state"""
        with self._lock:
            return {
                'input': self.input.copy(),
                'output': self.output.copy(),
                'task': self.task,
                'mode': self.mode,
                'go': self.go,
                'simulation': self.simulation,
                'frequency': self.frequency,
                'last_update': self._last_update_time,
                'device_type': self.device['type'],
                'enabled_sensors': list(self.sensors.keys())
            }
    
    def set_mode(self, mode: str):
        """Set control mode"""
        valid_modes = ['ai', 'hybrid', 'manual']
        if mode in valid_modes:
            with self._lock:
                old_mode = self.mode
                self.mode = mode
                logger.info(f"Mode changed: {old_mode} -> {mode}")
        else:
            logger.error(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
    
    def set_go(self, go_state: str):
        """Set go state"""
        valid_states = ['off', 'operate', 'idle', 'float']
        if go_state in valid_states:
            with self._lock:
                old_go = self.go
                self.go = go_state
                logger.info(f"Go state changed: {old_go} -> {go_state}")
        else:
            logger.error(f"Invalid go state: {go_state}. Valid states: {valid_states}")
    
    def set_task(self, task: Optional[str]):
        """Set current task"""
        valid_tasks = [None, 'take_off', 'land', 'follow', 'back_to_base', 'projectile']
        if task in valid_tasks:
            with self._lock:
                old_task = self.task
                self.task = task
                logger.info(f"Task changed: {old_task} -> {task}")
        else:
            logger.error(f"Invalid task: {task}. Valid tasks: {valid_tasks}")
    
    def emergency_stop(self):
        """Emergency stop - set safe state"""
        with self._lock:
            logger.critical("EMERGENCY STOP ACTIVATED")
            self.mode = 'ai'
            self.go = 'float'
            self.task = None
            self.output = [0.0, 0.0, 0.0, 0.0]  # Zero all engine voltages
    
    def get_sensor_data(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """Get data from specific sensor"""
        with self._lock:
            return self.input.get(sensor_name)
    
    def is_device_connected(self) -> bool:
        """Check if device is connected and working"""
        device_status = self.input.get('device_status')
        if device_status:
            return device_status.get('connected', False)
        return False
    
    def get_engine_voltages(self) -> List[float]:
        """Get current engine voltages"""
        return self.get_output_data()
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate all input data"""
        validation = {}
        
        with self._lock:
            # Check device data
            validation['device'] = self.is_device_connected()
            
            # Check sensor data
            for sensor_name in self.sensors.keys():
                sensor_data = self.input.get(sensor_name)
                validation[sensor_name] = sensor_data is not None
        
        validation['overall'] = all(validation.values())
        return validation
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information about hub state"""
        with self._lock:
            data_age = time.time() - self._last_update_time
            
            return {
                'data_age_seconds': data_age,
                'data_fresh': data_age < 1.0,  # Data less than 1 second old
                'validation': self.validate_data(),
                'state': self.get_state(),
                'initialization_complete': self._initialized,
                'device_initialized': self._device is not None,
                'sensors_initialized': len(self._sensors),
                'memory_address': id(self)  # Confirm singleton
            }
    
    