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
                
                # Set hub reference for mode control
                self._device.set_hub_reference(self)
                
                # Start the device
                if not self._device.start():
                    logger.error("Failed to start keyboard device")
                    self._device = None
                    return
                
                logger.info("Keyboard device initialized and started")
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

    def poll_device_safe(self) -> Optional[Dict[str, Any]]:
        """
        Poll the device safely, handling keyboard special requirements.
        This method should be called from the main thread for keyboard devices.
        """
        if not self._device:
            return None
        
        try:
            # Special handling for keyboard device that needs main thread access
            if isinstance(self._device, KeyboardDevice):
                return self._device.poll_pygame_safe()
            else:
                # Regular device polling
                return self._device.poll()
        except Exception as e:
            logger.error(f"Error polling device: {e}")
            return None
    
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
        """Get all current input data"""
        with self._lock:
            return self.input.copy()
    
    def get_output_data(self) -> List[float]:
        """Get current output voltages"""
        with self._lock:
            return self.output.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current control state"""
        return {
            'mode': self.mode,
            'go': self.go,
            'task': self.task,
            'simulation': self.simulation,
            'frequency': self.frequency,
            'device_type': self.device['type'],
            'device_connected': self.is_device_connected(),
            'last_update': self._last_update_time
        }
    
    # State control methods
    def set_mode(self, mode: str):
        """Set control mode (ai, hybrid, manual)"""
        valid_modes = ['ai', 'hybrid', 'manual']
        if mode in valid_modes:
            old_mode = self.mode
            self.mode = mode
            logger.info(f"Mode changed: {old_mode} -> {mode}")
        else:
            logger.warning(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
    
    def set_go(self, go_state: str):
        """Set go state (off, operate, idle, float)"""
        valid_states = ['off', 'operate', 'idle', 'float']
        if go_state in valid_states:
            old_go = self.go
            self.go = go_state
            logger.info(f"Go state changed: {old_go} -> {go_state}")
        else:
            logger.warning(f"Invalid go state: {go_state}. Valid states: {valid_states}")
    
    def set_task(self, task: Optional[str]):
        """Set current task"""
        valid_tasks = [None, 'take_off', 'land', 'follow', 'back_to_base', 'projectile']
        if task in valid_tasks:
            old_task = self.task
            self.task = task
            logger.info(f"Task changed: {old_task} -> {task}")
        else:
            logger.warning(f"Invalid task: {task}. Valid tasks: {valid_tasks}")
    
    def emergency_stop(self):
        """Emergency stop - set go state to 'off'"""
        logger.warning("EMERGENCY STOP activated!")
        self.set_go('off')
        if self._device and hasattr(self._device, 'emergency_stop_all'):
            self._device.emergency_stop_all()
    
    def get_sensor_data(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """Get data from specific sensor"""
        return self.input.get(sensor_name)
    
    def is_device_connected(self) -> bool:
        """Check if device is connected"""
        if not self._device:
            return False
        
        # Check device status
        device_status = self.input.get('device_status')
        if device_status:
            return device_status.get('connected', False)
        
        return False
    
    def get_engine_voltages(self) -> List[float]:
        """Get current engine voltages from device input"""
        return self.input.get('device_voltages', [0.0, 0.0, 0.0, 0.0])
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate all current data"""
        validation = {
            'device_connected': self.is_device_connected(),
            'has_device_voltages': self.input.get('device_voltages') is not None,
            'valid_mode': self.mode in ['ai', 'hybrid', 'manual'],
            'valid_go': self.go in ['off', 'operate', 'idle', 'float'],
            'valid_task': self.task in [None, 'take_off', 'land', 'follow', 'back_to_base', 'projectile']
        }
        
        # Check sensor data
        for sensor_name in self.sensors:
            if not self.sensors[sensor_name].get('group_sensor', False):
                validation[f'has_{sensor_name}'] = self.input.get(sensor_name) is not None
        
        return validation
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        return {
            'state': self.get_state(),
            'input_data': {k: v is not None for k, v in self.input.items()},
            'output_data': self.output,
            'validation': self.validate_data(),
            'last_update': self._last_update_time,
            'timestamp': time.time()
        }
    
    