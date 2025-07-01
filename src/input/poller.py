"""
Poller Module
Handles polling of all devices and sensors according to Hub specifications
"""
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from .hub import Hub

logger = logging.getLogger(__name__)

class Poller:
    """
    Poller manages the polling of all devices and sensors
    Runs in separate thread and updates Hub data in real-time
    """
    
    def __init__(self, hub: Hub):
        self.hub = hub
        self._running = False
        self._poll_thread = None
        self._force_poll_event = threading.Event()
        self._last_poll_time = 0.0
        
        # Error handling
        self._error_count = 0
        self._max_errors = 10
        self._error_recovery_attempts = 0
        self._max_recovery_attempts = 3
        self._recovery_timeout = 20.0  # 20ms between recovery attempts
        self._emergency_timeout = 180.0  # 3 minutes total recovery time
        
        logger.info("Poller initialized")
    
    def start(self):
        """Start the polling thread"""
        if self._running:
            logger.warning("Poller already running")
            return True
        
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("Poller started")
        return True
    
    def stop(self):
        """Stop the polling thread"""
        if not self._running:
            return
        
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
        logger.info("Poller stopped")
    
    def force_poll(self):
        """Force an immediate poll cycle"""
        self._force_poll_event.set()
        logger.debug("Force poll requested")
    
    def test(self) -> Dict[str, Any]:
        """Test all devices and sensors with single poll"""
        logger.info("Running poller test...")
        
        results = {
            'device_test': self._test_device(),
            'sensor_tests': self._test_sensors(),
            'timestamp': time.time()
        }
        
        # Analyze results
        device_ok = results['device_test']['success']
        sensors_ok = all(test['success'] for test in results['sensor_tests'].values())
        
        results['overall_success'] = device_ok and sensors_ok
        results['errors'] = []
        
        if not device_ok:
            results['errors'].append(f"Device test failed: {results['device_test']['error']}")
        
        for sensor_id, test in results['sensor_tests'].items():
            if not test['success']:
                results['errors'].append(f"Sensor {sensor_id} test failed: {test['error']}")
        
        logger.info(f"Poller test complete: {'PASS' if results['overall_success'] else 'FAIL'}")
        return results
    
    def _poll_loop(self):
        """Main polling loop"""
        logger.info("Poll loop started")
        
        while self._running:
            start_time = time.time()
            
            try:
                # Check if force poll requested
                if self._force_poll_event.is_set():
                    self._force_poll_event.clear()
                    logger.debug("Force poll executed")
                
                # Poll device and sensors
                self._poll_cycle()
                
                # Reset error count on successful poll
                self._error_count = 0
                self._error_recovery_attempts = 0
                
            except Exception as e:
                self._handle_poll_error(e)
            
            # Calculate sleep time to maintain frequency
            poll_frequency = self.hub.settings.get('GENERAL.POLL_FREQUENCY', 50)
            target_interval = 1.0 / poll_frequency
            
            elapsed = time.time() - start_time
            sleep_time = max(0, target_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            self._last_poll_time = time.time()
        
        logger.info("Poll loop ended")
    
    def _poll_cycle(self):
        """Execute one complete poll cycle"""
        # Poll device
        device_data = self._poll_device()
        if device_data:
            self._update_hub_from_device(device_data)
        
        # Poll all sensors
        for sensor_id, sensor in self.hub._sensors.items():
            try:
                # Skip group sensors like 'cameras' that contain lists
                if sensor_id in self.hub.sensors and self.hub.sensors[sensor_id].get('group_sensor', False):
                    continue
                    
                if hasattr(sensor, 'poll') and callable(sensor.poll):
                    sensor_data = sensor.poll()
                    if sensor_data:
                        # Update hub input directly
                        self.hub.input[sensor_id] = sensor_data
                elif isinstance(sensor, list):
                    # Handle list of sensors (shouldn't happen with our new structure but just in case)
                    logger.debug(f"Skipping list sensor {sensor_id}")
                    continue
                else:
                    logger.warning(f"Sensor {sensor_id} has no poll method")
            except Exception as e:
                logger.error(f"Error polling sensor {sensor_id}: {e}")
        
        # Update timestamp
        self.hub._last_update_time = time.time()
    
    def _poll_device(self) -> Optional[Dict[str, Any]]:
        """Poll the current device using Hub's safe polling method"""
        try:
            # Skip keyboard device polling from background thread to avoid macOS threading issues
            # Keyboard polling will be handled from main thread
            if (hasattr(self.hub, '_device') and self.hub._device and 
                hasattr(self.hub._device, '__class__') and 
                'KeyboardDevice' in str(self.hub._device.__class__)):
                # Return None so no device data is processed from background thread
                return None
            
            return self.hub.poll_device_safe()
        except Exception as e:
            logger.error(f"Error polling device: {e}")
            return None
    
    def _update_hub_from_device(self, device_data: Dict[str, Any]):
        """Update hub state from device data"""
        # Extract control state changes
        if 'mode' in device_data:
            old_mode = self.hub.mode
            self.hub.mode = device_data['mode']
            if old_mode != self.hub.mode:
                logger.info(f"Mode changed: {old_mode} -> {self.hub.mode}")
        
        if 'go' in device_data:
            old_go = self.hub.go
            self.hub.go = device_data['go']
            if old_go != self.hub.go:
                logger.info(f"Go state changed: {old_go} -> {self.hub.go}")
        
        if 'task' in device_data:
            old_task = self.hub.task
            self.hub.task = device_data['task']
            if old_task != self.hub.task:
                logger.info(f"Task changed: {old_task} -> {self.hub.task}")
        
        # Store full device data
        if 'voltages' in device_data:
            self.hub.input['device_voltages'] = device_data['voltages']
        
        # Store device status
        self.hub.input['device_status'] = {
            'connected': device_data.get('connected', True),
            'type': device_data.get('device_type', 'unknown'),
            'last_update': time.time()
        }
    
    def _handle_poll_error(self, error: Exception):
        """Handle polling errors with recovery logic"""
        self._error_count += 1
        logger.error(f"Poll error #{self._error_count}: {error}")
        
        if self._error_count >= self._max_errors:
            logger.critical("Too many poll errors, entering recovery mode")
            self._enter_recovery_mode()
    
    def _enter_recovery_mode(self):
        """Enter emergency recovery mode"""
        logger.warning("Entering emergency recovery mode")
        
        # Set hub to safe state
        self.hub.mode = 'ai'
        self.hub.go = 'float'
        self.hub.task = None
        
        # Try recovery
        start_time = time.time()
        
        while (time.time() - start_time) < self._emergency_timeout and self._running:
            self._error_recovery_attempts += 1
            logger.info(f"Recovery attempt {self._error_recovery_attempts}/{self._max_recovery_attempts}")
            
            try:
                # Test device and sensors
                test_result = self.test()
                
                if test_result['overall_success']:
                    logger.info("Recovery successful")
                    self._error_count = 0
                    self._error_recovery_attempts = 0
                    return
                
                if self._error_recovery_attempts >= self._max_recovery_attempts:
                    logger.critical("Max recovery attempts reached")
                    break
                
            except Exception as e:
                logger.error(f"Recovery attempt failed: {e}")
            
            time.sleep(self._recovery_timeout / 1000.0)  # Convert to seconds
        
        # Final emergency state
        logger.critical("Recovery failed, setting emergency state")
        self.hub.mode = 'ai'
        self.hub.go = 'operate'
        self.hub.task = 'back_to_base'
    
    def _test_device(self) -> Dict[str, Any]:
        """Test device functionality"""
        if not self.hub._device:
            return {'success': False, 'error': 'No device configured'}
        
        try:
            data = self.hub._device.poll()
            return {
                'success': data is not None,
                'error': None if data else 'Device returned no data',
                'data': data
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'data': None}
    
    def _test_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Test all sensors"""
        results = {}
        
        for sensor_id, sensor in self.hub._sensors.items():
            try:
                data = sensor.poll()
                results[sensor_id] = {
                    'success': data is not None,
                    'error': None if data else 'Sensor returned no data',
                    'data': data
                }
            except Exception as e:
                results[sensor_id] = {
                    'success': False,
                    'error': str(e),
                    'data': None
                }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get poller status information"""
        return {
            'running': self._running,
            'last_poll_time': self._last_poll_time,
            'error_count': self._error_count,
            'recovery_attempts': self._error_recovery_attempts,
            'poll_frequency': self.hub.settings.get('GENERAL.POLL_FREQUENCY', 50),
            'thread_alive': self._poll_thread.is_alive() if self._poll_thread else False
        } 