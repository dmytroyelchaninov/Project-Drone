"""
Base device classes for input polling and sensor management
"""
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DeviceStatus(Enum):
    """Device connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    TESTING = "testing"

class DeviceConfig:
    """Base configuration for all devices"""
    def __init__(self, name: str, poll_rate: float = 100.0, timeout: float = 1.0, 
                 retry_attempts: int = 3, auto_reconnect: bool = True, 
                 validation_enabled: bool = True):
        self.name = name
        self.poll_rate = poll_rate
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.auto_reconnect = auto_reconnect
        self.validation_enabled = validation_enabled

class BaseDevice(ABC):
    """
    Base class for all input devices and sensors
    Provides common polling, connection management, and validation
    """
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.status = DeviceStatus.DISCONNECTED
        self.last_data: Optional[Dict[str, Any]] = None
        self.last_poll_time = 0.0
        self.error_count = 0
        self.connection_attempts = 0
        
        # Threading
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_polling = threading.Event()
        self._data_lock = threading.Lock()
        
        # Callbacks
        self._data_callbacks: list[Callable] = []
        self._error_callbacks: list[Callable] = []
        self._status_callbacks: list[Callable] = []
    
    @abstractmethod
    def _connect_device(self) -> bool:
        """Connect to the physical device. Return True if successful."""
        pass
    
    @abstractmethod
    def _disconnect_device(self):
        """Disconnect from the physical device."""
        pass
    
    @abstractmethod
    def _poll_data(self) -> Dict[str, Any]:
        """Poll data from device. Should raise exception on error."""
        pass
    
    @abstractmethod
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate polled data. Return True if valid."""
        pass
    
    def start(self) -> bool:
        """
        Start device polling
        Tests connection and begins polling thread
        Returns True if successful
        """
        logger.info(f"Starting device: {self.config.name}")
        
        # Test connection
        if not self._test_connection():
            return False
        
        # Start polling thread
        self._stop_polling.clear()
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            name=f"{self.config.name}_poller",
            daemon=True
        )
        self._polling_thread.start()
        
        logger.info(f"Device {self.config.name} started successfully")
        return True
    
    def stop(self):
        """Stop device polling and disconnect"""
        logger.info(f"Stopping device: {self.config.name}")
        
        # Stop polling thread
        self._stop_polling.set()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2.0)
        
        # Disconnect device
        self._disconnect_device()
        self._set_status(DeviceStatus.DISCONNECTED)
        
        logger.info(f"Device {self.config.name} stopped")
    
    def poll(self) -> Optional[Dict[str, Any]]:
        """
        Single poll operation
        Returns latest data or None if error/no data
        """
        try:
            if self.status != DeviceStatus.CONNECTED:
                return None
            
            data = self._poll_data()
            
            if self.config.validation_enabled and not self._validate_data(data):
                self._handle_error(ValueError("Data validation failed"))
                return None
            
            with self._data_lock:
                self.last_data = data
                self.last_poll_time = time.time()
            
            # Notify callbacks
            for callback in self._data_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.warning(f"Data callback error: {e}")
            
            self.error_count = 0  # Reset error count on success
            return data
            
        except Exception as e:
            self._handle_error(e)
            return None
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the most recent data (thread-safe)"""
        with self._data_lock:
            return self.last_data.copy() if self.last_data else None
    
    def get_status(self) -> DeviceStatus:
        """Get current device status"""
        return self.status
    
    def is_connected(self) -> bool:
        """Check if device is connected and operational"""
        return self.status == DeviceStatus.CONNECTED
    
    def add_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for new data"""
        self._data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors"""
        self._error_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[DeviceStatus], None]):
        """Add callback for status changes"""
        self._status_callbacks.append(callback)
    
    def _test_connection(self) -> bool:
        """Test device connection with retries"""
        self._set_status(DeviceStatus.TESTING)
        
        for attempt in range(self.config.retry_attempts):
            self.connection_attempts += 1
            logger.debug(f"Connection attempt {attempt + 1} for {self.config.name}")
            
            try:
                if self._connect_device():
                    # Test with a poll
                    test_data = self._poll_data()
                    if not self.config.validation_enabled or self._validate_data(test_data):
                        self._set_status(DeviceStatus.CONNECTED)
                        return True
                    else:
                        raise ValueError("Initial data validation failed")
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        self._set_status(DeviceStatus.ERROR)
        return False
    
    def _polling_loop(self):
        """Main polling loop (runs in separate thread)"""
        poll_interval = 1.0 / self.config.poll_rate
        
        while not self._stop_polling.is_set():
            start_time = time.time()
            
            # Poll data
            self.poll()
            
            # Handle reconnection if needed
            if self.status == DeviceStatus.ERROR and self.config.auto_reconnect:
                if self._test_connection():
                    logger.info(f"Device {self.config.name} reconnected successfully")
            
            # Sleep for remaining time
            elapsed = time.time() - start_time
            sleep_time = max(0, poll_interval - elapsed)
            if sleep_time > 0:
                self._stop_polling.wait(sleep_time)
    
    def _handle_error(self, error: Exception):
        """Handle polling errors"""
        self.error_count += 1
        logger.error(f"Device {self.config.name} error: {error}")
        
        # Notify error callbacks
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
        
        # Set error status if too many consecutive errors
        if self.error_count >= 3:
            self._set_status(DeviceStatus.ERROR)
    
    def _set_status(self, status: DeviceStatus):
        """Set device status and notify callbacks"""
        if self.status != status:
            old_status = self.status
            self.status = status
            logger.debug(f"Device {self.config.name} status: {old_status} -> {status}")
            
            # Notify status callbacks
            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.warning(f"Status callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics"""
        return {
            'name': self.config.name,
            'status': self.status.value,
            'error_count': self.error_count,
            'connection_attempts': self.connection_attempts,
            'last_poll_time': self.last_poll_time,
            'poll_rate': self.config.poll_rate,
            'has_data': self.last_data is not None
        } 