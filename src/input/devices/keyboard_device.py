"""
Keyboard input device for manual drone control
Handles keyboard polling and converts to voltage commands
"""
import pygame
import numpy as np
from typing import Dict, Any, Set, Optional
from dataclasses import dataclass
import time
import threading
import logging

from .base_device import BaseDevice, DeviceConfig, DeviceStatus
from .voltage_controller import VoltageController, VoltageControllerConfig

# Add logging
logger = logging.getLogger(__name__)

class KeyboardDeviceConfig:
    """Configuration for keyboard input device"""
    def __init__(self, name: str, poll_rate: float = 100.0, **kwargs):
        self.name = name
        self.poll_rate = poll_rate
        # Control sensitivity
        self.voltage_sensitivity: float = 2.0  # Voltage change per key press
        self.differential_sensitivity: float = 1.0  # Roll/pitch differential
        
        # Key mappings - Use pygame constants for cross-platform correctness
        # Primary flight controls
        self.throttle_up_key: int = pygame.K_SPACE
        self.throttle_down_key: int = pygame.K_LSHIFT
        # Roll (left/right) – allow both arrow keys and A/D for left-hand setups
        self.roll_left_key: int = pygame.K_LEFT
        self.roll_right_key: int = pygame.K_RIGHT
        # Pitch (forward/backward)
        self.pitch_forward_key: int = pygame.K_UP
        self.pitch_backward_key: int = pygame.K_DOWN
        # Yaw (rotation)
        self.yaw_left_key: int = pygame.K_a
        self.yaw_right_key: int = pygame.K_d
        # Emergency / reset
        self.emergency_stop_key: int = pygame.K_ESCAPE
        
        # Mode control keys – use letter keys via pygame constants
        self.mode_manual_key: int = pygame.K_m
        self.mode_ai_key: int = pygame.K_i
        self.mode_hybrid_key: int = pygame.K_h
        self.go_operate_key: int = pygame.K_o
        self.go_idle_key: int = pygame.K_p
        self.task_takeoff_key: int = pygame.K_t
        self.task_landing_key: int = pygame.K_l
        self.reset_controls_key: int = pygame.K_r
        
        # Response characteristics
        self.key_repeat_rate: float = 10.0  # Hz - how fast to repeat key actions
        self.key_release_delay: float = 0.1  # seconds - delay before considering key released

class KeyboardDevice(BaseDevice):
    """
    Keyboard input device for drone control
    Converts keyboard input to voltage commands for engines
    
    Special implementation that overrides BaseDevice to avoid background threads
    and only processes pygame events when called from the main thread.
    """
    
    def __init__(self, config: KeyboardDeviceConfig):
        # Create a basic device config for BaseDevice
        from .base_device import DeviceConfig
        device_config = DeviceConfig(
            name=config.name,
            poll_rate=config.poll_rate
        )
        super().__init__(device_config)
        self.keyboard_config = config
        
        # Initialize pygame for keyboard input
        self.pygame_initialized = False
        self.screen = None
        
        # Thread-safe state management
        self.state_lock = threading.Lock()
        
        # Key states (protected by lock)
        self.pressed_keys: Set[int] = set()
        self.key_press_times: Dict[int, float] = {}
        self.key_release_times: Dict[int, float] = {}
        self.last_logged_keys: Set[int] = set()
        
        # Control states
        self.throttle_delta = 0.0
        self.roll_delta = 0.0
        self.pitch_delta = 0.0
        self.yaw_delta = 0.0
        
        # Mode control state
        self.hub_reference = None  # Will be set by Hub
        self.last_mode_change = 0.0
        
        # Track whether pygame events can be safely processed
        self.safe_to_poll_pygame = False
        self.last_poll_data = None
        
        # Create voltage controller
        voltage_config = VoltageControllerConfig(
            name="engine_voltage_controller",
            poll_rate=config.poll_rate
        )
        self.voltage_controller = VoltageController(voltage_config)
        
        # Start voltage controller
        self.voltage_controller.start()
        logger.info(f"KeyboardDevice '{config.name}' initialized")
    
    def set_hub_reference(self, hub):
        """Set reference to hub for mode control"""
        self.hub_reference = hub
        logger.info("Hub reference set for keyboard device")
    
    def start(self) -> bool:
        """
        Override BaseDevice.start() to avoid background threading.
        Just initialize pygame and mark as connected.
        """
        logger.info(f"Starting device: {self.config.name}")
        
        # Initialize pygame
        if self._connect_device():
            self._set_status(DeviceStatus.CONNECTED)
            logger.info(f"Device {self.config.name} started successfully")
            return True
        else:
            self._set_status(DeviceStatus.ERROR)
            return False
    
    def stop(self):
        """Stop device and cleanup"""
        logger.info(f"Stopping device: {self.config.name}")
        self._disconnect_device()
        self._set_status(DeviceStatus.DISCONNECTED)
        logger.info(f"Device {self.config.name} stopped")
    
    def poll_pygame_safe(self) -> Optional[Dict[str, Any]]:
        """
        Safe method to poll pygame events from the main thread.
        This should be called by the Hub from the main thread.
        """
        self.safe_to_poll_pygame = True
        try:
            return self.poll()
        finally:
            self.safe_to_poll_pygame = False
    
    def _connect_device(self) -> bool:
        """Initialize pygame for keyboard input"""
        try:
            if not self.pygame_initialized:
                pygame.init()
                # Create a small window to capture keyboard events
                self.screen = pygame.display.set_mode((100, 100))
                pygame.display.set_caption("Drone Control")
                self.pygame_initialized = True
                logger.info("Pygame initialized for keyboard input")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize keyboard: {e}")
            return False
    
    def _disconnect_device(self):
        """Cleanup pygame"""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
            logger.info("Pygame cleaned up")
        
        # Stop voltage controller
        self.voltage_controller.stop()
    
    def _poll_data(self) -> Dict[str, Any]:
        """
        Poll keyboard input and update voltage commands
        
        Only processes pygame events if marked as safe (called from main thread).
        Otherwise returns cached data to avoid threading issues.
        """
        current_time = time.time()
        
        # Only process pygame events if we're in a safe context (main thread)
        if self.safe_to_poll_pygame and self.pygame_initialized:
            # DEBUG: Log that we're polling pygame
            logger.info("🔍 Polling pygame events...")
            
            # Process pygame events (safe in main thread)
            events_processed = 0
            mode_commands = []
            
            for event in pygame.event.get():
                events_processed += 1
                logger.info(f"🎯 Pygame event detected: {event.type}")
                
                if event.type == pygame.KEYDOWN:
                    logger.info(f"🔽 Key down: {event.key}")
                    self._handle_key_press(event.key, current_time)
                    # Check for mode control commands
                    mode_cmd = self._handle_mode_key_press(event.key, current_time)
                    if mode_cmd:
                        mode_commands.append(mode_cmd)
                elif event.type == pygame.KEYUP:
                    logger.info(f"🔼 Key up: {event.key}")
                    self._handle_key_release(event.key, current_time)
                elif event.type == pygame.QUIT:
                    logger.info("Quit event received")
                    return {'quit_requested': True}
            
            logger.info(f"📊 Processed {events_processed} pygame events")
            
            # Log key state changes for debugging
            self._log_key_changes()
            
            # Update control deltas based on current key states
            self._update_control_deltas(current_time)
            
            # Convert control inputs to voltage commands
            voltage_commands = self._calculate_voltage_commands()
            
            # Send to voltage controller
            self.voltage_controller.set_engine_voltages(voltage_commands)
            
            # Get latest voltage controller data
            voltage_data = self.voltage_controller.get_latest_data()
            
            # Create and cache the poll data
            self.last_poll_data = {
                'pressed_keys': list(self.pressed_keys),
                'throttle_delta': self.throttle_delta,
                'roll_delta': self.roll_delta,
                'pitch_delta': self.pitch_delta,
                'yaw_delta': self.yaw_delta,
                'voltage_commands': voltage_commands.tolist(),
                'voltages': voltage_commands.tolist(),
                'voltage_data': voltage_data,
                'device_type': 'keyboard',
                'connected': True,
                'timestamp': current_time,
                'events_processed': events_processed,
                'mode_commands': mode_commands,
                'pygame_events_processed': True
            }
            
        else:
            # Not safe to process pygame events, return cached/simulated data
            if self.last_poll_data is None:
                # Create minimal initial data
                voltage_commands = self._calculate_voltage_commands()
                self.last_poll_data = {
                    'pressed_keys': [],
                    'throttle_delta': 0.0,
                    'roll_delta': 0.0,
                    'pitch_delta': 0.0,
                    'yaw_delta': 0.0,
                    'voltage_commands': voltage_commands.tolist(),
                    'voltages': voltage_commands.tolist(),
                    'voltage_data': None,
                    'device_type': 'keyboard',
                    'connected': True,
                    'timestamp': current_time,
                    'events_processed': 0,
                    'mode_commands': [],
                    'pygame_events_processed': False
                }
            
            # Update timestamp on cached data
            self.last_poll_data['timestamp'] = current_time
        
        return self.last_poll_data
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate keyboard input data"""
        try:
            required_keys = ['pressed_keys', 'throttle_delta', 'roll_delta', 'pitch_delta', 'yaw_delta']
            return all(key in data for key in required_keys)
        except Exception:
            return False
    
    def _handle_key_press(self, key: int, timestamp: float):
        """Handle key press event (thread-safe)"""
        with self.state_lock:
            if key not in self.pressed_keys:
                self.pressed_keys.add(key)
                self.key_press_times[key] = timestamp
                
                # Handle emergency stop immediately
                if key == self.keyboard_config.emergency_stop_key:
                    logger.warning("EMERGENCY STOP activated!")
                    self.voltage_controller.emergency_stop_all()
                
                # Handle reset controls
                if key == self.keyboard_config.reset_controls_key:
                    logger.info("Controls reset")
                    self.reset_controls()
    
    def _handle_key_release(self, key: int, timestamp: float):
        """Handle key release event (thread-safe)"""
        with self.state_lock:
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)
                self.key_release_times[key] = timestamp
    
    def _handle_mode_key_press(self, key: int, timestamp: float) -> Dict[str, Any]:
        """Handle mode control key presses"""
        if not self.hub_reference:
            return None
        
        # Prevent rapid mode changes
        if timestamp - self.last_mode_change < 0.5:
            return None
        
        mode_command = None
        
        try:
            if key == self.keyboard_config.mode_manual_key:
                self.hub_reference.set_mode('manual')
                mode_command = {'action': 'set_mode', 'value': 'manual'}
                logger.info("Mode switched to MANUAL")
                
            elif key == self.keyboard_config.mode_ai_key:
                self.hub_reference.set_mode('ai')
                mode_command = {'action': 'set_mode', 'value': 'ai'}
                logger.info("Mode switched to AI")
                
            elif key == self.keyboard_config.mode_hybrid_key:
                self.hub_reference.set_mode('hybrid')
                mode_command = {'action': 'set_mode', 'value': 'hybrid'}
                logger.info("Mode switched to HYBRID")
                
            elif key == self.keyboard_config.go_operate_key:
                self.hub_reference.set_go('operate')
                mode_command = {'action': 'set_go', 'value': 'operate'}
                logger.info("Go state set to OPERATE")
                
            elif key == self.keyboard_config.go_idle_key:
                self.hub_reference.set_go('idle')
                mode_command = {'action': 'set_go', 'value': 'idle'}
                logger.info("Go state set to IDLE")
                
            elif key == self.keyboard_config.task_takeoff_key:
                self.hub_reference.set_task('takeoff')
                mode_command = {'action': 'set_task', 'value': 'takeoff'}
                logger.info("Task set to TAKEOFF")
                
            elif key == self.keyboard_config.task_landing_key:
                self.hub_reference.set_task('landing')
                mode_command = {'action': 'set_task', 'value': 'landing'}
                logger.info("Task set to LANDING")
            
            if mode_command:
                self.last_mode_change = timestamp
                
        except Exception as e:
            logger.error(f"Error handling mode key press: {e}")
        
        return mode_command
    
    def _log_key_changes(self):
        """Log key state changes for debugging (thread-safe)"""
        with self.state_lock:
            current_keys = self.pressed_keys.copy()
        
        # Keys newly pressed
        new_keys = current_keys - self.last_logged_keys
        if new_keys:
            key_names = [self._get_key_name(k) for k in new_keys]
            logger.info(f"🎮 Keys pressed: {key_names}")
        
        # Keys newly released
        released_keys = self.last_logged_keys - current_keys
        if released_keys:
            key_names = [self._get_key_name(k) for k in released_keys]
            logger.info(f"🎮 Keys released: {key_names}")
        
        self.last_logged_keys = current_keys
    
    def _get_key_name(self, key: int) -> str:
        """Get human readable key name"""
        key_map = {
            pygame.K_SPACE: "SPACE", pygame.K_LSHIFT: "L_SHIFT", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT",
            pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_a: "A", pygame.K_d: "D", pygame.K_ESCAPE: "ESC",
            pygame.K_m: "M", pygame.K_i: "I", pygame.K_h: "H", pygame.K_o: "O", pygame.K_p: "P",
            pygame.K_t: "T", pygame.K_l: "L", pygame.K_r: "R"
        }
        return key_map.get(key, f"KEY_{key}")
    
    def _update_control_deltas(self, current_time: float):
        """Update control deltas based on current key states (thread-safe)"""
        dt = 1.0 / self.keyboard_config.poll_rate
        
        # Reset deltas
        throttle_input = 0.0
        roll_input = 0.0
        pitch_input = 0.0
        yaw_input = 0.0
        
        # Get current key state safely
        with self.state_lock:
            pressed_keys = self.pressed_keys.copy()
        
        # Check each control key
        if self.keyboard_config.throttle_up_key in pressed_keys:
            throttle_input += 1.0
        if self.keyboard_config.throttle_down_key in pressed_keys:
            throttle_input -= 1.0
            
        if self.keyboard_config.roll_right_key in pressed_keys:
            roll_input += 1.0
        if self.keyboard_config.roll_left_key in pressed_keys:
            roll_input -= 1.0
            
        if self.keyboard_config.pitch_forward_key in pressed_keys:
            pitch_input += 1.0
        if self.keyboard_config.pitch_backward_key in pressed_keys:
            pitch_input -= 1.0
            
        if self.keyboard_config.yaw_right_key in pressed_keys:
            yaw_input += 1.0
        if self.keyboard_config.yaw_left_key in pressed_keys:
            yaw_input -= 1.0
        
        # Apply sensitivity and integrate over time
        self.throttle_delta += throttle_input * self.keyboard_config.voltage_sensitivity * dt
        self.roll_delta = roll_input * self.keyboard_config.differential_sensitivity
        self.pitch_delta = pitch_input * self.keyboard_config.differential_sensitivity
        self.yaw_delta = yaw_input * self.keyboard_config.differential_sensitivity * 0.5
        
        # Apply limits
        max_throttle_delta = 6.0  # Maximum voltage deviation from hover
        self.throttle_delta = np.clip(self.throttle_delta, -max_throttle_delta, max_throttle_delta)
    
    def _calculate_voltage_commands(self) -> np.ndarray:
        """Convert control inputs to individual engine voltage commands"""
        # Start with hover voltage for all engines
        hover_voltage = self.voltage_controller.voltage_config.hover_voltage
        base_voltages = np.full(4, hover_voltage + self.throttle_delta)
        
        # Apply control mixing for quadcopter (+ configuration)
        # Engine layout:
        #   0 (Front)
        # 3   1 (Left/Right)
        #   2 (Back)
        
        voltages = base_voltages.copy()
        
        # Roll (left/right)
        voltages[1] += self.roll_delta   # Right engine
        voltages[3] -= self.roll_delta   # Left engine
        
        # Pitch (forward/backward)
        voltages[0] += self.pitch_delta  # Front engine
        voltages[2] -= self.pitch_delta  # Back engine
        
        # Yaw (rotation) - opposite pairs
        voltages[0] -= self.yaw_delta    # Front (CCW)
        voltages[1] += self.yaw_delta    # Right (CW)
        voltages[2] -= self.yaw_delta    # Back (CCW)
        voltages[3] += self.yaw_delta    # Left (CW)
        
        return voltages
    
    def reset_controls(self):
        """Reset all control inputs to neutral (thread-safe)"""
        with self.state_lock:
            self.throttle_delta = 0.0
            self.roll_delta = 0.0
            self.pitch_delta = 0.0
            self.yaw_delta = 0.0
            self.pressed_keys.clear()
            self.key_press_times.clear()
            self.key_release_times.clear()
        
        # Reset voltage controller to hover
        self.voltage_controller.reset_emergency_stop()
    
    def get_control_status(self) -> Dict[str, Any]:
        """Get current control status (thread-safe)"""
        with self.state_lock:
            active_keys = len(self.pressed_keys)
        
        return {
            'active_keys': active_keys,
            'throttle_delta': self.throttle_delta,
            'roll_delta': self.roll_delta,
            'pitch_delta': self.pitch_delta,
            'yaw_delta': self.yaw_delta,
            'emergency_stop': self.voltage_controller.emergency_stop,
            'voltage_controller_connected': self.voltage_controller.is_connected()
        }
    
    def get_engine_data(self) -> Dict[str, Any]:
        """Get current engine data from voltage controller"""
        return self.voltage_controller.get_engine_diagnostics() 