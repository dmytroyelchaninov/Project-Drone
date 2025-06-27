"""
Keyboard input device for manual drone control
Handles keyboard polling and converts to voltage commands
"""
import pygame
import numpy as np
from typing import Dict, Any, Set
from dataclasses import dataclass
import time

from .base_device import BaseDevice, DeviceConfig
from .voltage_controller import VoltageController, VoltageControllerConfig

class KeyboardDeviceConfig:
    """Configuration for keyboard input device"""
    def __init__(self, name: str, poll_rate: float = 100.0, **kwargs):
        self.name = name
        self.poll_rate = poll_rate
        # Control sensitivity
        self.voltage_sensitivity: float = 2.0  # Voltage change per key press
        self.differential_sensitivity: float = 1.0  # Roll/pitch differential
        
        # Key mappings - Initialize after pygame import
        self.throttle_up_key: int = 32  # pygame.K_SPACE
        self.throttle_down_key: int = 304  # pygame.K_LSHIFT
        self.roll_left_key: int = 276  # pygame.K_LEFT
        self.roll_right_key: int = 275  # pygame.K_RIGHT
        self.pitch_forward_key: int = 273  # pygame.K_UP
        self.pitch_backward_key: int = 274  # pygame.K_DOWN
        self.yaw_left_key: int = 97  # pygame.K_a
        self.yaw_right_key: int = 100  # pygame.K_d
        self.emergency_stop_key: int = 27  # pygame.K_ESCAPE
        
        # Response characteristics
        self.key_repeat_rate: float = 10.0  # Hz - how fast to repeat key actions
        self.key_release_delay: float = 0.1  # seconds - delay before considering key released

class KeyboardDevice(BaseDevice):
    """
    Keyboard input device for drone control
    Converts keyboard input to voltage commands for engines
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
        
        # Key states
        self.pressed_keys: Set[int] = set()
        self.key_press_times: Dict[int, float] = {}
        self.key_release_times: Dict[int, float] = {}
        
        # Control states
        self.throttle_delta = 0.0
        self.roll_delta = 0.0
        self.pitch_delta = 0.0
        self.yaw_delta = 0.0
        
        # Create voltage controller
        voltage_config = VoltageControllerConfig(
            name="engine_voltage_controller",
            poll_rate=config.poll_rate
        )
        self.voltage_controller = VoltageController(voltage_config)
        
        # Start voltage controller
        self.voltage_controller.start()
    
    def _connect_device(self) -> bool:
        """Initialize pygame and keyboard input"""
        try:
            if not self.pygame_initialized:
                pygame.init()
                # Create a small window to capture keyboard events
                self.screen = pygame.display.set_mode((100, 100))
                pygame.display.set_caption("Drone Control")
                self.pygame_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize keyboard: {e}")
            return False
    
    def _disconnect_device(self):
        """Cleanup pygame"""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        
        # Stop voltage controller
        self.voltage_controller.stop()
    
    def _poll_data(self) -> Dict[str, Any]:
        """Poll keyboard input and update voltage commands"""
        current_time = time.time()
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self._handle_key_press(event.key, current_time)
            elif event.type == pygame.KEYUP:
                self._handle_key_release(event.key, current_time)
            elif event.type == pygame.QUIT:
                return {'quit_requested': True}
        
        # Update control deltas based on current key states
        self._update_control_deltas(current_time)
        
        # Convert control inputs to voltage commands
        voltage_commands = self._calculate_voltage_commands()
        
        # Send to voltage controller
        self.voltage_controller.set_engine_voltages(voltage_commands)
        
        # Get latest voltage controller data
        voltage_data = self.voltage_controller.get_latest_data()
        
        return {
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
            'timestamp': current_time
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate keyboard input data"""
        try:
            required_keys = ['pressed_keys', 'throttle_delta', 'roll_delta', 'pitch_delta', 'yaw_delta']
            return all(key in data for key in required_keys)
        except Exception:
            return False
    
    def _handle_key_press(self, key: int, timestamp: float):
        """Handle key press event"""
        if key not in self.pressed_keys:
            self.pressed_keys.add(key)
            self.key_press_times[key] = timestamp
            
            # Handle emergency stop immediately
            if key == self.keyboard_config.emergency_stop_key:
                self.voltage_controller.emergency_stop_all()
    
    def _handle_key_release(self, key: int, timestamp: float):
        """Handle key release event"""
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
            self.key_release_times[key] = timestamp
    
    def _update_control_deltas(self, current_time: float):
        """Update control deltas based on current key states"""
        dt = 1.0 / self.keyboard_config.poll_rate
        
        # Reset deltas
        throttle_input = 0.0
        roll_input = 0.0
        pitch_input = 0.0
        yaw_input = 0.0
        
        # Check each control key
        if self.keyboard_config.throttle_up_key in self.pressed_keys:
            throttle_input += 1.0
        if self.keyboard_config.throttle_down_key in self.pressed_keys:
            throttle_input -= 1.0
            
        if self.keyboard_config.roll_right_key in self.pressed_keys:
            roll_input += 1.0
        if self.keyboard_config.roll_left_key in self.pressed_keys:
            roll_input -= 1.0
            
        if self.keyboard_config.pitch_forward_key in self.pressed_keys:
            pitch_input += 1.0
        if self.keyboard_config.pitch_backward_key in self.pressed_keys:
            pitch_input -= 1.0
            
        if self.keyboard_config.yaw_right_key in self.pressed_keys:
            yaw_input += 1.0
        if self.keyboard_config.yaw_left_key in self.pressed_keys:
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
        """Reset all control inputs to neutral"""
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
        """Get current control status"""
        return {
            'active_keys': len(self.pressed_keys),
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