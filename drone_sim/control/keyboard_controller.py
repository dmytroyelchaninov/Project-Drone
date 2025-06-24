#!/usr/bin/env python3
"""
Enhanced Keyboard Controller with Smooth Thrust Control
Physics-based keyboard control with accelerating thrust and smooth transitions
"""

import pygame
import numpy as np
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base_controller import BaseController, ControllerState, ControllerReference, ControllerOutput
from .quadcopter_controller import QuadcopterController, QuadcopterConfig

@dataclass
class ThrustControlConfig:
    """Configuration for smooth thrust control"""
    # Hover thrust (to counteract gravity)
    hover_thrust: float = 14.715  # 1.5kg * 9.81 m/s²
    
    # Thrust limits
    max_thrust: float = 35.0      # Increased maximum thrust for more range
    min_thrust: float = 5.0       # Minimum thrust (allows controlled descent)
    
    # Acceleration rates (N/s)
    thrust_accel_rate: float = 15.0    # How fast thrust builds up
    thrust_decel_rate: float = 8.0     # How fast thrust returns to hover
    
    # Roll/pitch differential thrust
    max_differential: float = 3.0      # Max thrust difference between sides
    differential_accel_rate: float = 10.0  # How fast differential builds up
    differential_decel_rate: float = 12.0  # How fast differential returns to zero
    
    # Response curves
    thrust_curve_power: float = 1.8    # Non-linear response curve
    differential_curve_power: float = 1.5
    
    # Key persistence settings
    key_release_delay: float = 0.1     # Delay before considering key truly released (100ms)

class KeyboardController(BaseController):
    """Enhanced keyboard controller with smooth, realistic thrust control"""
    
    def __init__(self, config: QuadcopterConfig):
        super().__init__("KeyboardController")
        
        # Store configuration
        self.config = config
        self.thrust_config = ThrustControlConfig()
        
        # Physics-based quadcopter controller for engine mixing
        self.quad_controller = QuadcopterController(config)
        
        # Smooth thrust state
        self.current_thrust = self.thrust_config.hover_thrust
        self.target_thrust = self.thrust_config.hover_thrust
        
        # Differential thrust state (for roll control)
        self.current_differential = 0.0  # Positive = right side higher
        self.target_differential = 0.0
        
        # Key state tracking with timing
        self.key_states: Dict[str, bool] = {}
        self.key_press_times: Dict[str, float] = {}  # When keys were first pressed
        self.key_release_times: Dict[str, float] = {}  # When keys were released (for persistence)
        self.persistent_key_states: Dict[str, bool] = {}  # Persistent state ignoring brief releases
        
        # Movement logger integration
        self.movement_logger = None
        
        # Last update time for smooth transitions
        self.last_update_time = time.time()
        
        # Key mapping for different control schemes
        self.key_mapping = {
            # Arrow keys - primary thrust control
            'up': 'thrust_up',
            'down': 'thrust_down', 
            'left': 'roll_left',
            'right': 'roll_right',
            
            # WASD - secondary/fine control
            'w': 'pitch_forward',
            's': 'pitch_backward',
            'a': 'yaw_left',
            'd': 'yaw_right',
            
            # Special keys
            'space': 'hover',
            'escape': 'emergency_stop',
        }
        
        print("🎮 Enhanced Keyboard Controller initialized")
        print(f"   Hover thrust: {self.thrust_config.hover_thrust:.1f}N")
        print(f"   Thrust range: {self.thrust_config.min_thrust:.1f}N - {self.thrust_config.max_thrust:.1f}N")
        print(f"   Max differential: ±{self.thrust_config.max_differential:.1f}N")
    
    def set_movement_logger(self, logger):
        """Set the movement logger for tracking key events"""
        self.movement_logger = logger
    
    def set_key_state(self, key: str, pressed: bool):
        """Set the state of a specific key with improved tracking and persistence"""
        current_time = time.time()
        old_state = self.key_states.get(key, False)
        
        # Initialize persistent state if needed
        if key not in self.persistent_key_states:
            self.persistent_key_states = getattr(self, 'persistent_key_states', {})
            self.key_release_times = getattr(self, 'key_release_times', {})
            
        old_persistent_state = self.persistent_key_states.get(key, False)
        
        if pressed and not old_state:
            # Key press - record timing
            self.key_states[key] = True
            self.persistent_key_states[key] = True
            self.key_press_times[key] = current_time
            
            # Clear any pending release
            if key in self.key_release_times:
                del self.key_release_times[key]
            
            # Log movement event only on actual new press
            if not old_persistent_state and self.movement_logger:
                sim_time = getattr(self, '_simulation_time', 0.0)
                self.movement_logger.log_key_event(key, 'press', sim_time)
                
            print(f"🟢 Key {key} PRESSED - persistent state active")
            
        elif not pressed and old_state:
            # Key release - but use persistence to ignore brief releases
            self.key_states[key] = False
            self.key_release_times[key] = current_time
            
            # Don't immediately update persistent state - wait for delay
            print(f"🟡 Key {key} released - checking persistence...")
            
        # Update persistent states based on release delays
        keys_to_clear = []
        for released_key, release_time in self.key_release_times.items():
            time_since_release = current_time - release_time
            
            if time_since_release >= self.thrust_config.key_release_delay:
                # Key has been released long enough - actually release it
                self.persistent_key_states[released_key] = False
                keys_to_clear.append(released_key)
                
                if released_key in self.key_press_times:
                    del self.key_press_times[released_key]
                
                # Log actual release event
                if self.movement_logger:
                    sim_time = getattr(self, '_simulation_time', 0.0)
                    self.movement_logger.log_key_event(released_key, 'release', sim_time)
                    
                print(f"🔴 Key {released_key} ACTUALLY RELEASED after {time_since_release:.3f}s delay")
        
        # Clean up processed releases
        for released_key in keys_to_clear:
            del self.key_release_times[released_key]
    
    def get_key_hold_duration(self, key: str) -> float:
        """Get how long a key has been held (using persistent state)"""
        # Use persistent state for more stable tracking
        if not hasattr(self, 'persistent_key_states'):
            self.persistent_key_states = {}
            
        if not self.persistent_key_states.get(key, False) or key not in self.key_press_times:
            return 0.0
        return time.time() - self.key_press_times[key]
    
    def calculate_accelerated_value(self, current: float, target: float, 
                                  accel_rate: float, decel_rate: float, dt: float,
                                  curve_power: float = 1.0) -> float:
        """Calculate smooth acceleration/deceleration toward target"""
        if abs(target - current) < 0.01:
            return target
        
        # Determine if we're moving toward target (acceleration) or away from hover (deceleration)
        # For thrust control, acceleration means moving away from hover, deceleration means returning to hover
        hover_thrust = self.thrust_config.hover_thrust
        
        # Calculate distances from hover for current and target
        current_distance_from_hover = abs(current - hover_thrust)
        target_distance_from_hover = abs(target - hover_thrust)
        
        # Choose rate: use accel_rate when moving away from hover, decel_rate when returning to hover
        if target_distance_from_hover > current_distance_from_hover:
            # Moving away from hover (e.g., increasing thrust with UP arrow)
            rate = accel_rate
        else:
            # Returning to hover or moving closer to hover
            rate = decel_rate
        
        # Calculate direction of change
        direction = 1.0 if target > current else -1.0
        change = rate * dt * direction
        
        # Apply curve to make acceleration feel more natural
        if curve_power != 1.0 and abs(target - hover_thrust) > 0.01:
            # Calculate progress toward target (0 = at starting point, 1 = at target)
            total_distance = abs(target - current)
            if total_distance > 0:
                progress = 1.0 - (abs(target - current) / total_distance)
                progress = max(0.0, min(progress, 1.0))  # Clamp between 0 and 1
                # Apply curve: stronger acceleration at start, smoother at end
                curve_factor = (1.0 - progress) ** (1.0 / curve_power)
                change *= curve_factor
        
        new_value = current + change
        
        # Don't overshoot target
        if direction > 0:
            return min(new_value, target)
        else:
            return max(new_value, target)
    
    def update_thrust_targets(self):
        """Update target thrust and differential based on current key states"""
        # Default targets
        self.target_thrust = self.thrust_config.hover_thrust
        self.target_differential = 0.0
        
        # Ensure persistent states are initialized
        if not hasattr(self, 'persistent_key_states'):
            self.persistent_key_states = {}
        
        # Vertical thrust control (UP/DOWN arrows) - use persistent states
        if self.persistent_key_states.get('up', False):
            hold_duration = self.get_key_hold_duration('up')
            # Accelerating thrust - longer hold = more thrust
            thrust_factor = min(1.0, hold_duration * 2.0)  # Reach max in 0.5 seconds
            thrust_factor = thrust_factor ** self.thrust_config.thrust_curve_power
            
            self.target_thrust = self.thrust_config.hover_thrust + \
                               (self.thrust_config.max_thrust - self.thrust_config.hover_thrust) * thrust_factor
        
        elif self.persistent_key_states.get('down', False):
            hold_duration = self.get_key_hold_duration('down')
            # Decreasing thrust - longer hold = less thrust
            thrust_factor = min(1.0, hold_duration * 2.0)  # Reach min in 0.5 seconds
            thrust_factor = thrust_factor ** self.thrust_config.thrust_curve_power
            
            self.target_thrust = self.thrust_config.hover_thrust - \
                               (self.thrust_config.hover_thrust - self.thrust_config.min_thrust) * thrust_factor
        
        # Roll control (LEFT/RIGHT arrows) - differential thrust
        if self.persistent_key_states.get('right', False):
            hold_duration = self.get_key_hold_duration('right')
            differential_factor = min(1.0, hold_duration * 3.0)  # Reach max in 0.33 seconds
            differential_factor = differential_factor ** self.thrust_config.differential_curve_power
            
            self.target_differential = self.thrust_config.max_differential * differential_factor
        
        elif self.persistent_key_states.get('left', False):
            hold_duration = self.get_key_hold_duration('left')
            differential_factor = min(1.0, hold_duration * 3.0)  # Reach max in 0.33 seconds
            differential_factor = differential_factor ** self.thrust_config.differential_curve_power
            
            self.target_differential = -self.thrust_config.max_differential * differential_factor
        
        # Special commands
        if self.persistent_key_states.get('space', False):
            # Hover mode - force return to hover thrust
            self.target_thrust = self.thrust_config.hover_thrust
            self.target_differential = 0.0
        
        if self.persistent_key_states.get('escape', False):
            # Emergency stop - cut thrust but keep some to avoid crash
            self.target_thrust = self.thrust_config.min_thrust
            self.target_differential = 0.0
    
    def update(self, reference: ControllerReference, 
               current_state: ControllerState, dt: float) -> ControllerOutput:
        """Update the controller with smooth thrust control"""
        if not self.enabled:
            return ControllerOutput(thrust=self.thrust_config.hover_thrust)
        
        # Store simulation time for logging
        self._simulation_time = getattr(reference, 'simulation_time', 0.0)
        
        # Update timing
        current_time = time.time()
        actual_dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update target values based on key states
        self.update_thrust_targets()
        
        # Smooth acceleration/deceleration toward targets
        self.current_thrust = self.calculate_accelerated_value(
            self.current_thrust, self.target_thrust,
            self.thrust_config.thrust_accel_rate, self.thrust_config.thrust_decel_rate,
            actual_dt, self.thrust_config.thrust_curve_power
        )
        
        self.current_differential = self.calculate_accelerated_value(
            self.current_differential, self.target_differential,
            self.thrust_config.differential_accel_rate, self.thrust_config.differential_decel_rate,
            actual_dt, self.thrust_config.differential_curve_power
        )
        
        # Calculate engine thrusts with differential
        base_thrust_per_engine = self.current_thrust / 4.0
        
        # Apply differential (left/right roll)
        left_engines_thrust = base_thrust_per_engine - (self.current_differential / 4.0)
        right_engines_thrust = base_thrust_per_engine + (self.current_differential / 4.0)
        
        # Individual engine thrusts (front-left, front-right, rear-left, rear-right)
        engine_thrusts = np.array([
            left_engines_thrust,   # Front-left
            right_engines_thrust,  # Front-right  
            left_engines_thrust,   # Rear-left
            right_engines_thrust   # Rear-right
        ])
        
        # Clamp engine thrusts to safe limits
        engine_thrusts = np.clip(engine_thrusts, 0.5, 10.0)
        
        # Calculate total thrust and moments
        total_thrust = np.sum(engine_thrusts)
        
        # Calculate moments from engine differential
        # Roll moment (around X-axis) from left/right differential
        roll_moment = self.current_differential * 0.225  # 0.225m arm length
        
        # Add pitch/yaw from WASD keys (smaller effects)
        pitch_moment = 0.0
        yaw_moment = 0.0
        
        if self.key_states.get('w', False):  # Pitch forward
            pitch_moment = -1.0
        elif self.key_states.get('s', False):  # Pitch backward
            pitch_moment = 1.0
            
        if self.key_states.get('a', False):  # Yaw left
            yaw_moment = -0.5
        elif self.key_states.get('d', False):  # Yaw right
            yaw_moment = 0.5
        
        moment = np.array([roll_moment, pitch_moment, yaw_moment])
        
        # Create output
        output = ControllerOutput(
            thrust=total_thrust,
            moment=moment,
            motor_commands=engine_thrusts
        )
        
        # Log movement frame
        if self.movement_logger:
            self.movement_logger.log_movement_frame(
                self._simulation_time,
                current_state.position,
                current_state.velocity,
                total_thrust,
                moment,
                engine_thrusts
            )
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.current_thrust = self.thrust_config.hover_thrust
        self.target_thrust = self.thrust_config.hover_thrust
        self.current_differential = 0.0
        self.target_differential = 0.0
        self.key_states.clear()
        self.key_press_times.clear()
        self.last_update_time = time.time()
    
    def get_debug_info(self) -> Dict:
        """Get debug information about controller state"""
        return {
            'enabled': self.enabled,
            'current_thrust': self.current_thrust,
            'target_thrust': self.target_thrust,
            'current_differential': self.current_differential,
            'target_differential': self.target_differential,
            'active_keys': [k for k, v in self.key_states.items() if v],
            'key_hold_durations': {k: self.get_key_hold_duration(k) 
                                 for k in self.key_states if self.key_states[k]},
            'thrust_config': {
                'hover_thrust': self.thrust_config.hover_thrust,
                'max_thrust': self.thrust_config.max_thrust,
                'min_thrust': self.thrust_config.min_thrust,
                'max_differential': self.thrust_config.max_differential
            }
        }
    
    def update_control_from_gui(self):
        """Update control from GUI key states (called by interface)"""
        # This method is called by the GUI interface but all processing
        # is done in the update() method when key states change
        pass 