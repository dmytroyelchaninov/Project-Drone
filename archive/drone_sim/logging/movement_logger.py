#!/usr/bin/env python3
"""
Movement Logger
Specialized logger for tracking drone movement commands and key press patterns
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class KeyPressEvent:
    """Record of a key press event"""
    key: str
    action: str  # 'press' or 'release'
    timestamp: float
    simulation_time: float
    hold_duration: Optional[float] = None  # Only set on release

@dataclass
class MovementFrame:
    """Record of movement state at a specific time"""
    timestamp: float
    simulation_time: float
    
    # Position and velocity
    position: List[float]
    velocity: List[float]
    
    # Control inputs
    thrust: float
    moment: List[float]  # [roll, pitch, yaw]
    
    # Engine thrusts (individual motors)
    engine_thrusts: List[float]
    
    # Active keys
    active_keys: List[str]
    
    # Key hold durations (for currently pressed keys)
    key_hold_durations: Dict[str, float]

class MovementLogger:
    """Specialized logger for movement tracking"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = time.time()
        self.simulation_start_time = 0.0
        
        # Event tracking
        self.key_events: List[KeyPressEvent] = []
        self.movement_frames: List[MovementFrame] = []
        
        # Key state tracking
        self.active_keys: Dict[str, float] = {}  # key -> press_time
        
        # Sampling control
        self.last_frame_time = 0.0
        self.frame_interval = 0.05  # 20 Hz sampling for movement
        
        print("🎯 Movement Logger initialized" if enabled else "🎯 Movement Logger disabled")
    
    def log_key_event(self, key: str, pressed: bool, simulation_time: float):
        """Log a key press or release event"""
        if not self.enabled:
            return
        
        current_time = time.time()
        hold_duration = None
        action = 'press' if pressed else 'release'
        
        if pressed:
            self.active_keys[key] = current_time
        elif key in self.active_keys:
            hold_duration = current_time - self.active_keys[key]
            del self.active_keys[key]
        
        event = KeyPressEvent(
            key=key,
            action=action,
            timestamp=current_time,
            simulation_time=simulation_time,
            hold_duration=hold_duration
        )
        
        self.key_events.append(event)
        
        if pressed:
            print(f"🎮 Key {key} pressed at t={simulation_time:.3f}s")
        else:
            duration_str = f" (held {hold_duration:.3f}s)" if hold_duration else ""
            print(f"🎮 Key {key} released at t={simulation_time:.3f}s{duration_str}")
    
    def log_movement_frame(self, simulation_time: float, position: np.ndarray, 
                          velocity: np.ndarray, thrust: float, moment: np.ndarray,
                          engine_thrusts: Optional[np.ndarray] = None):
        """Log a movement frame (position, velocity, control state)"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Sample at specified interval
        if current_time - self.last_frame_time < self.frame_interval:
            return
        
        self.last_frame_time = current_time
        
        # Calculate current hold durations
        key_hold_durations = {}
        for key, press_time in self.active_keys.items():
            key_hold_durations[key] = current_time - press_time
        
        frame = MovementFrame(
            timestamp=current_time,
            simulation_time=simulation_time,
            position=position.tolist(),
            velocity=velocity.tolist(),
            thrust=thrust,
            moment=moment.tolist(),
            engine_thrusts=engine_thrusts.tolist() if engine_thrusts is not None else [0, 0, 0, 0],
            active_keys=list(self.active_keys.keys()),
            key_hold_durations=key_hold_durations
        )
        
        self.movement_frames.append(frame)
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """Get statistics about key usage patterns"""
        if not self.key_events:
            return {}
        
        # Analyze key press patterns
        key_stats = {}
        
        for event in self.key_events:
            if event.action == 'release' and event.hold_duration is not None:
                key = event.key
                if key not in key_stats:
                    key_stats[key] = {
                        'total_presses': 0,
                        'total_hold_time': 0.0,
                        'min_hold': float('inf'),
                        'max_hold': 0.0,
                        'hold_durations': []
                    }
                
                stats = key_stats[key]
                stats['total_presses'] += 1
                stats['total_hold_time'] += event.hold_duration
                stats['min_hold'] = min(stats['min_hold'], event.hold_duration)
                stats['max_hold'] = max(stats['max_hold'], event.hold_duration)
                stats['hold_durations'].append(event.hold_duration)
        
        # Calculate averages
        for key, stats in key_stats.items():
            if stats['total_presses'] > 0:
                stats['average_hold'] = stats['total_hold_time'] / stats['total_presses']
                stats['median_hold'] = np.median(stats['hold_durations'])
                del stats['hold_durations']  # Remove raw data to save space
        
        return key_stats
    
    def get_movement_summary(self) -> Dict[str, Any]:
        """Get summary of movement patterns"""
        if not self.movement_frames:
            return {}
        
        # Extract data for analysis
        positions = np.array([frame.position for frame in self.movement_frames])
        velocities = np.array([frame.velocity for frame in self.movement_frames])
        thrusts = np.array([frame.thrust for frame in self.movement_frames])
        
        # Calculate movement statistics
        total_distance = 0.0
        max_speed = 0.0
        
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
        
        speeds = np.linalg.norm(velocities, axis=1)
        max_speed = np.max(speeds)
        avg_speed = np.mean(speeds)
        
        return {
            'total_frames': len(self.movement_frames),
            'duration': self.movement_frames[-1].simulation_time - self.movement_frames[0].simulation_time if self.movement_frames else 0,
            'total_distance_traveled': float(total_distance),
            'max_speed': float(max_speed),
            'average_speed': float(avg_speed),
            'min_thrust': float(np.min(thrusts)),
            'max_thrust': float(np.max(thrusts)),
            'average_thrust': float(np.mean(thrusts)),
            'start_position': positions[0].tolist() if len(positions) > 0 else [0, 0, 0],
            'end_position': positions[-1].tolist() if len(positions) > 0 else [0, 0, 0]
        }
    
    def save_movement_log(self, filename: Optional[str] = None) -> str:
        """Save detailed movement log to file"""
        if not self.enabled or not (self.key_events or self.movement_frames):
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"movement_log_{timestamp}.json"
        
        # Ensure logs directory exists
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        filepath = log_dir / filename
        
        # Prepare data for export
        log_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'duration_real': time.time() - self.start_time,
                'total_key_events': len(self.key_events),
                'total_movement_frames': len(self.movement_frames),
                'frame_interval': self.frame_interval
            },
            'key_statistics': self.get_key_statistics(),
            'movement_summary': self.get_movement_summary(),
            'key_events': [asdict(event) for event in self.key_events],
            'movement_frames': [asdict(frame) for frame in self.movement_frames]
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"💾 Movement log saved to: {filepath}")
            print(f"📊 Summary: {len(self.key_events)} key events, {len(self.movement_frames)} movement frames")
            
            # Print key statistics
            key_stats = self.get_key_statistics()
            if key_stats:
                print("🎮 Key Usage Summary:")
                for key, stats in key_stats.items():
                    print(f"   {key}: {stats['total_presses']} presses, avg hold {stats['average_hold']:.3f}s")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Failed to save movement log: {e}")
            return ""
    
    def clear_logs(self):
        """Clear all logged data"""
        self.key_events.clear()
        self.movement_frames.clear()
        self.active_keys.clear()
        print("🧹 Movement logs cleared")
    
    def get_live_stats(self) -> Dict[str, Any]:
        """Get current live statistics"""
        return {
            'active_keys': list(self.active_keys.keys()),
            'key_hold_durations': {k: time.time() - v for k, v in self.active_keys.items()},
            'total_key_events': len(self.key_events),
            'total_movement_frames': len(self.movement_frames),
            'recording_duration': time.time() - self.start_time
        } 