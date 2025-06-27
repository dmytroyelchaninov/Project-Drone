"""
Camera Sensor Implementation
Provides visual data for navigation and obstacle detection
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import time
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class Camera(BaseSensor):
    
    def _connect_device(self) -> bool:
        return self.fake or False
    
    def _disconnect_device(self):
        pass
    
    def _read_sensor(self) -> float:
        if self.fake:
            # For cameras, we return frame count as the primary reading
            frames = self.get_frames()
            return float(len(frames) if frames else 0)
        return 0.0
    
    def _convert_units(self, raw_value: float) -> float:
        return raw_value
    """
    Camera sensor for visual data acquisition
    Can simulate multiple cameras or interface with real hardware
    """
    
    def __init__(self, sensor_id: str = "camera", fake: bool = True, **kwargs):
        from .base_sensor import SensorConfig
        config = SensorConfig(
            name=sensor_id,
            poll_rate=kwargs.get('poll_rate', 30.0),
            measurement_range=(0.0, 1.0)  # Frame availability (0 or 1)
        )
        super().__init__(config)
        self.fake = fake
        
        # Camera-specific configuration
        self.resolution = kwargs.get('resolution', [640, 480])  # [width, height]
        self.fps = kwargs.get('fps', 30)
        self.update_rate = self.fps
        
        # Camera properties
        self.fov_horizontal = kwargs.get('fov_h', 90.0)  # degrees
        self.fov_vertical = kwargs.get('fov_v', 60.0)    # degrees
        self.camera_index = kwargs.get('camera_index', 0)
        
        # Image data
        self.frame = None
        self.frame_timestamp = 0.0
        
        # Features for navigation
        self.detected_objects = []
        self.optical_flow = np.zeros(2)  # [vx, vy] pixels/frame
        
        logger.info(f"Camera sensor '{sensor_id}' initialized (fake={fake})")
    
    def _poll_real(self) -> Dict[str, Any]:
        """Poll real camera hardware"""
        # This would interface with actual camera hardware (OpenCV, etc.)
        logger.warning("Real camera hardware not implemented")
        return None
    
    def _poll_fake(self) -> Dict[str, Any]:
        """Generate simulated camera data"""
        # Generate synthetic image data
        width, height = self.resolution
        
        # Create simple synthetic image
        frame = self._generate_synthetic_frame(width, height)
        
        # Simulate object detection
        objects = self._simulate_object_detection()
        
        # Simulate optical flow
        flow = self._simulate_optical_flow()
        
        self.frame = frame
        self.frame_timestamp = time.time()
        self.detected_objects = objects
        self.optical_flow = flow
        
        return {
            'frame': frame,
            'frame_shape': [height, width, 3],
            'resolution': self.resolution,
            'fps': self.fps,
            'timestamp': self.frame_timestamp,
            'objects_detected': objects,
            'optical_flow': flow,
            'exposure_time': 1.0 / self.fps,
            'brightness': np.mean(frame) if frame is not None else 128
        }
    
    def _generate_synthetic_frame(self, width: int, height: int) -> np.ndarray:
        """Generate a synthetic camera frame"""
        # Create simple gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add sky gradient (blue to white)
        for y in range(height // 2):
            intensity = int(200 + (255 - 200) * y / (height // 2))
            frame[y, :, 2] = intensity  # Blue channel
            frame[y, :, 1] = intensity // 2  # Green channel
        
        # Add ground (green to brown)
        for y in range(height // 2, height):
            ground_y = y - height // 2
            green_intensity = int(100 + 50 * ground_y / (height // 2))
            frame[y, :, 1] = green_intensity  # Green channel
            frame[y, :, 0] = green_intensity // 2  # Red channel for brown
        
        # Add some noise
        noise = np.random.randint(-10, 10, (height, width, 3))
        frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def _simulate_object_detection(self) -> list:
        """Simulate basic object detection"""
        objects = []
        
        # Randomly detect some objects
        num_objects = np.random.poisson(2)  # Average 2 objects per frame
        
        for i in range(num_objects):
            # Random object properties
            obj = {
                'id': i,
                'type': np.random.choice(['obstacle', 'landmark', 'target']),
                'bbox': [
                    np.random.randint(0, self.resolution[0] - 50),  # x
                    np.random.randint(0, self.resolution[1] - 50),  # y
                    np.random.randint(20, 100),  # width
                    np.random.randint(20, 100)   # height
                ],
                'confidence': np.random.uniform(0.5, 1.0),
                'distance_estimate': np.random.uniform(1.0, 10.0)  # meters
            }
            objects.append(obj)
        
        return objects
    
    def _simulate_optical_flow(self) -> np.ndarray:
        """Simulate optical flow calculation"""
        # Get motion from physics if available
        try:
            from ...physics import QuadcopterPhysics
            physics = QuadcopterPhysics()
            velocity = physics.state.velocity
            angular_velocity = physics.state.angular_velocity
            
            # Simple approximation: translate velocity to pixel motion
            # This is very simplified - real optical flow is much more complex
            pixel_scale = 10.0  # pixels per m/s
            flow_x = velocity[0] * pixel_scale
            flow_y = velocity[1] * pixel_scale
            
            # Add rotational component
            flow_x += angular_velocity[2] * 50  # yaw rate contribution
            
            return np.array([flow_x, flow_y])
            
        except ImportError:
            # No physics available, return small random flow
            return np.random.normal(0, 1.0, 2)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest camera frame"""
        if self.last_data:
            return self.last_data.get('frame')
        return None
    
    def get_objects(self) -> list:
        """Get detected objects"""
        if self.last_data:
            return self.last_data.get('objects_detected', [])
        return []
    
    def get_optical_flow(self) -> np.ndarray:
        """Get optical flow vector"""
        if self.last_data:
            return self.last_data.get('optical_flow', np.zeros(2))
        return np.zeros(2)
    
    def is_motion_detected(self, threshold: float = 2.0) -> bool:
        """Check if significant motion is detected"""
        flow = self.get_optical_flow()
        return np.linalg.norm(flow) > threshold
    
    def count_objects(self, object_type: str = None) -> int:
        """Count detected objects of specific type"""
        objects = self.get_objects()
        if object_type is None:
            return len(objects)
        return len([obj for obj in objects if obj.get('type') == object_type])
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera configuration info"""
        return {
            'resolution': self.resolution,
            'fps': self.fps,
            'fov_horizontal': self.fov_horizontal,
            'fov_vertical': self.fov_vertical,
            'camera_index': self.camera_index
        }
