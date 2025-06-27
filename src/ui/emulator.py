"""
Emulator Module
3D visualization and UI for drone simulation using pygame and OpenGL
"""
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import math
from typing import Dict, Any, List, Tuple
import logging

try:
    from .transmission import Transmission
    from ..cfg import settings
except ImportError:
    # Fallback for direct execution
    from ui.transmission import Transmission
    from cfg import settings

logger = logging.getLogger(__name__)

# Try to import OpenGL, fallback to basic pygame if not available
try:
    from OpenGL.arrays import vbo
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    logger.warning("OpenGL not available, using basic pygame rendering")

class Emulator:
    """
    3D Emulator for drone simulation
    
    Renders the drone from third-person view with real-time sensor data overlay
    Uses Transmission to get physics and sensor data for visualization
    """
    
    def __init__(self, transmission: Transmission):
        self.transmission = transmission
        
        # Display settings
        self.width = settings.get('UI.window_width', 1200)
        self.height = settings.get('UI.window_height', 800)
        self.fps = settings.get('UI.target_fps', 60)
        
        # Camera settings
        self.camera_distance = 5.0
        self.camera_height = 2.0
        self.camera_angle_h = 0.0  # Horizontal angle
        self.camera_angle_v = -15.0  # Vertical angle (degrees)
        
        # UI state
        self.show_sensors = True
        self.show_debug_info = True
        self.show_trajectory = True
        self.running = False
        
        # Rendering state
        self.clock = pygame.time.Clock()
        self.font = None
        self.trajectory_points = []
        self.max_trajectory_points = 1000
        
        # Mouse controls
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        logger.info("Emulator initialized")
    
    def initialize(self) -> bool:
        """Initialize pygame and OpenGL"""
        try:
            pygame.init()
            
            # Set up display
            display_flags = DOUBLEBUF | OPENGL
            if OPENGL_AVAILABLE:
                pygame.display.set_mode((self.width, self.height), display_flags)
                self._init_opengl()
            else:
                pygame.display.set_mode((self.width, self.height))
            
            pygame.display.set_caption("Drone Simulator - 3D View")
            
            # Initialize font for UI text
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            
            # Set up OpenGL if available
            if OPENGL_AVAILABLE:
                self._setup_lighting()
            
            logger.info("Emulator display initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize emulator: {e}")
            return False
    
    def run(self):
        """Main emulator loop"""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting emulator main loop")
        
        while self.running:
            frame_start = time.time()
            
            # Handle events
            self._handle_events()
            
            # Update transmission physics
            self.transmission.update_physics_from_hub()
            
            # Render frame if needed
            if self.transmission.should_render():
                self._render_frame()
            
            # Maintain target FPS
            self.clock.tick(self.fps)
            
            # Calculate actual FPS for debug
            frame_time = time.time() - frame_start
            actual_fps = 1.0 / frame_time if frame_time > 0 else 0
            
        pygame.quit()
        logger.info("Emulator stopped")
    
    def stop(self):
        """Stop the emulator"""
        self.running = False
    
    def _init_opengl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def _setup_lighting(self):
        """Set up OpenGL lighting"""
        # Ambient light
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        
        # Diffuse light
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Light position (sun)
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 1.0])
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            elif event.type == KEYDOWN:
                self._handle_keydown(event.key)
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            
            elif event.type == MOUSEMOTION:
                if self.mouse_dragging:
                    self._handle_mouse_drag(event.pos)
            
            elif event.type == MOUSEWHEEL:
                self._handle_mouse_wheel(event.y)
    
    def _handle_keydown(self, key):
        """Handle keyboard input"""
        if key == K_ESCAPE:
            self.running = False
        elif key == K_F1:
            self.show_sensors = not self.show_sensors
        elif key == K_F2:
            self.show_debug_info = not self.show_debug_info
        elif key == K_F3:
            self.show_trajectory = not self.show_trajectory
        elif key == K_r:
            # Reset camera
            self.camera_distance = 5.0
            self.camera_height = 2.0
            self.camera_angle_h = 0.0
            self.camera_angle_v = -15.0
    
    def _handle_mouse_drag(self, mouse_pos):
        """Handle mouse drag for camera control"""
        dx = mouse_pos[0] - self.last_mouse_pos[0]
        dy = mouse_pos[1] - self.last_mouse_pos[1]
        
        # Update camera angles
        self.camera_angle_h += dx * 0.5
        self.camera_angle_v += dy * 0.5
        
        # Clamp vertical angle
        self.camera_angle_v = max(-89, min(89, self.camera_angle_v))
        
        self.last_mouse_pos = mouse_pos
    
    def _handle_mouse_wheel(self, wheel_y):
        """Handle mouse wheel for zoom"""
        zoom_speed = 0.5
        self.camera_distance -= wheel_y * zoom_speed
        self.camera_distance = max(1.0, min(50.0, self.camera_distance))
    
    def _render_frame(self):
        """Render complete frame"""
        if OPENGL_AVAILABLE:
            self._render_3d_frame()
        else:
            self._render_2d_frame()
        
        pygame.display.flip()
    
    def _render_3d_frame(self):
        """Render 3D frame using OpenGL"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Get drone state from transmission
        drone_state = self.transmission.get_drone_state()
        environment_data = self.transmission.get_environment_data()
        
        # Set up camera
        self._setup_camera(drone_state['position'])
        
        # Render environment
        self._render_environment_3d(environment_data)
        
        # Render drone
        self._render_drone_3d(drone_state)
        
        # Update trajectory
        self._update_trajectory(drone_state['position'])
        
        # Render trajectory
        if self.show_trajectory:
            self._render_trajectory_3d()
        
        # Render UI overlay
        if self.show_sensors or self.show_debug_info:
            self._render_ui_overlay()
    
    def _render_2d_frame(self):
        """Render 2D frame using basic pygame"""
        # Fill background
        screen = pygame.display.get_surface()
        screen.fill((50, 100, 150))  # Sky blue
        
        # Get data
        drone_state = self.transmission.get_drone_state()
        
        # Simple 2D representation
        center_x, center_y = self.width // 2, self.height // 2
        
        # Draw drone as simple shape
        drone_x = int(center_x + drone_state['position']['x'] * 50)
        drone_y = int(center_y - drone_state['position']['y'] * 50)
        
        pygame.draw.circle(screen, (255, 255, 255), (drone_x, drone_y), 10)
        
        # Render UI
        if self.show_sensors or self.show_debug_info:
            self._render_ui_overlay_2d(screen)
    
    def _setup_camera(self, drone_pos: Dict[str, float]):
        """Set up 3D camera to follow drone"""
        glLoadIdentity()
        
        # Calculate camera position
        h_rad = math.radians(self.camera_angle_h)
        v_rad = math.radians(self.camera_angle_v)
        
        # Camera position relative to drone
        cam_x = drone_pos['x'] + self.camera_distance * math.cos(v_rad) * math.sin(h_rad)
        cam_y = drone_pos['y'] + self.camera_distance * math.cos(v_rad) * math.cos(h_rad)
        cam_z = drone_pos['z'] + self.camera_height + self.camera_distance * math.sin(v_rad)
        
        # Look at drone
        gluLookAt(cam_x, cam_y, cam_z,  # Camera position
                  drone_pos['x'], drone_pos['y'], drone_pos['z'],  # Look at point
                  0, 0, 1)  # Up vector
    
    def _render_environment_3d(self, env_data: Dict[str, Any]):
        """Render 3D environment"""
        # Render simple ground plane
        ground_size = 50.0
        ground_height = env_data.get('ground_height', 0.0)
        
        glColor3f(0.3, 0.7, 0.3)  # Green ground
        glBegin(GL_QUADS)
        glVertex3f(-ground_size, -ground_size, ground_height)
        glVertex3f(ground_size, -ground_size, ground_height)
        glVertex3f(ground_size, ground_size, ground_height)
        glVertex3f(-ground_size, ground_size, ground_height)
        glEnd()
        
        # Render grid lines
        glColor3f(0.2, 0.5, 0.2)
        glBegin(GL_LINES)
        for i in range(-50, 51, 5):
            # Vertical lines
            glVertex3f(i, -ground_size, ground_height + 0.01)
            glVertex3f(i, ground_size, ground_height + 0.01)
            # Horizontal lines
            glVertex3f(-ground_size, i, ground_height + 0.01)
            glVertex3f(ground_size, i, ground_height + 0.01)
        glEnd()
    
    def _render_drone_3d(self, drone_state: Dict[str, Any]):
        """Render 3D drone model"""
        glPushMatrix()
        
        # Translate to drone position
        pos = drone_state['position']
        glTranslatef(pos['x'], pos['y'], pos['z'])
        
        # Apply drone orientation
        orient = drone_state['orientation']
        glRotatef(math.degrees(orient['yaw']), 0, 0, 1)
        glRotatef(math.degrees(orient['pitch']), 0, 1, 0)
        glRotatef(math.degrees(orient['roll']), 1, 0, 0)
        
        # Render drone body
        self._render_drone_body()
        
        # Render propellers
        self._render_propellers(drone_state['engines']['rpms'])
        
        glPopMatrix()
    
    def _render_drone_body(self):
        """Render drone body geometry"""
        # Central body
        glColor3f(0.2, 0.2, 0.2)  # Dark gray
        self._draw_cube(0.1, 0.1, 0.05)
        
        # Arms
        arm_length = 0.225
        arm_thickness = 0.02
        
        glColor3f(0.3, 0.3, 0.3)
        
        # Front arm
        glPushMatrix()
        glTranslatef(arm_length/2, 0, 0)
        self._draw_cube(arm_length, arm_thickness, arm_thickness)
        glPopMatrix()
        
        # Right arm
        glPushMatrix()
        glTranslatef(0, arm_length/2, 0)
        self._draw_cube(arm_thickness, arm_length, arm_thickness)
        glPopMatrix()
        
        # Back arm
        glPushMatrix()
        glTranslatef(-arm_length/2, 0, 0)
        self._draw_cube(arm_length, arm_thickness, arm_thickness)
        glPopMatrix()
        
        # Left arm
        glPushMatrix()
        glTranslatef(0, -arm_length/2, 0)
        self._draw_cube(arm_thickness, arm_length, arm_thickness)
        glPopMatrix()
    
    def _render_propellers(self, rpms: List[float]):
        """Render spinning propellers"""
        arm_length = 0.225
        propeller_positions = [
            [arm_length, 0, 0.02],     # Front
            [0, arm_length, 0.02],     # Right
            [-arm_length, 0, 0.02],    # Back
            [0, -arm_length, 0.02]     # Left
        ]
        
        for i, (pos, rpm) in enumerate(zip(propeller_positions, rpms)):
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            
            # Color based on RPM
            intensity = min(1.0, rpm / 3000.0)
            glColor3f(0.5 + intensity * 0.5, 0.5, 0.5)
            
            # Simple propeller representation
            self._draw_propeller(rpm)
            
            glPopMatrix()
    
    def _draw_cube(self, width: float, height: float, depth: float):
        """Draw a simple cube"""
        w, h, d = width/2, height/2, depth/2
        
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-w, -h, d)
        glVertex3f(w, -h, d)
        glVertex3f(w, h, d)
        glVertex3f(-w, h, d)
        
        # Back face
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, -h, -d)
        
        # Top face
        glVertex3f(-w, h, -d)
        glVertex3f(-w, h, d)
        glVertex3f(w, h, d)
        glVertex3f(w, h, -d)
        
        # Bottom face
        glVertex3f(-w, -h, -d)
        glVertex3f(w, -h, -d)
        glVertex3f(w, -h, d)
        glVertex3f(-w, -h, d)
        
        # Right face
        glVertex3f(w, -h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, h, d)
        glVertex3f(w, -h, d)
        
        # Left face
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, -h, d)
        glVertex3f(-w, h, d)
        glVertex3f(-w, h, -d)
        glEnd()
    
    def _draw_propeller(self, rpm: float):
        """Draw spinning propeller"""
        # Simple spinning disk representation
        radius = 0.1
        segments = 8
        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            glVertex3f(x, y, 0)
        
        glEnd()
    
    def _update_trajectory(self, position: Dict[str, float]):
        """Update drone trajectory trail"""
        self.trajectory_points.append([position['x'], position['y'], position['z']])
        
        # Limit trajectory points
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
    
    def _render_trajectory_3d(self):
        """Render drone trajectory as line trail"""
        if len(self.trajectory_points) < 2:
            return
        
        glColor3f(1.0, 1.0, 0.0)  # Yellow trail
        glBegin(GL_LINE_STRIP)
        
        for point in self.trajectory_points:
            glVertex3f(point[0], point[1], point[2])
        
        glEnd()
    
    def _render_ui_overlay(self):
        """Render UI overlay with sensor data"""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        
        # Get sensor data
        sensor_data = self.transmission.get_sensor_data()
        drone_state = self.transmission.get_drone_state()
        
        # Render sensor panel
        if self.show_sensors:
            self._render_sensor_panel(sensor_data, drone_state)
        
        # Render debug info
        if self.show_debug_info:
            self._render_debug_panel(drone_state)
        
        glEnable(GL_DEPTH_TEST)
        
        # Restore 3D projection
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _render_ui_overlay_2d(self, screen):
        """Render UI overlay for 2D mode"""
        # Get data
        sensor_data = self.transmission.get_sensor_data()
        drone_state = self.transmission.get_drone_state()
        
        y_offset = 10
        
        if self.show_sensors:
            y_offset = self._render_sensor_text_2d(screen, sensor_data, y_offset)
        
        if self.show_debug_info:
            self._render_debug_text_2d(screen, drone_state, y_offset)
    
    def _render_sensor_panel(self, sensor_data: Dict[str, Any], drone_state: Dict[str, Any]):
        """Render sensor data panel (OpenGL text rendering would be complex)"""
        # For now, just indicate where sensor panel would be
        glColor3f(0.0, 0.0, 0.0)  # Black background
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(300, 10)
        glVertex2f(300, 200)
        glVertex2f(10, 200)
        glEnd()
    
    def _render_debug_panel(self, drone_state: Dict[str, Any]):
        """Render debug information panel"""
        # For now, just indicate where debug panel would be
        glColor3f(0.2, 0.2, 0.2)  # Dark gray background
        glBegin(GL_QUADS)
        glVertex2f(self.width - 310, 10)
        glVertex2f(self.width - 10, 10)
        glVertex2f(self.width - 10, 200)
        glVertex2f(self.width - 310, 200)
        glEnd()
    
    def _render_sensor_text_2d(self, screen, sensor_data: Dict[str, Any], y_offset: int) -> int:
        """Render sensor data as text in 2D mode"""
        x = 10
        
        # GPS
        gps = sensor_data.get('gps', {})
        if gps.get('status') == 'active':
            text = f"GPS: {gps.get('latitude', 0):.6f}, {gps.get('longitude', 0):.6f}, Alt: {gps.get('altitude', 0):.1f}m"
            self._draw_text(screen, text, x, y_offset)
            y_offset += 25
        
        # Barometer
        baro = sensor_data.get('barometer', {})
        if baro.get('status') == 'active':
            text = f"Barometer: {baro.get('pressure', 0):.0f} Pa, Alt: {baro.get('altitude', 0):.1f}m"
            self._draw_text(screen, text, x, y_offset)
            y_offset += 25
        
        # Add more sensors...
        
        return y_offset
    
    def _render_debug_text_2d(self, screen, drone_state: Dict[str, Any], y_offset: int):
        """Render debug information as text in 2D mode"""
        x = 10
        
        pos = drone_state['position']
        text = f"Position: X:{pos['x']:.2f} Y:{pos['y']:.2f} Z:{pos['z']:.2f}"
        self._draw_text(screen, text, x, y_offset)
        y_offset += 25
        
        vel = drone_state['velocity']
        text = f"Velocity: {vel['speed']:.2f} m/s"
        self._draw_text(screen, text, x, y_offset)
        y_offset += 25
        
        engines = drone_state['engines']
        text = f"Voltages: {engines['voltages']}"
        self._draw_text(screen, text, x, y_offset)
    
    def _draw_text(self, screen, text: str, x: int, y: int, color=(255, 255, 255)):
        """Draw text on screen"""
        if self.font:
            text_surface = self.font.render(text, True, color)
            screen.blit(text_surface, (x, y)) 