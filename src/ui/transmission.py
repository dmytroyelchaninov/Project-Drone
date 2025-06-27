"""
Transmission Module
Intermediary between Hub and Emulator for physics calculations and rendering data
"""
import numpy as np
import time
from typing import Dict, Any, List, Tuple
import logging

try:
    from ..cfg import settings
    from ..physics import QuadcopterPhysics, Environment, Propeller, PropellerConfig
    from ..input.hub import Hub
except ImportError:
    # Fallback for direct execution
    from cfg import settings
    from physics import QuadcopterPhysics, Environment, Propeller, PropellerConfig
    from input.hub import Hub

logger = logging.getLogger(__name__)

class Transmission:
    """
    Transmission class - intermediary between Hub and Emulator
    
    Takes voltage data from Hub.output and converts to physical parameters
    using drone configuration and physics calculations.
    Provides rendering data for the emulator.
    """
    
    def __init__(self, hub: Hub, environment: Environment, physics: QuadcopterPhysics):
        self.hub = hub
        self.environment = environment
        self.physics = physics
        
        # Load drone configuration
        self.drone_cfg = settings.get_section('DRONE')
        
        # Initialize propeller physics for voltage-to-thrust conversion
        propeller_config = PropellerConfig()
        self.propeller = Propeller(propeller_config)
        
        # Rendering state
        self.last_render_time = 0.0
        self.render_frequency = settings.get('UI.render_frequency', 60.0)  # Hz
        
        # Cache for performance
        self._cached_environment_data = None
        self._cache_expire_time = 0.0
        self._cache_duration = 0.1  # Cache for 100ms
        
        logger.info("Transmission system initialized")
    
    def update_physics_from_hub(self):
        """
        Update physics simulation based on current Hub output voltages
        """
        try:
            # Get voltage outputs from hub
            voltages = self.hub.get_output_data()
            
            if len(voltages) >= 4:
                # Convert voltages to thrusts
                thrusts = []
                for voltage in voltages[:4]:  # Only use first 4 engines
                    thrust = self.propeller.calculate_thrust_from_voltage(voltage)
                    thrusts.append(thrust)
                
                # Update physics engine
                thrust_array = np.array(thrusts)
                self.physics.set_engine_thrusts(thrust_array)
                
                # Run physics update
                dt = 1.0 / settings.get('GENERAL.control_frequency', 100.0)
                self.physics.update(dt)
            
        except Exception as e:
            logger.error(f"Failed to update physics from hub: {e}")
    
    def get_drone_state(self) -> Dict[str, Any]:
        """
        Get complete drone state for rendering
        
        Returns:
            Dictionary with position, orientation, velocities, and engine states
        """
        try:
            state = self.physics.get_state_dict()
            
            # Add derived values for rendering
            euler_angles = self.physics.get_euler_angles()
            
            # Convert to rendering-friendly format
            drone_state = {
                'position': {
                    'x': float(state['position'][0]),
                    'y': float(state['position'][1]),
                    'z': float(state['position'][2])
                },
                'orientation': {
                    'roll': float(euler_angles[0]),
                    'pitch': float(euler_angles[1]), 
                    'yaw': float(euler_angles[2]),
                    'quaternion': state['quaternion'].tolist()
                },
                'velocity': {
                    'linear': state['velocity'].tolist(),
                    'angular': state['angular_velocity'].tolist(),
                    'speed': float(np.linalg.norm(state['velocity']))
                },
                'engines': {
                    'thrusts': state['engine_thrusts'].tolist(),
                    'voltages': self.hub.get_output_data(),
                    'rpms': self._calculate_rpms(state['engine_thrusts'])
                },
                'forces': {
                    'total_force': state['total_force_world'].tolist(),
                    'total_moment': state['total_moment_body'].tolist()
                }
            }
            
            return drone_state
            
        except Exception as e:
            logger.error(f"Failed to get drone state: {e}")
            return self._get_default_drone_state()
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Get formatted sensor data for display
        
        Returns:
            Dictionary with all sensor readings formatted for UI
        """
        hub_input = self.hub.get_input_data()
        
        sensor_data = {
            'gps': self._format_gps_data(hub_input.get('gps')),
            'barometer': self._format_barometer_data(hub_input.get('barometer')),
            'gyroscope': self._format_gyroscope_data(hub_input.get('gyroscope')),
            'temperature': self._format_temperature_data(hub_input.get('temperature')),
            'compass': self._format_compass_data(hub_input.get('compass')),
            'anemometer': self._format_anemometer_data(hub_input.get('anemometer')),
            'cameras': self._format_camera_data(hub_input.get('cameras'))
        }
        
        return sensor_data
    
    def get_environment_data(self) -> Dict[str, Any]:
        """
        Get environment data for rendering (cached for performance)
        """
        current_time = time.time()
        
        if (self._cached_environment_data is None or 
            current_time > self._cache_expire_time):
            
            try:
                drone_position = self.physics.state.position
                
                # Get ground height and terrain data
                ground_height = self.environment.get_ground_height(drone_position[:2])
                altitude_agl = drone_position[2] - ground_height
                
                # Get environmental effects
                wind_data = self.environment.get_wind_at_position(drone_position)
                air_density = self.environment.get_air_density(drone_position[2])
                
                self._cached_environment_data = {
                    'ground_height': float(ground_height),
                    'altitude_agl': float(altitude_agl),
                    'altitude_msl': float(drone_position[2]),
                    'wind': {
                        'velocity': wind_data.tolist() if hasattr(wind_data, 'tolist') else [0, 0, 0],
                        'speed': float(np.linalg.norm(wind_data)) if hasattr(wind_data, 'shape') else 0.0
                    },
                    'air_density': float(air_density),
                    'temperature': 15.0 + (ground_height - drone_position[2]) * 0.0065,  # Standard atmosphere
                    'pressure': 101325.0 * (1 - 0.0065 * drone_position[2] / 288.15) ** 5.255
                }
                
                self._cache_expire_time = current_time + self._cache_duration
                
            except Exception as e:
                logger.error(f"Failed to get environment data: {e}")
                self._cached_environment_data = self._get_default_environment_data()
        
        return self._cached_environment_data
    
    def render_environment(self) -> Dict[str, Any]:
        """
        Prepare environment rendering data
        """
        try:
            drone_position = self.physics.state.position
            
            # Generate terrain mesh around drone
            terrain_data = self._generate_terrain_mesh(drone_position)
            
            # Get obstacles from environment scanning
            obstacle_data = self._get_obstacle_data(drone_position)
            
            # Sky and lighting
            lighting_data = self._get_lighting_data()
            
            return {
                'terrain': terrain_data,
                'obstacles': obstacle_data,
                'lighting': lighting_data,
                'weather': self.get_environment_data()['wind']
            }
            
        except Exception as e:
            logger.error(f"Failed to render environment: {e}")
            return {}
    
    def render_drone(self) -> Dict[str, Any]:
        """
        Prepare drone rendering data including 3D model information
        """
        try:
            drone_state = self.get_drone_state()
            
            # Add 3D model information
            model_data = {
                'body': self._get_drone_body_data(),
                'propellers': self._get_propeller_data(drone_state['engines']['rpms']),
                'landing_gear': self._get_landing_gear_data(),
                'lights': self._get_drone_lights_data()
            }
            
            return {
                'state': drone_state,
                'model': model_data,
                'effects': self._get_visual_effects(drone_state)
            }
            
        except Exception as e:
            logger.error(f"Failed to render drone: {e}")
            return {}
    
    def should_render(self) -> bool:
        """Check if it's time for next render frame"""
        current_time = time.time()
        render_interval = 1.0 / self.render_frequency
        
        if current_time - self.last_render_time >= render_interval:
            self.last_render_time = current_time
            return True
        return False
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get formatted sensor data for display"""
        hub_input = self.hub.get_input_data()
        
        sensor_data = {
            'gps': self._format_gps_data(hub_input.get('gps')),
            'barometer': self._format_barometer_data(hub_input.get('barometer')),
            'gyroscope': self._format_gyroscope_data(hub_input.get('gyroscope')),
            'temperature': self._format_temperature_data(hub_input.get('temperature')),
            'compass': self._format_compass_data(hub_input.get('compass')),
            'anemometer': self._format_anemometer_data(hub_input.get('anemometer')),
            'cameras': self._format_camera_data(hub_input.get('cameras'))
        }
        
        return sensor_data
    
    def get_environment_data(self) -> Dict[str, Any]:
        """Get environment data for rendering (cached for performance)"""
        current_time = time.time()
        
        if (self._cached_environment_data is None or 
            current_time > self._cache_expire_time):
            
            try:
                drone_position = self.physics.state.position
                
                # Get ground height and terrain data
                ground_height = self.environment.get_ground_height(drone_position[:2])
                altitude_agl = drone_position[2] - ground_height
                
                # Get environmental effects
                wind_data = self.environment.get_wind_at_position(drone_position)
                air_density = self.environment.get_air_density(drone_position[2])
                
                self._cached_environment_data = {
                    'ground_height': float(ground_height),
                    'altitude_agl': float(altitude_agl),
                    'altitude_msl': float(drone_position[2]),
                    'wind': {
                        'velocity': wind_data.tolist() if hasattr(wind_data, 'tolist') else [0, 0, 0],
                        'speed': float(np.linalg.norm(wind_data)) if hasattr(wind_data, 'shape') else 0.0
                    },
                    'air_density': float(air_density),
                    'temperature': 15.0 + (ground_height - drone_position[2]) * 0.0065,  # Standard atmosphere
                    'pressure': 101325.0 * (1 - 0.0065 * drone_position[2] / 288.15) ** 5.255
                }
                
                self._cache_expire_time = current_time + self._cache_duration
                
            except Exception as e:
                logger.error(f"Failed to get environment data: {e}")
                self._cached_environment_data = self._get_default_environment_data()
        
        return self._cached_environment_data
    
    def render_environment(self) -> Dict[str, Any]:
        """Prepare environment rendering data"""
        try:
            drone_position = self.physics.state.position
            
            # Generate terrain mesh around drone
            terrain_data = self._generate_terrain_mesh(drone_position)
            
            # Get obstacles from environment scanning
            obstacle_data = self._get_obstacle_data(drone_position)
            
            # Sky and lighting
            lighting_data = self._get_lighting_data()
            
            return {
                'terrain': terrain_data,
                'obstacles': obstacle_data,
                'lighting': lighting_data,
                'weather': self.get_environment_data()['wind']
            }
            
        except Exception as e:
            logger.error(f"Failed to render environment: {e}")
            return {}
    
    def render_drone(self) -> Dict[str, Any]:
        """Prepare drone rendering data including 3D model information"""
        try:
            drone_state = self.get_drone_state()
            
            # Add 3D model information
            model_data = {
                'body': self._get_drone_body_data(),
                'propellers': self._get_propeller_data(drone_state['engines']['rpms']),
                'landing_gear': self._get_landing_gear_data(),
                'lights': self._get_drone_lights_data()
            }
            
            return {
                'state': drone_state,
                'model': model_data,
                'effects': self._get_visual_effects(drone_state)
            }
            
        except Exception as e:
            logger.error(f"Failed to render drone: {e}")
            return {}
    
    def _format_gps_data(self, gps_raw) -> Dict[str, Any]:
        """Format GPS data for display"""
        if not gps_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'latitude': gps_raw.get('latitude', 0.0),
            'longitude': gps_raw.get('longitude', 0.0),
            'altitude': gps_raw.get('altitude', 0.0),
            'speed': gps_raw.get('speed', 0.0),
            'heading': gps_raw.get('heading', 0.0),
            'satellites': gps_raw.get('satellites', 0),
            'hdop': gps_raw.get('hdop', 99.0)
        }
    
    def _format_barometer_data(self, baro_raw) -> Dict[str, Any]:
        """Format barometer data for display"""
        if not baro_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'pressure': baro_raw.get('pressure', 101325.0),
            'altitude': baro_raw.get('altitude', 0.0),
            'temperature': baro_raw.get('temperature', 15.0)
        }
    
    def _format_gyroscope_data(self, gyro_raw) -> Dict[str, Any]:
        """Format gyroscope data for display"""
        if not gyro_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'angular_velocity': gyro_raw.get('angular_velocity', [0, 0, 0]),
            'angular_velocity_deg': gyro_raw.get('angular_velocity_deg', [0, 0, 0])
        }
    
    def _format_temperature_data(self, temp_raw) -> Dict[str, Any]:
        """Format temperature data for display"""
        if not temp_raw:
            return {'status': 'no_data', 'temperature': 15.0}
        
        return {
            'status': 'active',
            'temperature': temp_raw.get('temperature', 15.0)
        }
    
    def _format_compass_data(self, compass_raw) -> Dict[str, Any]:
        """Format compass data for display"""
        if not compass_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'heading': compass_raw.get('heading', 0.0),
            'magnetic_declination': compass_raw.get('declination', 0.0)
        }
    
    def _format_anemometer_data(self, anem_raw) -> Dict[str, Any]:
        """Format anemometer data for display"""
        if not anem_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'wind_speed': anem_raw.get('wind_speed', 0.0),
            'wind_direction': anem_raw.get('wind_direction', 0.0)
        }
    
    def _format_camera_data(self, camera_raw) -> Dict[str, Any]:
        """Format camera data for display"""
        if not camera_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'frames': camera_raw if isinstance(camera_raw, list) else []
        }
    
    def _generate_terrain_mesh(self, center_position: np.ndarray) -> Dict[str, Any]:
        """Generate terrain mesh data around drone position"""
        # Create a simple grid around the drone
        grid_size = 50  # meters
        resolution = 2  # meters per vertex
        
        x_range = np.arange(-grid_size, grid_size + resolution, resolution)
        y_range = np.arange(-grid_size, grid_size + resolution, resolution)
        
        vertices = []
        for x in x_range:
            for y in y_range:
                world_x = center_position[0] + x
                world_y = center_position[1] + y
                # Pass full position to get_ground_height
                height = self.environment.get_ground_height(np.array([world_x, world_y]))
                vertices.append([float(world_x), float(world_y), float(height)])
        
        return {
            'vertices': vertices,
            'grid_size': grid_size,
            'resolution': resolution,
            'center': center_position[:2].tolist()
        }
    
    def _get_obstacle_data(self, position: np.ndarray) -> List[Dict[str, Any]]:
        """Get obstacle data for rendering"""
        # For now, return empty list - would interface with environment obstacle detection
        return []
    
    def _get_lighting_data(self) -> Dict[str, Any]:
        """Get lighting conditions for rendering"""
        return {
            'sun_direction': [0.3, 0.3, -0.9],  # Normalized sun direction
            'ambient_light': 0.3,
            'sun_intensity': 0.8,
            'sky_color': [0.5, 0.7, 1.0],
            'fog_density': 0.01
        }
    
    def _get_drone_body_data(self) -> Dict[str, Any]:
        """Get drone body model data"""
        return {
            'type': 'quadcopter',
            'arm_length': self.drone_cfg.get('arm_length', 0.225),
            'body_size': [0.1, 0.1, 0.05],  # [length, width, height] of central body
            'color': [0.2, 0.2, 0.2]  # Dark gray
        }
    
    def _get_propeller_data(self, rpms: List[float]) -> List[Dict[str, Any]]:
        """Get propeller visual data"""
        propellers = []
        positions = self.physics.config.engine_positions
        
        for i, (pos, rpm) in enumerate(zip(positions, rpms)):
            propellers.append({
                'position': pos.tolist(),
                'rpm': rpm,
                'radius': 0.1,  # Propeller radius in meters
                'blur_alpha': min(1.0, rpm / 3000.0)  # Visual blur based on RPM
            })
        
        return propellers
    
    def _get_landing_gear_data(self) -> Dict[str, Any]:
        """Get landing gear data"""
        return {
            'extended': True,  # Could be controlled based on flight state
            'height': 0.05,
            'color': [0.1, 0.1, 0.1]
        }
    
    def _get_drone_lights_data(self) -> List[Dict[str, Any]]:
        """Get drone navigation lights"""
        return [
            {'type': 'front', 'color': [1.0, 1.0, 1.0], 'position': [0.1, 0, 0], 'intensity': 0.8},
            {'type': 'rear', 'color': [1.0, 0.0, 0.0], 'position': [-0.1, 0, 0], 'intensity': 0.6},
            {'type': 'left', 'color': [0.0, 1.0, 0.0], 'position': [0, 0.1, 0], 'intensity': 0.5},
            {'type': 'right', 'color': [0.0, 1.0, 0.0], 'position': [0, -0.1, 0], 'intensity': 0.5}
        ]
    
    def _get_visual_effects(self, drone_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get visual effects data"""
        return {
            'engine_glow': [min(1.0, thrust/5.0) for thrust in drone_state['engines']['thrusts']],
            'motion_blur': min(1.0, drone_state['velocity']['speed'] / 10.0),
            'dust_particles': drone_state['position']['z'] < 2.0  # Show dust if close to ground
        }
    
    def _get_default_environment_data(self) -> Dict[str, Any]:
        """Default environment data when unavailable"""
        return {
            'ground_height': 0.0,
            'altitude_agl': 1.0,
            'altitude_msl': 1.0,
            'wind': {'velocity': [0, 0, 0], 'speed': 0.0},
            'air_density': 1.225,
            'temperature': 15.0,
            'pressure': 101325.0
        }
    
    def _calculate_rpms(self, thrusts: np.ndarray) -> List[float]:
        """Convert thrusts back to RPMs for visual effects"""
        rpms = []
        for thrust in thrusts:
            # Use propeller model to estimate RPM from thrust
            rpm = self.propeller.calculate_rpm_from_thrust(thrust)
            rpms.append(float(rpm))
        return rpms
    
    def _format_gps_data(self, gps_raw) -> Dict[str, Any]:
        """Format GPS data for display"""
        if not gps_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'latitude': gps_raw.get('latitude', 0.0),
            'longitude': gps_raw.get('longitude', 0.0),
            'altitude': gps_raw.get('altitude', 0.0),
            'speed': gps_raw.get('speed', 0.0),
            'heading': gps_raw.get('heading', 0.0),
            'satellites': gps_raw.get('satellites', 0),
            'hdop': gps_raw.get('hdop', 99.0)
        }
    
    def _format_barometer_data(self, baro_raw) -> Dict[str, Any]:
        """Format barometer data for display"""
        if not baro_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'pressure': baro_raw.get('pressure', 101325.0),
            'altitude': baro_raw.get('altitude', 0.0),
            'temperature': baro_raw.get('temperature', 15.0)
        }
    
    def _format_gyroscope_data(self, gyro_raw) -> Dict[str, Any]:
        """Format gyroscope data for display"""
        if not gyro_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'angular_velocity': gyro_raw.get('angular_velocity', [0, 0, 0]),
            'angular_velocity_deg': gyro_raw.get('angular_velocity_deg', [0, 0, 0])
        }
    
    def _format_temperature_data(self, temp_raw) -> Dict[str, Any]:
        """Format temperature data for display"""
        if not temp_raw:
            return {'status': 'no_data', 'temperature': 15.0}
        
        return {
            'status': 'active',
            'temperature': temp_raw.get('temperature', 15.0)
        }
    
    def _format_compass_data(self, compass_raw) -> Dict[str, Any]:
        """Format compass data for display"""
        if not compass_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'heading': compass_raw.get('heading', 0.0),
            'magnetic_declination': compass_raw.get('declination', 0.0)
        }
    
    def _format_anemometer_data(self, anem_raw) -> Dict[str, Any]:
        """Format anemometer data for display"""
        if not anem_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'wind_speed': anem_raw.get('wind_speed', 0.0),
            'wind_direction': anem_raw.get('wind_direction', 0.0)
        }
    
    def _format_camera_data(self, camera_raw) -> Dict[str, Any]:
        """Format camera data for display"""
        if not camera_raw:
            return {'status': 'no_data'}
        
        return {
            'status': 'active',
            'frames': camera_raw if isinstance(camera_raw, list) else []
        }
    
    def _generate_terrain_mesh(self, center_position: np.ndarray) -> Dict[str, Any]:
        """Generate terrain mesh data around drone position"""
        # Create a simple grid around the drone
        grid_size = 50  # meters
        resolution = 2  # meters per vertex
        
        x_range = np.arange(-grid_size, grid_size + resolution, resolution)
        y_range = np.arange(-grid_size, grid_size + resolution, resolution)
        
        vertices = []
        for x in x_range:
            for y in y_range:
                world_x = center_position[0] + x
                world_y = center_position[1] + y
                # Pass full position to get_ground_height
                height = self.environment.get_ground_height(np.array([world_x, world_y]))
                vertices.append([float(world_x), float(world_y), float(height)])
        
        return {
            'vertices': vertices,
            'grid_size': grid_size,
            'resolution': resolution,
            'center': center_position[:2].tolist()
        }
    
    def _get_obstacle_data(self, position: np.ndarray) -> List[Dict[str, Any]]:
        """Get obstacle data for rendering"""
        # For now, return empty list - would interface with environment obstacle detection
        return []
    
    def _get_lighting_data(self) -> Dict[str, Any]:
        """Get lighting conditions for rendering"""
        return {
            'sun_direction': [0.3, 0.3, -0.9],  # Normalized sun direction
            'ambient_light': 0.3,
            'sun_intensity': 0.8,
            'sky_color': [0.5, 0.7, 1.0],
            'fog_density': 0.01
        }
    
    def _get_drone_body_data(self) -> Dict[str, Any]:
        """Get drone body model data"""
        return {
            'type': 'quadcopter',
            'arm_length': self.drone_cfg.get('arm_length', 0.225),
            'body_size': [0.1, 0.1, 0.05],  # [length, width, height] of central body
            'color': [0.2, 0.2, 0.2]  # Dark gray
        }
    
    def _get_propeller_data(self, rpms: List[float]) -> List[Dict[str, Any]]:
        """Get propeller visual data"""
        propellers = []
        positions = self.physics.config.engine_positions
        
        for i, (pos, rpm) in enumerate(zip(positions, rpms)):
            propellers.append({
                'position': pos.tolist(),
                'rpm': rpm,
                'radius': 0.1,  # Propeller radius in meters
                'blur_alpha': min(1.0, rpm / 3000.0)  # Visual blur based on RPM
            })
        
        return propellers
    
    def _get_landing_gear_data(self) -> Dict[str, Any]:
        """Get landing gear data"""
        return {
            'extended': True,  # Could be controlled based on flight state
            'height': 0.05,
            'color': [0.1, 0.1, 0.1]
        }
    
    def _get_drone_lights_data(self) -> List[Dict[str, Any]]:
        """Get drone navigation lights"""
        return [
            {'type': 'front', 'color': [1.0, 1.0, 1.0], 'position': [0.1, 0, 0], 'intensity': 0.8},
            {'type': 'rear', 'color': [1.0, 0.0, 0.0], 'position': [-0.1, 0, 0], 'intensity': 0.6},
            {'type': 'left', 'color': [0.0, 1.0, 0.0], 'position': [0, 0.1, 0], 'intensity': 0.5},
            {'type': 'right', 'color': [0.0, 1.0, 0.0], 'position': [0, -0.1, 0], 'intensity': 0.5}
        ]
    
    def _get_visual_effects(self, drone_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get visual effects data"""
        return {
            'engine_glow': [min(1.0, thrust/5.0) for thrust in drone_state['engines']['thrusts']],
            'motion_blur': min(1.0, drone_state['velocity']['speed'] / 10.0),
            'dust_particles': drone_state['position']['z'] < 2.0  # Show dust if close to ground
        }
    
    def _get_default_drone_state(self) -> Dict[str, Any]:
        """Default drone state when physics unavailable"""
        return {
            'position': {'x': 0.0, 'y': 0.0, 'z': 1.0},
            'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'quaternion': [1, 0, 0, 0]},
            'velocity': {'linear': [0, 0, 0], 'angular': [0, 0, 0], 'speed': 0.0},
            'engines': {'thrusts': [0, 0, 0, 0], 'voltages': [0, 0, 0, 0], 'rpms': [0, 0, 0, 0]},
            'forces': {'total_force': [0, 0, 0], 'total_moment': [0, 0, 0]}
        }
    
    def _get_default_environment_data(self) -> Dict[str, Any]:
        """Default environment data when unavailable"""
        return {
            'ground_height': 0.0,
            'altitude_agl': 1.0,
            'altitude_msl': 1.0,
            'wind': {'velocity': [0, 0, 0], 'speed': 0.0},
            'air_density': 1.225,
            'temperature': 15.0,
            'pressure': 101325.0
        } 