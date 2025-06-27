"""
Main Drone Control Module
Handles transformation of unified input from sensors/devices and writes voltage output to HUB
"""
import numpy as np
import time
import threading
import logging
from typing import Dict, Any, Optional
from input.hub import Hub
from physics import QuadcopterPhysics, Environment
from cfg import settings

logger = logging.getLogger(__name__)

class Drone:
    """
    Main drone control system
    
    Takes unified input from sensors/devices (listens to HUB) and writes output voltage to HUB.
    Depending on mode, task, and go state, may be interrupted by AI (ignoring device input)
    and utilizing sensors to complete tasks, or AI may take control in danger situations.
    """
    
    def __init__(self):
        self.hub = Hub()
        self.physics = QuadcopterPhysics()
        self.environment = Environment()
        
        # Control state
        self._running = False
        self._control_thread = None
        self._last_update_time = time.time()
        
        # Control parameters
        self.control_frequency = settings.get('HUB.update_rate', 100)  # Hz
        self.emergency_enabled = settings.get('HUB.emergency_stop_enabled', True)
        self.watchdog_timeout = settings.get('HUB.watchdog_timeout', 2.0)
        
        # PID controllers for manual control assistance
        self._attitude_controller = self._init_attitude_controller()
        self._altitude_controller = self._init_altitude_controller()
        
        # AI system (placeholder for future implementation)
        self.ai_system = None
        
        # Safety monitoring
        self._last_input_time = time.time()
        self._danger_detected = False
        
        logger.info("Drone control system initialized")
    
    def start(self):
        """Start the drone control loop"""
        if self._running:
            logger.warning("Drone control already running")
            return True
        
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        logger.info("Drone control started")
        return True
    
    def stop(self):
        """Stop the drone control loop"""
        if not self._running:
            return
        
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        
        # Zero all outputs
        self.hub.update_output([0.0, 0.0, 0.0, 0.0])
        logger.info("Drone control stopped")
    
    def _control_loop(self):
        """Main control loop that runs continuously"""
        logger.info("Drone control loop started")
        
        while self._running:
            start_time = time.time()
            
            try:
                # Get current hub state
                hub_state = self.hub.get_state()
                
                # Check for danger conditions
                self._check_danger_conditions()
                
                # Determine control source and compute voltages
                output_voltages = self._compute_control_output(hub_state)
                
                # Apply safety limits
                safe_voltages = self._apply_safety_limits(output_voltages)
                
                # Update hub output
                self.hub.update_output(safe_voltages)
                
                # Update physics simulation
                if self.hub.simulation:
                    self._update_physics_simulation(safe_voltages)
                
                self._last_update_time = time.time()
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                # Emergency response
                self.hub.update_output([0.0, 0.0, 0.0, 0.0])
            
            # Maintain control frequency
            elapsed = time.time() - start_time
            target_interval = 1.0 / self.control_frequency
            sleep_time = max(0, target_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Drone control loop ended")
    
    def _compute_control_output(self, hub_state: Dict[str, Any]) -> list:
        """
        Compute control output based on current mode, task, and go state
        
        Returns:
            List of voltages for each engine [engine0, engine1, engine2, engine3]
        """
        mode = hub_state['mode']
        go_state = hub_state['go']
        task = hub_state['task']
        
        # Handle different go states
        if go_state == 'off':
            return [0.0, 0.0, 0.0, 0.0]
        elif go_state == 'idle':
            return self._compute_idle_output()
        elif go_state == 'float':
            return self._compute_float_output()
        elif go_state == 'operate':
            # Active operation - use mode-specific control
            if self._danger_detected:
                logger.warning("Danger detected - AI taking control")
                return self._compute_ai_emergency_control()
            elif mode == 'manual':
                return self._compute_manual_control()
            elif mode == 'hybrid':
                return self._compute_hybrid_control(task)
            elif mode == 'ai':
                return self._compute_ai_control(task)
            else:
                logger.error(f"Unknown control mode: {mode}")
                return self._compute_float_output()
        else:
            logger.error(f"Unknown go state: {go_state}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _compute_manual_control(self) -> list:
        """Compute manual control from device input"""
        device_voltages = self.hub.input.get('device_voltages')
        
        if device_voltages is None:
            logger.warning("No device input - switching to float mode")
            return self._compute_float_output()
        
        # Check input freshness
        device_status = self.hub.input.get('device_status', {})
        last_update = device_status.get('last_update', 0)
        input_age = time.time() - last_update
        
        if input_age > self.watchdog_timeout:
            logger.warning(f"Device input timeout ({input_age:.1f}s) - switching to float mode")
            return self._compute_float_output()
        
        # Use device voltages directly (they come from KeyboardDevice/VoltageController)
        return list(device_voltages[:4])  # Ensure exactly 4 engines
    
    def _compute_hybrid_control(self, task: Optional[str]) -> list:
        """Compute hybrid control (AI assistance + manual input)"""
        # Get manual input
        manual_voltages = self._compute_manual_control()
        
        # Get AI suggestions
        ai_adjustments = self._compute_ai_adjustments(task)
        
        # Blend manual and AI control
        blended_voltages = []
        for manual_v, ai_adj in zip(manual_voltages, ai_adjustments):
            # AI can modify manual input by ±20%
            adjusted_v = manual_v + ai_adj * 0.2
            blended_voltages.append(adjusted_v)
        
        return blended_voltages
    
    def _compute_ai_control(self, task: Optional[str]) -> list:
        """Compute full AI control for given task"""
        if task == 'take_off':
            return self._ai_takeoff_control()
        elif task == 'land':
            return self._ai_landing_control()
        elif task == 'follow':
            return self._ai_follow_control()
        elif task == 'back_to_base':
            return self._ai_return_to_base_control()
        elif task == 'projectile':
            return self._ai_projectile_control()
        else:
            # No specific task - maintain position/hover
            return self._ai_hover_control()
    
    def _compute_idle_output(self) -> list:
        """Compute output for idle mode (minimal power)"""
        # Very low RPM to keep engines spinning
        idle_voltage = 1.0  # Volts
        return [idle_voltage, idle_voltage, idle_voltage, idle_voltage]
    
    def _compute_float_output(self) -> list:
        """Compute output for float mode (hover in place)"""
        # Calculate hover thrust needed
        total_weight = settings.get('DRONE.mass', 1.5) + settings.get('DRONE.cargo_mass', 0.0)
        gravity = settings.get('ENVIRONMENT.gravity', 9.81)
        total_thrust_needed = total_weight * gravity
        
        # Distribute equally among 4 engines
        thrust_per_engine = total_thrust_needed / 4.0
        
        # Convert thrust to voltage (simplified model)
        hover_voltage = settings.get('DRONE.engines.hover_voltage', 6.0)
        
        # Apply ground effect if close to ground
        try:
            position = self.physics.state.position
            ground_effect = self.environment.get_ground_effect_factor(position)
            effective_voltage = hover_voltage / ground_effect
        except:
            effective_voltage = hover_voltage
        
        return [effective_voltage, effective_voltage, effective_voltage, effective_voltage]
    
    def _compute_ai_emergency_control(self) -> list:
        """Emergency AI control when danger is detected"""
        # Emergency response depends on the type of danger
        # For now, implement emergency landing
        return self._ai_emergency_landing()
    
    def _ai_hover_control(self) -> list:
        """AI hover control using sensor feedback"""
        # Get sensor data
        gyro_data = self.hub.input.get('gyroscope')
        baro_data = self.hub.input.get('barometer')
        
        base_voltage = settings.get('DRONE.engines.hover_voltage', 6.0)
        
        if gyro_data and baro_data:
            # Simple PID control for attitude and altitude
            attitude_correction = self._attitude_controller.compute(gyro_data)
            altitude_correction = self._altitude_controller.compute(baro_data)
            
            # Apply corrections to base hover voltage
            voltages = [base_voltage] * 4
            
            # Apply attitude corrections (simplified)
            voltages[0] += attitude_correction[0]  # Front
            voltages[1] += attitude_correction[1]  # Right  
            voltages[2] -= attitude_correction[0]  # Back
            voltages[3] -= attitude_correction[1]  # Left
            
            # Apply altitude correction to all engines
            for i in range(4):
                voltages[i] += altitude_correction
            
            return voltages
        else:
            # No sensor data - use basic hover
            return [base_voltage] * 4
    
    def _ai_takeoff_control(self) -> list:
        """AI takeoff sequence"""
        # Simple takeoff: gradually increase thrust
        current_altitude = 0.0
        
        baro_data = self.hub.input.get('barometer')
        if baro_data:
            current_altitude = baro_data.get('altitude_agl', 0.0)
        
        target_altitude = 2.0  # meters
        hover_voltage = settings.get('DRONE.engines.hover_voltage', 6.0)
        
        if current_altitude < target_altitude:
            # Increase thrust for takeoff
            takeoff_voltage = hover_voltage * 1.2
            return [takeoff_voltage] * 4
        else:
            # Reached target altitude - switch to hover
            self.hub.set_task(None)  # Clear takeoff task
            return self._ai_hover_control()
    
    def _ai_landing_control(self) -> list:
        """AI landing sequence"""
        current_altitude = 2.0
        
        baro_data = self.hub.input.get('barometer')
        if baro_data:
            current_altitude = baro_data.get('altitude_agl', 2.0)
        
        hover_voltage = settings.get('DRONE.engines.hover_voltage', 6.0)
        
        if current_altitude > 0.1:
            # Gradually reduce thrust for controlled descent
            landing_voltage = hover_voltage * 0.8
            return [landing_voltage] * 4
        else:
            # Landed - turn off engines
            self.hub.set_task(None)  # Clear landing task
            self.hub.set_go('off')
            return [0.0] * 4
    
    def _ai_emergency_landing(self) -> list:
        """Emergency landing procedure"""
        hover_voltage = settings.get('DRONE.engines.hover_voltage', 6.0)
        emergency_voltage = hover_voltage * 0.6  # Controlled descent
        return [emergency_voltage] * 4
    
    def _ai_follow_control(self) -> list:
        """AI follow mode (placeholder)"""
        # This would implement following a target
        # For now, just hover
        return self._ai_hover_control()
    
    def _ai_return_to_base_control(self) -> list:
        """AI return to base (placeholder)"""
        # This would implement navigation to home position
        # For now, just hover and then land
        return self._ai_landing_control()
    
    def _ai_projectile_control(self) -> list:
        """AI projectile mode (placeholder)"""
        # This would implement ballistic trajectory
        # For now, just hover
        return self._ai_hover_control()
    
    def _compute_ai_adjustments(self, task: Optional[str]) -> list:
        """Compute AI adjustments for hybrid mode"""
        # Placeholder for AI assistance
        # Would provide small corrections to manual input
        return [0.0, 0.0, 0.0, 0.0]
    
    def _check_danger_conditions(self):
        """Check for dangerous conditions that require AI intervention"""
        self._danger_detected = False
        
        # Check battery voltage (if implemented)
        # Check altitude limits
        try:
            position = self.physics.state.position
            altitude = position[2]
            max_altitude = settings.get('ENVIRONMENT.max_altitude', 1000.0)
            
            if altitude > max_altitude:
                logger.warning(f"Altitude limit exceeded: {altitude:.1f}m > {max_altitude}m")
                self._danger_detected = True
            
            if altitude < 0:
                logger.warning(f"Ground collision detected: altitude = {altitude:.1f}m")
                self._danger_detected = True
        except:
            pass
        
        # Check angular rates (if too high, could lose control)
        gyro_data = self.hub.input.get('gyroscope')
        if gyro_data:
            angular_rates = gyro_data.get('angular_velocity_rad', [0, 0, 0])
            max_rate = np.radians(180)  # 180 deg/s
            
            if any(abs(rate) > max_rate for rate in angular_rates):
                logger.warning("Excessive angular rates detected")
                self._danger_detected = True
    
    def _apply_safety_limits(self, voltages: list) -> list:
        """Apply safety limits to output voltages"""
        max_voltage = settings.get('DRONE.engines.max_voltage', 12.0)
        min_voltage = settings.get('DRONE.engines.min_voltage', 0.0)
        
        safe_voltages = []
        for v in voltages:
            safe_v = max(min_voltage, min(max_voltage, v))
            safe_voltages.append(safe_v)
        
        return safe_voltages
    
    def _update_physics_simulation(self, voltages: list):
        """Update physics simulation with current voltages"""
        try:
            # Convert voltages to thrusts using propeller model
            from physics import Propeller, PropellerConfig
            propeller = Propeller(PropellerConfig())
            
            thrusts = []
            for voltage in voltages:
                thrust = propeller.calculate_thrust_from_voltage(voltage)
                thrusts.append(thrust)
            
            # Update physics
            self.physics.set_engine_thrusts(np.array(thrusts))
            self.physics.update(1.0 / self.control_frequency)
            
        except Exception as e:
            logger.error(f"Physics simulation update failed: {e}")
    
    def _init_attitude_controller(self):
        """Initialize attitude PID controller (placeholder)"""
        class SimpleAttitudeController:
            def compute(self, gyro_data):
                # Simple proportional control
                rates = gyro_data.get('angular_velocity_rad', [0, 0, 0])
                return [-rate * 0.1 for rate in rates[:2]]  # Only roll/pitch
        
        return SimpleAttitudeController()
    
    def _init_altitude_controller(self):
        """Initialize altitude PID controller (placeholder)"""
        class SimpleAltitudeController:
            def __init__(self):
                self.target_altitude = 2.0  # meters
            
            def compute(self, baro_data):
                current_alt = baro_data.get('altitude_agl', 0.0)
                error = self.target_altitude - current_alt
                return error * 0.5  # Simple proportional control
        
        return SimpleAltitudeController()
    
    def get_status(self) -> Dict[str, Any]:
        """Get drone control status"""
        return {
            'running': self._running,
            'last_update': self._last_update_time,
            'control_frequency': self.control_frequency,
            'danger_detected': self._danger_detected,
            'hub_state': self.hub.get_state(),
            'physics_state': self.physics.get_state_dict() if self.hub.simulation else None
        } 