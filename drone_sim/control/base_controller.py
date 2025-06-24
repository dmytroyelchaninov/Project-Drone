"""
Base controller interface for all flight controllers
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ControllerState:
    """State information for controllers"""
    position: np.ndarray = None
    quaternion: np.ndarray = None
    velocity: np.ndarray = None
    angular_velocity: np.ndarray = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)

@dataclass
class ControllerReference:
    """Reference commands for the controller"""
    position: np.ndarray = None
    velocity: np.ndarray = None
    acceleration: np.ndarray = None
    attitude: np.ndarray = None  # Quaternion or Euler angles
    angular_velocity: np.ndarray = None
    thrust: float = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.acceleration is None:
            self.acceleration = np.zeros(3)
        if self.attitude is None:
            self.attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)
        if self.thrust is None:
            self.thrust = 0.0

@dataclass
class ControllerOutput:
    """Output from the controller"""
    thrust: float = 0.0
    moment: np.ndarray = None
    motor_commands: np.ndarray = None  # Individual motor commands
    
    def __post_init__(self):
        if self.moment is None:
            self.moment = np.zeros(3)

class BaseController(ABC):
    """
    Abstract base class for all flight controllers
    """
    
    def __init__(self, controller_name: str = "BaseController"):
        self.name = controller_name
        self.enabled = True
        self.parameters = {}
        self.internal_state = {}
        
        # Performance metrics
        self.control_effort = 0.0
        self.tracking_error = 0.0
        self.update_count = 0
        
    @abstractmethod
    def update(self, reference: ControllerReference, 
               current_state: ControllerState, 
               dt: float) -> ControllerOutput:
        """
        Update the controller
        
        Args:
            reference: Reference/desired state
            current_state: Current drone state
            dt: Time step in seconds
            
        Returns:
            Control commands
        """
        pass
        
    @abstractmethod
    def reset(self):
        """Reset controller internal state"""
        pass
        
    def enable(self):
        """Enable the controller"""
        self.enabled = True
        
    def disable(self):
        """Disable the controller"""
        self.enabled = False
        
    def set_parameter(self, name: str, value: Any):
        """
        Set a controller parameter
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
        
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a controller parameter
        
        Args:
            name: Parameter name
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value
        """
        return self.parameters.get(name, default)
        
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set multiple parameters
        
        Args:
            params: Dictionary of parameter names and values
        """
        self.parameters.update(params)
        
    @property
    def ml_interface(self) -> Dict[str, Any]:
        """
        Expose controller internals for ML integration
        
        Returns:
            Dictionary with controller weights, state, and metrics
        """
        return {
            'parameters': self.parameters.copy(),
            'internal_state': self.internal_state.copy(),
            'enabled': self.enabled,
            'control_effort': self.control_effort,
            'tracking_error': self.tracking_error,
            'update_count': self.update_count
        }
        
    def load_ml_state(self, ml_state: Dict[str, Any]):
        """
        Load state from ML interface
        
        Args:
            ml_state: Dictionary with ML-optimized parameters
        """
        if 'parameters' in ml_state:
            self.parameters.update(ml_state['parameters'])
        if 'internal_state' in ml_state:
            self.internal_state.update(ml_state['internal_state'])
            
    def get_status(self) -> Dict[str, Any]:
        """
        Get controller status
        
        Returns:
            Status dictionary
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'parameters': self.parameters,
            'control_effort': self.control_effort,
            'tracking_error': self.tracking_error,
            'update_count': self.update_count
        }
        
    def compute_tracking_error(self, reference: ControllerReference, 
                             current_state: ControllerState) -> float:
        """
        Compute tracking error metric
        
        Args:
            reference: Reference state
            current_state: Current state
            
        Returns:
            Tracking error scalar
        """
        position_error = np.linalg.norm(reference.position - current_state.position)
        velocity_error = np.linalg.norm(reference.velocity - current_state.velocity)
        
        # Attitude error (simplified)
        attitude_error = 0.0
        if hasattr(reference, 'attitude') and reference.attitude is not None:
            # Assume quaternion representation
            q_ref = reference.attitude / np.linalg.norm(reference.attitude)
            q_curr = current_state.quaternion / np.linalg.norm(current_state.quaternion)
            
            # Quaternion error magnitude
            q_error = 1 - abs(np.dot(q_ref, q_curr))
            attitude_error = q_error
            
        # Combined error
        total_error = position_error + 0.1 * velocity_error + attitude_error
        
        return total_error
        
    def compute_control_effort(self, output: ControllerOutput) -> float:
        """
        Compute control effort metric
        
        Args:
            output: Controller output
            
        Returns:
            Control effort scalar
        """
        thrust_effort = abs(output.thrust)
        moment_effort = np.linalg.norm(output.moment)
        
        return thrust_effort + moment_effort
        
    def _update_metrics(self, reference: ControllerReference, 
                       current_state: ControllerState, 
                       output: ControllerOutput):
        """Update internal performance metrics"""
        self.tracking_error = self.compute_tracking_error(reference, current_state)
        self.control_effort = self.compute_control_effort(output)
        self.update_count += 1

class ControllerManager:
    """
    Manages multiple controllers and switching between them
    """
    
    def __init__(self):
        self.controllers: Dict[str, BaseController] = {}
        self.active_controller: Optional[str] = None
        self.switching_enabled = True
        
    def register_controller(self, name: str, controller: BaseController):
        """
        Register a controller
        
        Args:
            name: Controller name
            controller: Controller instance
        """
        self.controllers[name] = controller
        
        # Set as active if it's the first one
        if self.active_controller is None:
            self.active_controller = name
            
    def unregister_controller(self, name: str):
        """
        Unregister a controller
        
        Args:
            name: Controller name
        """
        if name in self.controllers:
            del self.controllers[name]
            
        if self.active_controller == name:
            # Switch to first available controller
            if self.controllers:
                self.active_controller = next(iter(self.controllers))
            else:
                self.active_controller = None
                
    def switch_controller(self, name: str) -> bool:
        """
        Switch to a different controller
        
        Args:
            name: Controller name to switch to
            
        Returns:
            True if switch was successful
        """
        if not self.switching_enabled:
            return False
            
        if name in self.controllers:
            self.active_controller = name
            return True
        return False
        
    def update(self, reference: ControllerReference, 
               current_state: ControllerState, 
               dt: float) -> Optional[ControllerOutput]:
        """
        Update the active controller
        
        Args:
            reference: Reference state
            current_state: Current state
            dt: Time step
            
        Returns:
            Controller output or None if no active controller
        """
        if self.active_controller and self.active_controller in self.controllers:
            controller = self.controllers[self.active_controller]
            if controller.enabled:
                return controller.update(reference, current_state, dt)
        return None
        
    def get_active_controller(self) -> Optional[BaseController]:
        """Get the currently active controller"""
        if self.active_controller:
            return self.controllers.get(self.active_controller)
        return None
        
    def get_controller_names(self) -> list:
        """Get list of registered controller names"""
        return list(self.controllers.keys())
        
    def reset_all(self):
        """Reset all controllers"""
        for controller in self.controllers.values():
            controller.reset() 