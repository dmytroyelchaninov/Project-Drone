"""
Drone Simulation Package

A physics-accurate, reactive simulator for multirotor drones with acoustic analysis capabilities.
"""

__version__ = "0.1.0"
__author__ = "Drone Simulation Team"
__email__ = "info@dronesim.com"

# Core imports
from .core.simulator import Simulator, SimulationConfig, SimulationState
from .core.state_manager import StateManager, DroneState
from .core.event_system import EventSystem, Event, EventType

# Physics imports
from .physics.rigid_body import RigidBody, RigidBodyConfig
from .physics.environment import Environment, EnvironmentConfig
from .physics.aerodynamics.propeller import Propeller, PropellerConfig, PropellerArray
from .physics.aerodynamics.noise_model import PropellerNoiseModel, ObserverPosition

# Control imports
from .control.base_controller import BaseController, ControllerState, ControllerReference, ControllerOutput
from .control.pid_controller import PIDController

__all__ = [
    # Core
    'Simulator', 'SimulationConfig', 'SimulationState',
    'StateManager', 'DroneState',
    'EventSystem', 'Event', 'EventType',
    
    # Physics
    'RigidBody', 'RigidBodyConfig',
    'Environment', 'EnvironmentConfig',
    'Propeller', 'PropellerConfig', 'PropellerArray',
    'PropellerNoiseModel', 'ObserverPosition',
    
    # Control
    'BaseController', 'ControllerState', 'ControllerReference', 'ControllerOutput',
    'PIDController',
] 