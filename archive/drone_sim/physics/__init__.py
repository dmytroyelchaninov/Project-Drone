"""
Physics Package

This package contains all physics-related components for the drone simulation,
including rigid body dynamics, aerodynamics, environment modeling, and physics validation.
"""

from .rigid_body import RigidBody, RigidBodyConfig
from .environment import Environment, EnvironmentConfig
from .aerodynamics.propeller import Propeller, PropellerConfig, PropellerArray
from .aerodynamics.noise_model import PropellerNoiseModel
from .physics_validator import PhysicsValidator, physics_validator, PhysicsConstraint, ValidationResult

__all__ = [
    'RigidBody',
    'RigidBodyConfig', 
    'Environment',
    'EnvironmentConfig',
    'Propeller',
    'PropellerConfig',
    'PropellerArray',
    'PropellerNoiseModel',
    'PhysicsValidator',
    'physics_validator',
    'PhysicsConstraint',
    'ValidationResult'
]
