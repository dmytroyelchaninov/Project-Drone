#!/usr/bin/env python3
"""
Physics Validation System

Validates simulation parameters against physical constraints and
provides automatic corrections for unrealistic values.
"""

import numpy as np
import warnings
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class PhysicsConstraint:
    """Represents a physical constraint with validation bounds"""
    name: str
    min_value: float
    max_value: float
    unit: str
    description: str
    critical: bool = False

@dataclass
class ValidationResult:
    """Result of physics validation"""
    is_valid: bool
    violations: List[str]
    corrections: Dict[str, Any]
    warnings: List[str]

class PhysicsValidator:
    """Validates and corrects physics parameters for drone simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_constraints()
    
    def _setup_constraints(self):
        """Setup physical constraints for drone parameters"""
        self.constraints = {
            # Real-time performance constraints
            'real_time_factor': PhysicsConstraint(
                'real_time_factor', 0.01, 100.0, 'ratio',
                'Real-time factor should be between 0.01x and 100x', critical=True
            ),
            'simulation_rate': PhysicsConstraint(
                'simulation_rate', 10.0, 100000.0, 'Hz',
                'Simulation rate should be between 10 Hz and 100 kHz', critical=True
            ),
            
            # Drone physical constraints
            'mass': PhysicsConstraint(
                'mass', 0.1, 100.0, 'kg',
                'Drone mass should be between 0.1 kg and 100 kg', critical=True
            ),
            'max_thrust': PhysicsConstraint(
                'max_thrust', 1.0, 1000.0, 'N',
                'Maximum thrust should be between 1 N and 1000 N', critical=True
            ),
            'max_angular_rate': PhysicsConstraint(
                'max_angular_rate', 10.0, 2000.0, 'deg/s',
                'Maximum angular rate should be between 10 and 2000 deg/s'
            ),
            
            # Flight performance constraints
            'max_velocity': PhysicsConstraint(
                'max_velocity', 0.1, 100.0, 'm/s',
                'Maximum velocity should be between 0.1 and 100 m/s'
            ),
            'max_acceleration': PhysicsConstraint(
                'max_acceleration', 0.1, 50.0, 'm/s²',
                'Maximum acceleration should be between 0.1 and 50 m/s²'
            ),
            'g_force': PhysicsConstraint(
                'g_force', 0.1, 10.0, 'g',
                'G-force should be between 0.1g and 10g for typical drones'
            ),
            
            # Control system constraints
            'settling_time': PhysicsConstraint(
                'settling_time', 0.1, 30.0, 's',
                'Settling time should be between 0.1s and 30s'
            ),
            'steady_state_error': PhysicsConstraint(
                'steady_state_error', 0.0, 1.0, 'm',
                'Steady state error should be between 0 and 1 meter'
            ),
            'tracking_error': PhysicsConstraint(
                'tracking_error', 0.0, 5.0, 'm',
                'Tracking error should be between 0 and 5 meters'
            ),
            
            # Energy and power constraints
            'power_consumption': PhysicsConstraint(
                'power_consumption', 10.0, 10000.0, 'W',
                'Power consumption should be between 10W and 10kW'
            ),
            'energy_efficiency': PhysicsConstraint(
                'energy_efficiency', 0.1, 1.0, 'ratio',
                'Energy efficiency should be between 10% and 100%'
            ),
            
            # Environmental constraints
            'wind_speed': PhysicsConstraint(
                'wind_speed', 0.0, 50.0, 'm/s',
                'Wind speed should be between 0 and 50 m/s'
            ),
            'altitude': PhysicsConstraint(
                'altitude', -100.0, 10000.0, 'm',
                'Altitude should be between -100m and 10km'
            ),
            
            # Time and duration constraints
            'completion_time': PhysicsConstraint(
                'completion_time', 0.1, 3600.0, 's',
                'Completion time should be between 0.1s and 1 hour'
            ),
            'recovery_time': PhysicsConstraint(
                'recovery_time', 0.1, 60.0, 's',
                'Recovery time should be between 0.1s and 60s'
            )
        }
    
    def validate_test_results(self, test_name: str, results: Dict[str, Any]) -> ValidationResult:
        """Validate test results against physical constraints"""
        violations = []
        corrections = {}
        warnings_list = []
        
        # Recursively validate all numeric values
        self._validate_recursive(results, "", violations, corrections, warnings_list)
        
        # Specific validations for known test types
        if "Performance Stress" in test_name:
            self._validate_performance_test(results, violations, corrections, warnings_list)
        elif "Figure-8" in test_name or "figure_eight" in str(results):
            self._validate_maneuver_test(results, violations, corrections, warnings_list)
        elif "Formation" in test_name:
            self._validate_formation_test(results, violations, corrections, warnings_list)
        
        is_valid = len([v for v in violations if "CRITICAL" in v]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            corrections=corrections,
            warnings=warnings_list
        )
    
    def _validate_recursive(self, data: Any, path: str, violations: List, corrections: Dict, warnings_list: List):
        """Recursively validate nested data structures"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._validate_recursive(value, new_path, violations, corrections, warnings_list)
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                self._validate_recursive(item, new_path, violations, corrections, warnings_list)
        elif isinstance(data, (int, float)):
            self._validate_single_value(path, data, violations, corrections, warnings_list)
    
    def _validate_single_value(self, path: str, value: float, violations: List, corrections: Dict, warnings_list: List):
        """Validate a single numeric value"""
        # Extract parameter name from path
        param_name = path.split('.')[-1].lower()
        
        # Check against known constraints
        for constraint_key, constraint in self.constraints.items():
            if constraint_key in param_name or param_name in constraint_key:
                if not (constraint.min_value <= value <= constraint.max_value):
                    severity = "CRITICAL" if constraint.critical else "WARNING"
                    violation_msg = f"{severity}: {path} = {value} {constraint.unit} violates {constraint.description}"
                    violations.append(violation_msg)
                    
                    # Suggest correction
                    corrected_value = np.clip(value, constraint.min_value, constraint.max_value)
                    corrections[path] = corrected_value
                    
                    if constraint.critical:
                        self.logger.error(violation_msg)
                    else:
                        self.logger.warning(violation_msg)
                break
        
        # Special case validations
        self._validate_special_cases(path, value, violations, corrections, warnings_list)
    
    def _validate_special_cases(self, path: str, value: float, violations: List, corrections: Dict, warnings_list: List):
        """Validate special physical cases"""
        param_name = path.split('.')[-1].lower()
        
        # Efficiency values should be <= 1.0
        if 'efficiency' in param_name and value > 1.0:
            violations.append(f"CRITICAL: {path} = {value} exceeds 100% efficiency (physically impossible)")
            corrections[path] = min(value, 1.0)
        
        # Real-time factors > 100x are suspicious
        if 'rtf' in param_name or 'real_time' in param_name:
            if value > 100.0:
                violations.append(f"CRITICAL: {path} = {value}x real-time is unrealistic for accurate simulation")
                corrections[path] = min(value, 100.0)
        
        # Negative values where they shouldn't be
        if any(keyword in param_name for keyword in ['time', 'duration', 'rate', 'speed', 'power']):
            if value < 0:
                violations.append(f"CRITICAL: {path} = {value} cannot be negative")
                corrections[path] = abs(value)
        
        # Extremely small values that might indicate calculation errors
        if value != 0 and abs(value) < 1e-10:
            warnings_list.append(f"WARNING: {path} = {value} is extremely small, possible numerical precision issue")
    
    def _validate_performance_test(self, results: Dict, violations: List, corrections: Dict, warnings_list: List):
        """Validate performance test specific constraints"""
        # Check for unrealistic real-time factors
        if 'realtime_factors' in results:
            for rtf_key, rtf_data in results['realtime_factors'].items():
                if isinstance(rtf_data, dict) and 'efficiency' in rtf_data:
                    efficiency = rtf_data['efficiency']
                    if efficiency > 100.0:
                        violations.append(f"CRITICAL: RTF {rtf_key} efficiency {efficiency} is physically impossible")
                        corrections[f"realtime_factors.{rtf_key}.efficiency"] = min(efficiency, 10.0)
    
    def _validate_maneuver_test(self, results: Dict, violations: List, corrections: Dict, warnings_list: List):
        """Validate maneuver test specific constraints"""
        # Check completion time reasonableness
        if 'completion_time' in results and results['completion_time'] > 300:
            violations.append(f"WARNING: Completion time {results['completion_time']}s seems excessive for maneuver")
        
        # Check G-force limits
        if 'g_force_peak' in results and results['g_force_peak'] > 5.0:
            violations.append(f"WARNING: Peak G-force {results['g_force_peak']}g exceeds typical drone limits")
    
    def _validate_formation_test(self, results: Dict, violations: List, corrections: Dict, warnings_list: List):
        """Validate formation flying specific constraints"""
        # Check for unrealistic precision
        if 'formation_accuracy' in results:
            accuracy = results['formation_accuracy']
            if accuracy < 1e-10 and accuracy != 0:
                warnings_list.append(f"WARNING: Formation accuracy {accuracy} is unrealistically precise")
    
    def apply_corrections(self, data: Dict[str, Any], corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Apply corrections to data"""
        corrected_data = data.copy()
        
        for path, corrected_value in corrections.items():
            self._set_nested_value(corrected_data, path, corrected_value)
        
        return corrected_data
    
    def _set_nested_value(self, data: Dict, path: str, value: Any):
        """Set a value in nested dictionary using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def generate_physics_report(self, validation_results: List[Tuple[str, ValidationResult]]) -> str:
        """Generate a comprehensive physics validation report"""
        report = []
        report.append("="*80)
        report.append("🔬 PHYSICS VALIDATION REPORT")
        report.append("="*80)
        
        total_violations = sum(len(result.violations) for _, result in validation_results)
        critical_violations = sum(len([v for v in result.violations if "CRITICAL" in v]) 
                                for _, result in validation_results)
        
        report.append(f"📊 Summary:")
        report.append(f"   Tests analyzed: {len(validation_results)}")
        report.append(f"   Total violations: {total_violations}")
        report.append(f"   Critical violations: {critical_violations}")
        report.append("")
        
        for test_name, result in validation_results:
            if result.violations or result.warnings:
                report.append(f"🧪 {test_name}:")
                
                for violation in result.violations:
                    if "CRITICAL" in violation:
                        report.append(f"   🚨 {violation}")
                    else:
                        report.append(f"   ⚠️  {violation}")
                
                for warning in result.warnings:
                    report.append(f"   💡 {warning}")
                
                if result.corrections:
                    report.append(f"   🔧 Suggested corrections:")
                    for path, value in result.corrections.items():
                        report.append(f"      {path} → {value}")
                
                report.append("")
        
        report.append("="*80)
        return "\n".join(report)

# Global validator instance
physics_validator = PhysicsValidator() 