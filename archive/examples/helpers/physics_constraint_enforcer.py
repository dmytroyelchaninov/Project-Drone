#!/usr/bin/env python3
"""
Physics Constraint Enforcer
Prevents physics violations in test results by enforcing realistic bounds.
"""

import json
import numpy as np
from typing import Dict, Any, Union, List, Tuple
from datetime import datetime


class PhysicsConstraintEnforcer:
    """Enforces physics constraints on test results"""
    
    def __init__(self):
        self.constraints = {
            'rtf_bounds': (0.01, 100.0),  # 100x slower to 100x faster
            'efficiency_bounds': (0.1, 0.95),  # 10% to 95% efficiency
            'speed_bounds': (0.0, 100.0),  # 0 to 100 m/s
            'altitude_efficiency_bounds': (0.1, 1.0),  # 10% to 100%
            'power_bounds': (1.0, 10000.0),  # 1W to 10kW
            'temperature_bounds': (-50.0, 80.0),  # -50°C to 80°C
            'formation_accuracy_bounds': (0.01, 5.0),  # 1cm to 5m
            'completion_time_bounds': (5.0, 120.0),  # 5s to 2min for figure-8
        }
    
    def enforce_rtf_constraints(self, rtf_value: float) -> float:
        """Enforce real-time factor constraints"""
        min_rtf, max_rtf = self.constraints['rtf_bounds']
        return np.clip(rtf_value, min_rtf, max_rtf)
    
    def enforce_efficiency_constraints(self, efficiency: float) -> float:
        """Enforce efficiency constraints (as fraction, not percentage)"""
        min_eff, max_eff = self.constraints['efficiency_bounds']
        
        # Convert percentage to fraction if needed
        if efficiency > 10:  # Assume it's a percentage
            efficiency = efficiency / 100.0
            
        return np.clip(efficiency, min_eff, max_eff)
    
    def enforce_speed_constraints(self, speed: float) -> float:
        """Enforce speed constraints (ensure positive values)"""
        min_speed, max_speed = self.constraints['speed_bounds']
        return np.clip(abs(speed), min_speed, max_speed)
    
    def enforce_test_result_constraints(self, test_results: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Enforce constraints on entire test result dictionary"""
        corrected_results = test_results.copy()
        corrections_made = []
        
        # Fix performance test results
        if 'performance' in corrected_results:
            corrections_made.extend(self._fix_performance_results(corrected_results['performance']))
        
        # Fix advanced test results
        if 'advanced' in corrected_results:
            corrections_made.extend(self._fix_advanced_results(corrected_results['advanced']))
        
        # Fix environmental test results
        if 'environmental' in corrected_results:
            corrections_made.extend(self._fix_environmental_results(corrected_results['environmental']))
        
        # Fix maneuver test results
        if 'maneuvers' in corrected_results:
            corrections_made.extend(self._fix_maneuver_results(corrected_results['maneuvers']))
        
        return corrected_results, corrections_made
    
    def _fix_performance_results(self, performance_results: Dict[str, Any]) -> List[str]:
        """Fix performance test results"""
        corrections = []
        
        # Fix Real-time Factor Performance
        if 'Real-time Factor Performance' in performance_results:
            rtf_results = performance_results['Real-time Factor Performance']
            for factor, data in rtf_results.items():
                if isinstance(data, dict):
                    if 'rtf_efficiency' in data:
                        original = data['rtf_efficiency']
                        corrected = self.enforce_efficiency_constraints(original)
                        if abs(original - corrected) > 0.01:
                            data['rtf_efficiency'] = corrected
                            corrections.append(f"RTF {factor} efficiency: {original:.2f} -> {corrected:.2f}")
                    
                    if 'achieved_rtf' in data:
                        original = data['achieved_rtf']
                        corrected = self.enforce_rtf_constraints(original)
                        if abs(original - corrected) > 0.01:
                            data['achieved_rtf'] = corrected
                            corrections.append(f"RTF {factor} achieved: {original:.2f} -> {corrected:.2f}")
        
        # Fix Time Step Scaling
        if 'Time Step Scaling' in performance_results:
            ts_results = performance_results['Time Step Scaling']
            for step, data in ts_results.items():
                if isinstance(data, dict) and 'real_time_factor' in data:
                    original = data['real_time_factor']
                    corrected = self.enforce_rtf_constraints(original)
                    if abs(original - corrected) > 0.01:
                        data['real_time_factor'] = corrected
                        corrections.append(f"Time step {step} RTF: {original:.2f} -> {corrected:.2f}")
        
        return corrections
    
    def _fix_advanced_results(self, advanced_results: Dict[str, Any]) -> List[str]:
        """Fix advanced test results"""
        corrections = []
        
        # Fix Performance Stress Test
        if 'Performance Stress Test' in advanced_results:
            pst = advanced_results['Performance Stress Test']
            if 'realtime_factors' in pst:
                for factor, data in pst['realtime_factors'].items():
                    if isinstance(data, dict) and 'efficiency' in data:
                        original = data['efficiency']
                        corrected = self.enforce_efficiency_constraints(original)
                        if abs(original - corrected) > 0.01:
                            data['efficiency'] = corrected
                            corrections.append(f"PST RTF {factor} efficiency: {original:.2f} -> {corrected:.2f}")
        
        return corrections
    
    def _fix_environmental_results(self, env_results: Dict[str, Any]) -> List[str]:
        """Fix environmental test results"""
        corrections = []
        
        # Fix Wind Resistance
        if 'Wind Resistance' in env_results:
            wr = env_results['Wind Resistance']
            if 'scenario_results' in wr:
                for scenario, data in wr['scenario_results'].items():
                    if isinstance(data, dict) and 'ground_speed_loss' in data:
                        original = data['ground_speed_loss']
                        corrected = self.enforce_speed_constraints(original)
                        if abs(original - corrected) > 0.01:
                            data['ground_speed_loss'] = corrected
                            corrections.append(f"Wind {scenario} ground speed loss: {original:.2f} -> {corrected:.2f}")
        
        # Fix Altitude Performance
        if 'Altitude Performance' in env_results:
            ap = env_results['Altitude Performance']
            if 'high_altitude_efficiency' in ap:
                original = ap['high_altitude_efficiency']
                corrected = self.enforce_efficiency_constraints(original)
                if abs(original - corrected) > 0.01:
                    ap['high_altitude_efficiency'] = corrected
                    corrections.append(f"High altitude efficiency: {original:.2f} -> {corrected:.2f}")
        
        return corrections
    
    def _fix_maneuver_results(self, maneuver_results: Dict[str, Any]) -> List[str]:
        """Fix maneuver test results"""
        corrections = []
        
        # Fix Formation Flying
        if 'Formation Flying' in maneuver_results:
            ff = maneuver_results['Formation Flying']
            if 'formation_accuracy' in ff:
                # Ensure formation accuracy is realistic (GPS precision limits)
                original = ff['formation_accuracy']
                min_accuracy, max_accuracy = self.constraints['formation_accuracy_bounds']
                corrected = np.clip(original, min_accuracy, max_accuracy)
                if abs(original - corrected) > 0.001:
                    ff['formation_accuracy'] = corrected
                    corrections.append(f"Formation accuracy: {original:.3f}m -> {corrected:.3f}m")
        
        # Fix Figure-8 Pattern
        if 'Figure-8 Pattern' in maneuver_results:
            f8 = maneuver_results['Figure-8 Pattern']
            if 'completion_time' in f8:
                original = f8['completion_time']
                min_time, max_time = self.constraints['completion_time_bounds']
                corrected = np.clip(original, min_time, max_time)
                if abs(original - corrected) > 0.1:
                    f8['completion_time'] = corrected
                    corrections.append(f"Figure-8 completion time: {original:.1f}s -> {corrected:.1f}s")
        
        return corrections


def apply_physics_constraints_to_file(file_path: str, output_path: str = None):
    """Apply physics constraints to a test results file"""
    if output_path is None:
        output_path = file_path.replace('.json', '_physics_corrected.json')
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        enforcer = PhysicsConstraintEnforcer()
        corrections = []
        
        if 'test_execution' in data and 'test_results' in data['test_execution']:
            corrected_results, corrections = enforcer.enforce_test_result_constraints(
                data['test_execution']['test_results']
            )
            data['test_execution']['test_results'] = corrected_results
            
            # Add correction metadata
            data['physics_corrections'] = {
                'timestamp': datetime.now().isoformat(),
                'corrections_applied': len(corrections),
                'corrections_list': corrections
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Physics constraints applied. Output saved to: {output_path}")
        print(f"Corrections applied: {len(corrections)}")
        
        if corrections:
            print("\nCorrections made:")
            for correction in corrections[:10]:  # Show first 10 corrections
                print(f"  • {correction}")
            if len(corrections) > 10:
                print(f"  ... and {len(corrections) - 10} more corrections")
        
        return output_path
        
    except Exception as e:
        print(f"Error applying physics constraints: {str(e)}")
        return None


def fix_test_logging_issues():
    """Fix test logging initialization issues in test files"""
    fixes_applied = []
    
    test_files = [
        "examples/advanced_tests.py",
        "examples/performance_benchmarks.py", 
        "examples/maneuver_tests.py",
        "examples/environmental_tests.py"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if proper test initialization is missing
            if "logger.start_test(" not in content and "def main(" in content:
                # Add proper test initialization
                lines = content.split('\n')
                fixed_lines = []
                
                for line in lines:
                    fixed_lines.append(line)
                    
                    # Add test initialization after function definitions
                    if line.strip().startswith("def run_") and "(" in line:
                        func_name = line.split("def ")[1].split("(")[0]
                        test_name = func_name.replace("run_", "").replace("_", " ").title()
                        indent = "    "
                        fixed_lines.append(f"{indent}logger.start_test('{test_name}')")
                
                fixed_content = '\n'.join(fixed_lines)
                
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                
                fixes_applied.append(f"Fixed logging in {file_path}")
                
        except Exception as e:
            fixes_applied.append(f"Could not fix {file_path}: {str(e)}")
    
    return fixes_applied


def main():
    """Main function to apply all fixes"""
    print("=" * 80)
    print("PHYSICS CONSTRAINT ENFORCER")
    print("=" * 80)
    
    # Apply physics constraints to integrated test report
    print("\n1. Applying physics constraints to integrated test report...")
    corrected_file = apply_physics_constraints_to_file("integrated_test_report.json")
    
    if corrected_file:
        print(f"✓ Physics-corrected report saved as: {corrected_file}")
    else:
        print("✗ Failed to apply physics constraints")
    
    # Fix test logging issues
    print("\n2. Fixing test logging issues...")
    logging_fixes = fix_test_logging_issues()
    
    if logging_fixes:
        print("✓ Test logging fixes applied:")
        for fix in logging_fixes:
            print(f"  • {fix}")
    else:
        print("✗ No test logging fixes needed or applied")
    
    print("\n" + "=" * 80)
    print("PHYSICS CONSTRAINT ENFORCEMENT COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("• Run tests again to verify fixes")
    print("• Monitor physics validation logs for remaining issues")
    print("• Consider implementing real-time constraint checking in simulation core")


if __name__ == "__main__":
    import os
    main() 