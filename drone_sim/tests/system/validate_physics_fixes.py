#!/usr/bin/env python3
"""
Physics Fixes Validation

Tests that the physics anomaly fixes are working correctly
and that unrealistic values are now bounded appropriately.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent))

from drone_sim.physics.physics_validator import physics_validator
from drone_sim.core.simulator import Simulator, SimulationConfig
from examples.performance_benchmarks import calculate_realistic_efficiency, calculate_realistic_rtf
from examples.maneuver_tests import ManeuverTestSuite
from drone_sim.utils import TestLogger

class PhysicsFixValidator:
    """Validates that physics fixes are working correctly"""
    
    def __init__(self):
        self.logger = TestLogger("physics_fix_validation")
        self.test_results = {}
        self.validation_passed = True
    
    def run_validation_suite(self):
        """Run comprehensive validation of physics fixes"""
        print("🔬 VALIDATING PHYSICS FIXES")
        print("=" * 50)
        
        self.logger.start_test("Physics Fix Validation", {"scope": "all_fixes"})
        
        # Test 1: Real-time factor bounds
        self._test_rtf_bounds()
        
        # Test 2: Efficiency bounds
        self._test_efficiency_bounds()
        
        # Test 3: Formation flying realism
        self._test_formation_realism()
        
        # Test 4: Simulator physics validation
        self._test_simulator_validation()
        
        # Test 5: Figure-8 completion time
        self._test_figure_eight_realism()
        
        # Generate validation report
        self._generate_validation_report()
        
        # Finalize
        status = "PASSED" if self.validation_passed else "FAILED"
        self.logger.end_test(status, {
            "tests_run": len(self.test_results),
            "validations_passed": sum(1 for r in self.test_results.values() if r['passed']),
            "overall_result": status
        })
        
        log_dir = self.logger.finalize_session()
        print(f"\n📋 Validation logs saved to: {log_dir}")
        return self.validation_passed
    
    def _test_rtf_bounds(self):
        """Test that real-time factor calculations are now bounded"""
        print("\n🧪 Testing Real-Time Factor Bounds...")
        
        test_cases = [
            (0.001, 10.0),   # Should give very high RTF
            (0.1, 1.0),      # Normal case
            (1.0, 0.1),      # Should give low RTF
        ]
        
        passed = True
        results = []
        
        for actual_duration, sim_time in test_cases:
            rtf = calculate_realistic_rtf(actual_duration, sim_time)
            
            # Check bounds
            if rtf > 100.0:
                print(f"  ❌ RTF {rtf:.1f} exceeds maximum bound of 100x")
                passed = False
            elif rtf < 0.01:
                print(f"  ❌ RTF {rtf:.1f} below minimum bound of 0.01x")
                passed = False
            else:
                print(f"  ✅ RTF {rtf:.1f}x is within bounds")
            
            results.append({
                'actual_duration': actual_duration,
                'sim_time': sim_time,
                'rtf': rtf,
                'bounded': 0.01 <= rtf <= 100.0
            })
        
        self.test_results['rtf_bounds'] = {
            'passed': passed,
            'results': results
        }
        
        self.logger.log_step("rtf_bounds_test", {
            "test_cases": len(test_cases),
            "passed": passed,
            "results": results
        })
    
    def _test_efficiency_bounds(self):
        """Test that efficiency calculations are now bounded"""
        print("\n🧪 Testing Efficiency Bounds...")
        
        test_cases = [
            (0.001, 1.0),    # Should give very high efficiency
            (1.0, 1.0),      # 100% efficiency case
            (2.0, 1.0),      # Should give 50% efficiency
        ]
        
        passed = True
        results = []
        
        for actual_duration, expected_duration in test_cases:
            efficiency = calculate_realistic_efficiency(actual_duration, expected_duration)
            
            # Check bounds (efficiency should be 0-95%)
            if efficiency > 0.95:
                print(f"  ❌ Efficiency {efficiency:.1%} exceeds maximum bound of 95%")
                passed = False
            elif efficiency < 0.0:
                print(f"  ❌ Efficiency {efficiency:.1%} is negative")
                passed = False
            else:
                print(f"  ✅ Efficiency {efficiency:.1%} is within bounds")
            
            results.append({
                'actual_duration': actual_duration,
                'expected_duration': expected_duration,
                'efficiency': efficiency,
                'bounded': 0.0 <= efficiency <= 0.95
            })
        
        self.test_results['efficiency_bounds'] = {
            'passed': passed,
            'results': results
        }
        
        self.logger.log_step("efficiency_bounds_test", {
            "test_cases": len(test_cases),
            "passed": passed,
            "results": results
        })
    
    def _test_formation_realism(self):
        """Test that formation flying now has realistic precision"""
        print("\n🧪 Testing Formation Flying Realism...")
        
        # Run formation flying test multiple times to check consistency
        test_suite = ManeuverTestSuite()
        formation_results = []
        
        for i in range(5):  # Run 5 times
            result = test_suite.test_formation_flying()
            formation_results.append(result)
        
        # Check that formation accuracy is realistic (not near-zero)
        accuracies = [r['formation_accuracy'] for r in formation_results]
        avg_accuracy = np.mean(accuracies)
        
        # Formation accuracy should be > 1cm (realistic GPS/sensor limits)
        passed = avg_accuracy > 0.01
        
        if passed:
            print(f"  ✅ Formation accuracy {avg_accuracy:.3f}m is realistic")
        else:
            print(f"  ❌ Formation accuracy {avg_accuracy:.3f}m is unrealistically precise")
        
        # Check formation stability (should be < 1.0)
        stabilities = [r['formation_stability'] for r in formation_results]
        avg_stability = np.mean(stabilities)
        
        stability_realistic = avg_stability < 1.0
        if stability_realistic:
            print(f"  ✅ Formation stability {avg_stability:.3f} is realistic")
        else:
            print(f"  ❌ Formation stability {avg_stability:.3f} is unrealistically perfect")
        
        overall_passed = passed and stability_realistic
        
        self.test_results['formation_realism'] = {
            'passed': overall_passed,
            'avg_accuracy': avg_accuracy,
            'avg_stability': avg_stability,
            'results': formation_results
        }
        
        self.logger.log_step("formation_realism_test", {
            "test_runs": len(formation_results),
            "avg_accuracy": avg_accuracy,
            "avg_stability": avg_stability,
            "passed": overall_passed
        })
    
    def _test_simulator_validation(self):
        """Test that simulator now validates physics properly"""
        print("\n🧪 Testing Simulator Physics Validation...")
        
        # Test with unrealistic configuration
        unrealistic_config = SimulationConfig(
            dt=0.001,
            max_steps=1000,
            real_time_factor=1000.0,  # Unrealistic RTF
            physics_validation=True
        )
        
        # The config should be automatically corrected
        corrected_rtf = unrealistic_config.real_time_factor
        
        passed = corrected_rtf <= 100.0
        
        if passed:
            print(f"  ✅ Unrealistic RTF corrected from 1000x to {corrected_rtf}x")
        else:
            print(f"  ❌ RTF {corrected_rtf}x was not properly bounded")
        
        self.test_results['simulator_validation'] = {
            'passed': passed,
            'original_rtf': 1000.0,
            'corrected_rtf': corrected_rtf
        }
        
        self.logger.log_step("simulator_validation_test", {
            "original_rtf": 1000.0,
            "corrected_rtf": corrected_rtf,
            "passed": passed
        })
    
    def _test_figure_eight_realism(self):
        """Test that figure-8 completion time is now realistic"""
        print("\n🧪 Testing Figure-8 Completion Time...")
        
        # Run figure-8 test multiple times
        test_suite = ManeuverTestSuite()
        figure_eight_results = []
        
        for i in range(3):  # Run 3 times
            result = test_suite.test_figure_eight()
            figure_eight_results.append(result)
        
        # Check completion times
        completion_times = [r['completion_time'] for r in figure_eight_results]
        avg_completion_time = np.mean(completion_times)
        
        # Completion time should be reasonable (10-60 seconds for 5m radius figure-8)
        passed = 10.0 <= avg_completion_time <= 60.0
        
        if passed:
            print(f"  ✅ Figure-8 completion time {avg_completion_time:.1f}s is realistic")
        else:
            print(f"  ❌ Figure-8 completion time {avg_completion_time:.1f}s is unrealistic")
        
        self.test_results['figure_eight_realism'] = {
            'passed': passed,
            'avg_completion_time': avg_completion_time,
            'results': figure_eight_results
        }
        
        self.logger.log_step("figure_eight_realism_test", {
            "test_runs": len(figure_eight_results),
            "avg_completion_time": avg_completion_time,
            "passed": passed
        })
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        print(f"\n{'='*60}")
        print("🔬 PHYSICS FIXES VALIDATION REPORT")
        print(f"{'='*60}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        print(f"📊 Summary:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print(f"\n{'='*60}")
        
        # Update overall validation status
        self.validation_passed = passed_tests == total_tests
        
        # Save detailed report
        report_filename = "physics_fixes_validation_report.json"
        with open(report_filename, 'w') as f:
            import json
            # Convert test results to JSON-serializable format
            serializable_results = {}
            for test_name, result in self.test_results.items():
                serializable_results[test_name] = {
                    'passed': bool(result['passed']),
                    # Only include basic metrics, skip complex objects
                    'summary': str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                }
            
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': passed_tests/total_tests,
                    'overall_passed': bool(self.validation_passed)
                },
                'test_results': serializable_results
            }, f, indent=2)
        
        print(f"📄 Detailed validation report saved to: {report_filename}")

def main():
    """Main validation function"""
    validator = PhysicsFixValidator()
    
    # Run validation suite
    success = validator.run_validation_suite()
    
    if success:
        print("\n🎉 ALL PHYSICS FIXES VALIDATED SUCCESSFULLY!")
        print("The simulation now has realistic physical bounds and behavior.")
    else:
        print("\n⚠️  SOME PHYSICS FIXES NEED ATTENTION")
        print("Check the validation report for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 