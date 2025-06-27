#!/usr/bin/env python3
"""
Integrated Test Runner with Background Physics Validation

Demonstrates the complete integration of all test suites with background
physics validation running in separate threads for real-time anomaly detection.
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
import json

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from drone_sim.utils.background_validator import background_validator, AnomalyReport
from drone_sim.utils.test_logger import TestLogger
from drone_sim.tests.benchmarks.advanced_tests import AdvancedTestSuite
from drone_sim.tests.benchmarks.performance_benchmarks import PerformanceBenchmark
from drone_sim.tests.benchmarks.maneuver_tests import ManeuverTestSuite
from drone_sim.tests.benchmarks.environmental_tests import EnvironmentalTestSuite


class IntegratedTestRunner:
    """Integrated test runner with background physics validation"""
    
    def __init__(self):
        self.logger = TestLogger("integrated_test_runner")
        self.test_suites = {
            'advanced': AdvancedTestSuite(),
            'performance': PerformanceBenchmark(),
            'maneuvers': ManeuverTestSuite(),
            'environmental': EnvironmentalTestSuite()
        }
        
        # Validation tracking
        self.anomaly_count = 0
        self.critical_anomalies = []
        self.test_results = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def run_integrated_tests(self, suites_to_run: List[str] = None) -> Dict[str, Any]:
        """Run integrated tests with background validation"""
        print("🚀 INTEGRATED TEST RUNNER WITH BACKGROUND VALIDATION")
        print("=" * 60)
        
        # Start the integrated test session
        self.logger.start_test("Integrated Test Session", {
            "suites_available": list(self.test_suites.keys()),
            "suites_to_run": suites_to_run or list(self.test_suites.keys()),
            "background_validation": True
        })
        
        # Configure background validation
        validation_config = {
            'validation_interval': 0.05,  # More frequent validation (50ms)
            'anomaly_threshold': 2,  # Lower threshold for testing
            'auto_correction': True,
            'real_time_logging': True,
            'detailed_reports': True,
            'physics_context_capture': True
        }
        
        # Initialize background validator with custom config
        background_validator.config.update(validation_config)
        
        # Register anomaly callbacks
        background_validator.register_anomaly_callback(self._on_anomaly_detected)
        background_validator.register_correction_callback(self._on_correction_applied)
        
        # Start background validation
        print("\n🔬 Starting background physics validation...")
        validation_started = background_validator.start_background_validation()
        
        if not validation_started:
            print("❌ Failed to start background validation")
            return {}
        
        # Start real-time monitoring
        self._start_monitoring()
        
        try:
            # Run test suites
            suites_to_execute = suites_to_run or list(self.test_suites.keys())
            
            for suite_name in suites_to_execute:
                if suite_name in self.test_suites:
                    print(f"\n{'='*50}")
                    print(f"🧪 RUNNING {suite_name.upper()} TEST SUITE")
                    print(f"{'='*50}")
                    
                    self._run_suite_with_validation(suite_name)
                else:
                    print(f"⚠️  Unknown test suite: {suite_name}")
            
            # Wait a moment for final validations
            print("\n⏳ Finalizing background validation...")
            time.sleep(2.0)
            
        finally:
            # Stop monitoring and validation
            self._stop_monitoring()
            validation_summary = background_validator.stop_background_validation()
            
            # Generate comprehensive report
            final_report = self._generate_final_report(validation_summary)
            
            # Finalize logging
            self.logger.end_test("COMPLETED", {
                "test_results": self.test_results,
                "validation_summary": validation_summary,
                "anomaly_count": self.anomaly_count,
                "critical_anomalies": len(self.critical_anomalies)
            })
            
            log_dir = self.logger.finalize_session()
            print(f"\n📋 Complete test logs saved to: {log_dir}")
            
            return final_report
    
    def _run_suite_with_validation(self, suite_name: str):
        """Run a test suite with integrated background validation"""
        suite = self.test_suites[suite_name]
        suite_start_time = time.time()
        
        # Notify background validator of suite start
        background_validator.submit_test_event(
            f"{suite_name}_suite",
            "test_start",
            {"suite_name": suite_name, "start_time": suite_start_time},
            {"suite_type": "test_suite", "integration_mode": True}
        )
        
        try:
            # Run the test suite
            if suite_name == 'advanced':
                results = self._run_advanced_tests_with_validation(suite)
            elif suite_name == 'performance':
                results = self._run_performance_tests_with_validation(suite)
            elif suite_name == 'maneuvers':
                results = self._run_maneuver_tests_with_validation(suite)
            elif suite_name == 'environmental':
                results = self._run_environmental_tests_with_validation(suite)
            else:
                results = {}
            
            # Store results
            self.test_results[suite_name] = results
            
            # Calculate suite metrics
            suite_duration = time.time() - suite_start_time
            suite_success_rate = self._calculate_success_rate(results)
            
            print(f"\n✅ {suite_name.upper()} SUITE COMPLETED")
            print(f"   Duration: {suite_duration:.2f}s")
            print(f"   Success Rate: {suite_success_rate:.1%}")
            print(f"   Tests: {len(results)}")
            
            # Notify background validator of suite completion
            background_validator.submit_test_event(
                f"{suite_name}_suite",
                "test_end",
                {
                    "suite_name": suite_name,
                    "duration": suite_duration,
                    "success_rate": suite_success_rate,
                    "test_count": len(results),
                    "results": results
                },
                {"suite_type": "test_suite", "integration_mode": True}
            )
            
        except Exception as e:
            print(f"❌ Error in {suite_name} suite: {e}")
            self.logger.log_error(f"{suite_name}_suite_error", e)
            
            # Notify background validator of error
            background_validator.submit_test_event(
                f"{suite_name}_suite",
                "test_error",
                {"error": str(e), "suite_name": suite_name},
                {"suite_type": "test_suite", "integration_mode": True}
            )
    
    def _run_advanced_tests_with_validation(self, suite: AdvancedTestSuite) -> Dict[str, Any]:
        """Run advanced tests with step-by-step validation"""
        results = {}
        
        # Get test methods
        test_methods = [
            ('Performance Stress Test', suite.test_performance_stress),
            ('Multi-Configuration Test', suite.test_multi_configurations),
            ('Environmental Effects Test', suite.test_environmental_effects),
            ('Control System Comparison', suite.test_control_systems),
            ('Acoustic Analysis Test', suite.test_acoustic_analysis),
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n🔬 Running {test_name}...")
            
            # Submit test start event
            background_validator.submit_test_event(
                test_name, "test_start", {}, {"test_type": "advanced"}
            )
            
            try:
                # Run the test
                result = test_method()
                results[test_name] = result
                
                # Submit test data for validation
                background_validator.submit_test_event(
                    test_name, "test_step", result, {"test_type": "advanced"}
                )
                
                # Submit test completion
                background_validator.submit_test_event(
                    test_name, "test_end", result, {"test_type": "advanced"}
                )
                
                print(f"   ✅ {test_name} completed")
                
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
                results[test_name] = {"error": str(e), "status": "FAILED"}
                
                # Submit error event
                background_validator.submit_test_event(
                    test_name, "test_error", {"error": str(e)}, {"test_type": "advanced"}
                )
        
        return results
    
    def _run_performance_tests_with_validation(self, suite: PerformanceBenchmark) -> Dict[str, Any]:
        """Run performance tests with validation"""
        results = {}
        
        test_methods = [
            ('Time Step Scaling', suite.benchmark_time_steps),
            ('Propeller Count Scaling', suite.benchmark_propeller_scaling),
            ('Real-time Factor Performance', suite.benchmark_realtime_factors),
            ('Memory Usage Analysis', suite.benchmark_memory_usage),
            ('CPU Utilization Analysis', suite.benchmark_cpu_utilization),
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n⚡ Running {test_name}...")
            
            background_validator.submit_test_event(
                test_name, "test_start", {}, {"test_type": "performance"}
            )
            
            try:
                result = test_method()
                results[test_name] = result
                
                # Validate performance metrics
                background_validator.submit_test_event(
                    test_name, "test_step", result, {"test_type": "performance"}
                )
                
                background_validator.submit_test_event(
                    test_name, "test_end", result, {"test_type": "performance"}
                )
                
                print(f"   ✅ {test_name} completed")
                
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
                results[test_name] = {"error": str(e), "status": "FAILED"}
                
                background_validator.submit_test_event(
                    test_name, "test_error", {"error": str(e)}, {"test_type": "performance"}
                )
        
        return results
    
    def _run_maneuver_tests_with_validation(self, suite: ManeuverTestSuite) -> Dict[str, Any]:
        """Run maneuver tests with validation"""
        results = {}
        
        test_methods = [
            ('Precision Hover', suite.test_precision_hover),
            ('Figure-8 Pattern', suite.test_figure_eight),
            ('Spiral Climb', suite.test_spiral_climb),
            ('Aggressive Banking', suite.test_aggressive_banking),
            ('Formation Flying', suite.test_formation_flying),
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n🛸 Running {test_name}...")
            
            background_validator.submit_test_event(
                test_name, "test_start", {}, {"test_type": "maneuver"}
            )
            
            try:
                result = test_method()
                results[test_name] = result
                
                # Validate maneuver physics
                background_validator.submit_test_event(
                    test_name, "test_step", result, {"test_type": "maneuver"}
                )
                
                background_validator.submit_test_event(
                    test_name, "test_end", result, {"test_type": "maneuver"}
                )
                
                print(f"   ✅ {test_name} completed")
                
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
                results[test_name] = {"error": str(e), "status": "FAILED"}
                
                background_validator.submit_test_event(
                    test_name, "test_error", {"error": str(e)}, {"test_type": "maneuver"}
                )
        
        return results
    
    def _run_environmental_tests_with_validation(self, suite: EnvironmentalTestSuite) -> Dict[str, Any]:
        """Run environmental tests with validation"""
        results = {}
        
        test_methods = [
            ('Wind Resistance', suite.test_wind_conditions),
            ('Turbulence Handling', suite.test_turbulence_effects),
            ('Temperature Extremes', suite.test_temperature_effects),
            ('Altitude Performance', suite.test_altitude_effects),
            ('Weather Conditions', suite.test_weather_scenarios),
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n🌪️ Running {test_name}...")
            
            background_validator.submit_test_event(
                test_name, "test_start", {}, {"test_type": "environmental"}
            )
            
            try:
                result = test_method()
                results[test_name] = result
                
                # Validate environmental physics
                background_validator.submit_test_event(
                    test_name, "test_step", result, {"test_type": "environmental"}
                )
                
                background_validator.submit_test_event(
                    test_name, "test_end", result, {"test_type": "environmental"}
                )
                
                print(f"   ✅ {test_name} completed")
                
            except Exception as e:
                print(f"   ❌ {test_name} failed: {e}")
                results[test_name] = {"error": str(e), "status": "FAILED"}
                
                background_validator.submit_test_event(
                    test_name, "test_error", {"error": str(e)}, {"test_type": "environmental"}
                )
        
        return results
    
    def _start_monitoring(self):
        """Start real-time monitoring thread"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="TestMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        print("📊 Real-time monitoring started")
    
    def _stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        print("📊 Real-time monitoring stopped")
    
    def _monitoring_worker(self):
        """Real-time monitoring worker"""
        while self.monitoring_active:
            try:
                stats = background_validator.get_real_time_stats()
                
                # Print periodic status updates
                if stats['stats']['tests_monitored'] > 0:
                    print(f"\r🔬 Validation: {stats['stats']['tests_monitored']} tests, "
                          f"{stats['stats']['anomalies_detected']} anomalies, "
                          f"Queue: {stats['queue_size']}", end='', flush=True)
                
                time.sleep(2.0)  # Update every 2 seconds
                
            except Exception as e:
                print(f"\n⚠️  Monitoring error: {e}")
                break
    
    def _on_anomaly_detected(self, anomaly_report: AnomalyReport):
        """Callback for anomaly detection"""
        self.anomaly_count += 1
        
        if anomaly_report.severity in ["CRITICAL", "FATAL"]:
            self.critical_anomalies.append(anomaly_report)
            print(f"\n🚨 {anomaly_report.severity} ANOMALY in {anomaly_report.test_name}:")
            print(f"   {anomaly_report.description}")
            print(f"   Impact: {anomaly_report.impact_assessment}")
            
            # Log critical anomaly
            self.logger.log_step("critical_anomaly_detected", {
                "test_name": anomaly_report.test_name,
                "severity": anomaly_report.severity,
                "description": anomaly_report.description,
                "detected_values": anomaly_report.detected_values,
                "corrections": anomaly_report.suggested_corrections
            })
    
    def _on_correction_applied(self, test_name: str, corrections: Dict[str, Any]):
        """Callback for correction application"""
        print(f"\n🔧 Applied {len(corrections)} corrections to {test_name}")
        
        self.logger.log_step("corrections_applied", {
            "test_name": test_name,
            "correction_count": len(corrections),
            "corrections": corrections
        })
    
    def _calculate_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate success rate for test results"""
        if not results:
            return 0.0
        
        successful_tests = sum(1 for result in results.values() 
                             if isinstance(result, dict) and result.get('status') != 'FAILED')
        
        return successful_tests / len(results)
    
    def _generate_final_report(self, validation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        total_tests = sum(len(results) for results in self.test_results.values())
        total_success_rate = sum(self._calculate_success_rate(results) 
                               for results in self.test_results.values()) / len(self.test_results) if self.test_results else 0
        
        report = {
            'test_execution': {
                'suites_run': len(self.test_results),
                'total_tests': total_tests,
                'overall_success_rate': total_success_rate,
                'test_results': self.test_results
            },
            'physics_validation': validation_summary,
            'anomaly_analysis': {
                'total_anomalies': self.anomaly_count,
                'critical_anomalies': len(self.critical_anomalies),
                'anomaly_rate': self.anomaly_count / total_tests if total_tests > 0 else 0,
                'critical_anomaly_details': [
                    {
                        'test_name': a.test_name,
                        'severity': a.severity,
                        'description': a.description,
                        'impact': a.impact_assessment
                    } for a in self.critical_anomalies
                ]
            },
            'integration_metrics': {
                'background_validation_efficiency': validation_summary.get('validation_efficiency', 0),
                'real_time_monitoring': True,
                'automated_corrections': validation_summary.get('corrections_applied', 0),
                'validation_coverage': min(1.0, validation_summary.get('total_tests_monitored', 0) / total_tests) if total_tests > 0 else 0
            },
            'recommendations': self._generate_recommendations(validation_summary)
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 INTEGRATED TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Suites Run: {report['test_execution']['suites_run']}")
        print(f"Total Tests: {report['test_execution']['total_tests']}")
        print(f"Success Rate: {report['test_execution']['overall_success_rate']:.1%}")
        print(f"Anomalies Detected: {report['anomaly_analysis']['total_anomalies']}")
        print(f"Critical Anomalies: {report['anomaly_analysis']['critical_anomalies']}")
        print(f"Validation Coverage: {report['integration_metrics']['validation_coverage']:.1%}")
        print(f"Corrections Applied: {report['integration_metrics']['automated_corrections']}")
        
        return report
    
    def _generate_recommendations(self, validation_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check anomaly rate
        if validation_summary.get('total_anomalies', 0) > 10:
            recommendations.append("High anomaly count detected. Review physics constraints and test parameters.")
        
        # Check critical anomalies
        if len(self.critical_anomalies) > 0:
            recommendations.append("Critical anomalies detected. Investigate test configurations and physics models.")
        
        # Check validation efficiency
        if validation_summary.get('validation_efficiency', 0) < 0.5:
            recommendations.append("Low validation efficiency. Consider optimizing validation intervals or test complexity.")
        
        # Check most problematic tests
        problematic_tests = validation_summary.get('most_problematic_tests', [])
        if problematic_tests:
            test_names = [t['test_name'] for t in problematic_tests[:3]]
            recommendations.append(f"Focus on improving: {', '.join(test_names)}")
        
        if not recommendations:
            recommendations.append("All tests executed successfully with good physics validation coverage.")
        
        return recommendations


def main():
    """Main function for integrated test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Test Runner with Background Validation")
    parser.add_argument('--suites', nargs='+', 
                       choices=['advanced', 'performance', 'maneuvers', 'environmental'],
                       help='Test suites to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick subset of tests')
    
    args = parser.parse_args()
    
    # Create and run integrated tests
    runner = IntegratedTestRunner()
    
    try:
        suites_to_run = args.suites
        if args.quick:
            suites_to_run = ['maneuvers', 'performance']  # Quick subset
        
        final_report = runner.run_integrated_tests(suites_to_run)
        
        # Save final report
        report_file = Path("integrated_test_report.json")
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n📋 Final report saved to: {report_file}")
        
        # Return appropriate exit code
        critical_count = final_report['anomaly_analysis']['critical_anomalies']
        success_rate = final_report['test_execution']['overall_success_rate']
        
        if critical_count > 0 or success_rate < 0.8:
            print("\n⚠️  Tests completed with issues - review report for details")
            return 1
        else:
            print("\n🎉 All tests completed successfully!")
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 