#!/usr/bin/env python3
"""
Background Validation Demonstration

Simple demonstration of how the background physics validation system works
alongside test execution, showing real-time anomaly detection and correction.
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, Any

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent))

from drone_sim.utils import background_validator, AnomalyReport
from drone_sim.core.simulator import Simulator, SimulationConfig
from drone_sim.physics.rigid_body import RigidBody, RigidBodyConfig
from drone_sim.control.pid_controller import PIDController


class BackgroundValidationDemo:
    """Demonstration of background validation capabilities"""
    
    def __init__(self):
        self.anomaly_count = 0
        self.correction_count = 0
        
    def run_demo(self):
        """Run the background validation demonstration"""
        print("🔬 BACKGROUND PHYSICS VALIDATION DEMONSTRATION")
        print("=" * 55)
        
        # Register callbacks to monitor validation events
        background_validator.register_anomaly_callback(self._on_anomaly)
        background_validator.register_correction_callback(self._on_correction)
        
        # Start background validation
        print("\n🚀 Starting background validation...")
        if not background_validator.start_background_validation():
            print("❌ Failed to start background validation")
            return
        
        try:
            # Run demonstration scenarios
            self._demo_normal_operation()
            self._demo_anomaly_detection()
            self._demo_correction_system()
            self._demo_escalation()
            
        finally:
            # Stop background validation and get summary
            print("\n⏹️  Stopping background validation...")
            summary = background_validator.stop_background_validation()
            
            # Print final results
            self._print_summary(summary)
    
    def _demo_normal_operation(self):
        """Demonstrate normal operation with valid physics"""
        print("\n📊 DEMO 1: Normal Operation")
        print("-" * 30)
        
        # Submit normal test data
        normal_data = {
            'real_time_factor': 1.0,
            'efficiency': 0.85,
            'completion_time': 15.2,
            'max_velocity': 12.5,
            'power_consumption': 150.0
        }
        
        print("Submitting normal test data...")
        background_validator.submit_test_event(
            "Normal Operation Test",
            "test_step",
            normal_data,
            {"demo": "normal_operation"}
        )
        
        time.sleep(0.5)  # Give validation time to process
        print("✅ Normal data processed without issues")
    
    def _demo_anomaly_detection(self):
        """Demonstrate anomaly detection with invalid physics"""
        print("\n🚨 DEMO 2: Anomaly Detection")
        print("-" * 30)
        
        # Submit data with physics violations
        anomalous_data = {
            'real_time_factor': 500.0,  # Unrealistic RTF
            'efficiency': 1.5,  # > 100% efficiency (impossible)
            'completion_time': 0.01,  # Unrealistically fast
            'max_velocity': 200.0,  # Extremely high velocity
            'power_consumption': -50.0  # Negative power (impossible)
        }
        
        print("Submitting anomalous test data...")
        background_validator.submit_test_event(
            "Anomaly Detection Test",
            "test_step",
            anomalous_data,
            {"demo": "anomaly_detection"}
        )
        
        time.sleep(1.0)  # Give validation time to process and log
        print("🔍 Anomalous data submitted - check for violations above")
    
    def _demo_correction_system(self):
        """Demonstrate automatic correction system"""
        print("\n🔧 DEMO 3: Automatic Corrections")
        print("-" * 30)
        
        # Submit data that can be corrected
        correctable_data = {
            'real_time_factor': 150.0,  # Too high, can be clamped to 100
            'efficiency': 0.98,  # Slightly too high, can be clamped to 0.95
            'completion_time': 3600.1,  # Slightly over 1 hour, can be clamped
            'max_velocity': 105.0,  # Slightly too high, can be corrected
        }
        
        print("Submitting correctable test data...")
        background_validator.submit_test_event(
            "Correction Demo Test",
            "test_step",
            correctable_data,
            {"demo": "correction_system"}
        )
        
        time.sleep(1.0)
        print("🔧 Correctable data submitted - corrections should be applied")
    
    def _demo_escalation(self):
        """Demonstrate anomaly escalation with consecutive violations"""
        print("\n⚠️  DEMO 4: Anomaly Escalation")
        print("-" * 30)
        
        # Submit multiple consecutive anomalies to trigger escalation
        for i in range(3):
            escalation_data = {
                'real_time_factor': 1000.0 + i * 100,  # Increasingly bad
                'efficiency': 2.0 + i * 0.5,  # Impossible efficiency
                'completion_time': 0.001,  # Impossibly fast
                'test_iteration': i + 1
            }
            
            print(f"Submitting escalation test #{i+1}...")
            background_validator.submit_test_event(
                "Escalation Demo Test",
                "test_step",
                escalation_data,
                {"demo": "escalation", "iteration": i + 1}
            )
            
            time.sleep(0.3)  # Short delay between submissions
        
        time.sleep(1.0)  # Allow processing
        print("⚡ Multiple consecutive anomalies submitted - escalation should occur")
    
    def _demo_real_simulation(self):
        """Demonstrate with actual simulation data"""
        print("\n🛸 DEMO 5: Real Simulation Integration")
        print("-" * 30)
        
        try:
            # Create a simple simulation
            config = SimulationConfig(
                dt=0.01,
                max_steps=100,
                real_time_factor=2.0,
                physics_validation=True
            )
            
            sim = Simulator(config)
            
            # Create rigid body
            rb_config = RigidBodyConfig(mass=1.5, inertia_matrix=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.2]])
            rigid_body = RigidBody(rb_config)
            sim.set_rigid_body(rigid_body)
            
            # Create controller
            controller = PIDController()
            sim.set_controller(controller)
            
            print("Running simulation with background validation...")
            
            # Submit simulation start event
            background_validator.submit_test_event(
                "Real Simulation Demo",
                "test_start",
                {"simulation_config": config.__dict__},
                {"demo": "real_simulation"}
            )
            
            # Run simulation
            results = sim.run_simulation()
            
            # Submit simulation results
            background_validator.submit_test_event(
                "Real Simulation Demo",
                "test_end",
                results,
                {"demo": "real_simulation"}
            )
            
            print(f"✅ Simulation completed: {results['steps_completed']} steps")
            
        except Exception as e:
            print(f"❌ Simulation demo failed: {e}")
            
            # Submit error event
            background_validator.submit_test_event(
                "Real Simulation Demo",
                "test_error",
                {"error": str(e)},
                {"demo": "real_simulation"}
            )
    
    def _on_anomaly(self, anomaly_report: AnomalyReport):
        """Callback for anomaly detection"""
        self.anomaly_count += 1
        
        print(f"\n🚨 ANOMALY #{self.anomaly_count} DETECTED:")
        print(f"   Test: {anomaly_report.test_name}")
        print(f"   Severity: {anomaly_report.severity}")
        print(f"   Description: {anomaly_report.description}")
        print(f"   Impact: {anomaly_report.impact_assessment}")
        
        if anomaly_report.detected_values:
            print(f"   Detected Values: {anomaly_report.detected_values}")
        
        if anomaly_report.suggested_corrections:
            print(f"   Suggested Corrections: {anomaly_report.suggested_corrections}")
    
    def _on_correction(self, test_name: str, corrections: Dict[str, Any]):
        """Callback for correction application"""
        self.correction_count += 1
        
        print(f"\n🔧 CORRECTION #{self.correction_count} APPLIED:")
        print(f"   Test: {test_name}")
        print(f"   Corrections: {corrections}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print final validation summary"""
        print("\n" + "=" * 55)
        print("📊 BACKGROUND VALIDATION DEMO SUMMARY")
        print("=" * 55)
        
        print(f"Duration: {summary.get('duration', 0):.2f} seconds")
        print(f"Tests Monitored: {summary.get('total_tests_monitored', 0)}")
        print(f"Total Anomalies: {summary.get('total_anomalies', 0)}")
        print(f"Corrections Applied: {summary.get('corrections_applied', 0)}")
        print(f"Validation Rate: {summary.get('validation_rate', 0):.1f} tests/sec")
        print(f"Validation Efficiency: {summary.get('validation_efficiency', 0):.1%}")
        
        # Print anomaly breakdown
        anomalies_by_severity = summary.get('anomalies_by_severity', {})
        if anomalies_by_severity:
            print("\nAnomalies by Severity:")
            for severity, count in anomalies_by_severity.items():
                print(f"   {severity}: {count}")
        
        # Print most problematic tests
        problematic_tests = summary.get('most_problematic_tests', [])
        if problematic_tests:
            print("\nMost Problematic Tests:")
            for test in problematic_tests[:3]:
                print(f"   {test['test_name']}: {test['total_anomalies']} anomalies "
                      f"(Risk Score: {test['risk_score']})")
        
        print("\n🎯 Demo completed successfully!")
        print("   - Normal operation: ✅ No anomalies expected")
        print("   - Anomaly detection: 🚨 Multiple violations detected")
        print("   - Correction system: 🔧 Automatic fixes applied")
        print("   - Escalation: ⚠️  Consecutive anomalies escalated")


def main():
    """Main demonstration function"""
    demo = BackgroundValidationDemo()
    
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        background_validator.stop_background_validation()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        background_validator.stop_background_validation()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 