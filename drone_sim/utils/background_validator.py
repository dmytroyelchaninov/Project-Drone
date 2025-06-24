#!/usr/bin/env python3
"""
Background Physics Validation System

Runs physics validation in background threads during test execution,
providing real-time anomaly detection and detailed logging.
"""

import threading
import time
import queue
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .test_logger import TestLogger
from ..physics.physics_validator import physics_validator, ValidationResult


@dataclass
class ValidationEvent:
    """Event for background validation"""
    timestamp: float
    test_name: str
    test_data: Dict[str, Any]
    event_type: str  # 'test_start', 'test_step', 'test_end', 'anomaly_detected'
    metadata: Dict[str, Any]


@dataclass
class AnomalyReport:
    """Detailed anomaly report"""
    timestamp: float
    test_name: str
    anomaly_type: str
    severity: str  # 'WARNING', 'CRITICAL', 'FATAL'
    description: str
    detected_values: Dict[str, Any]
    suggested_corrections: Dict[str, Any]
    physics_context: Dict[str, Any]
    impact_assessment: str


class BackgroundValidator:
    """Background physics validation system"""
    
    def __init__(self, validation_config: Optional[Dict] = None):
        self.config = validation_config or self._default_config()
        self.is_running = False
        self.validation_thread = None
        self.event_queue = queue.Queue()
        self.anomaly_reports = []
        self.validation_stats = {
            'tests_monitored': 0,
            'anomalies_detected': 0,
            'critical_violations': 0,
            'corrections_applied': 0,
            'validation_start_time': None
        }
        
        # Setup logging
        self.logger = TestLogger("background_validator")
        self.validation_logger = logging.getLogger("background_validator")
        
        # Callbacks for real-time notifications
        self.anomaly_callbacks = []
        self.correction_callbacks = []
        
        # Physics monitoring state
        self.current_test_context = {}
        self.validation_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for background validation"""
        return {
            'validation_interval': 0.1,  # Check every 100ms
            'anomaly_threshold': 3,  # Report after 3 consecutive anomalies
            'auto_correction': True,  # Apply corrections automatically
            'real_time_logging': True,  # Log anomalies in real-time
            'detailed_reports': True,  # Generate detailed anomaly reports
            'physics_context_capture': True,  # Capture physics state context
            'performance_monitoring': True,  # Monitor validation performance
            'max_queue_size': 1000,  # Maximum events in queue
            'validation_timeout': 30.0  # Timeout for validation operations
        }
    
    def start_background_validation(self) -> bool:
        """Start background validation thread"""
        if self.is_running:
            self.validation_logger.warning("Background validation already running")
            return False
        
        self.is_running = True
        self.validation_stats['validation_start_time'] = time.time()
        
        # Start validation thread
        self.validation_thread = threading.Thread(
            target=self._validation_worker,
            name="PhysicsValidator",
            daemon=True
        )
        self.validation_thread.start()
        
        self.validation_logger.info("🔬 Background physics validation started")
        self.logger.start_test("Background Validation", {
            "config": self.config,
            "thread_id": self.validation_thread.ident
        })
        
        return True
    
    def stop_background_validation(self) -> Dict[str, Any]:
        """Stop background validation and return summary"""
        if not self.is_running:
            return {}
        
        self.is_running = False
        
        # Signal shutdown
        self.event_queue.put(ValidationEvent(
            timestamp=time.time(),
            test_name="SHUTDOWN",
            test_data={},
            event_type="shutdown",
            metadata={}
        ))
        
        # Wait for thread to finish
        if self.validation_thread and self.validation_thread.is_alive():
            self.validation_thread.join(timeout=5.0)
        
        # Generate final report
        summary = self._generate_validation_summary()
        
        self.logger.end_test("COMPLETED", {
            "validation_stats": self.validation_stats,
            "anomalies_detected": len(self.anomaly_reports),
            "summary": summary
        })
        
        log_dir = self.logger.finalize_session()
        self.validation_logger.info(f"🔬 Background validation stopped. Logs: {log_dir}")
        
        return summary
    
    def submit_test_event(self, test_name: str, event_type: str, 
                         test_data: Dict[str, Any], metadata: Optional[Dict] = None):
        """Submit test event for background validation"""
        if not self.is_running:
            return
        
        event = ValidationEvent(
            timestamp=time.time(),
            test_name=test_name,
            test_data=test_data.copy() if test_data else {},
            event_type=event_type,
            metadata=metadata or {}
        )
        
        try:
            self.event_queue.put(event, timeout=1.0)
        except queue.Full:
            self.validation_logger.warning(f"Validation queue full, dropping event: {test_name}")
    
    def register_anomaly_callback(self, callback: Callable[[AnomalyReport], None]):
        """Register callback for anomaly notifications"""
        self.anomaly_callbacks.append(callback)
    
    def register_correction_callback(self, callback: Callable[[str, Dict], None]):
        """Register callback for correction notifications"""
        self.correction_callbacks.append(callback)
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time validation statistics"""
        return {
            'is_running': self.is_running,
            'queue_size': self.event_queue.qsize(),
            'stats': self.validation_stats.copy(),
            'recent_anomalies': self.anomaly_reports[-5:] if self.anomaly_reports else [],
            'current_test': self.current_test_context.get('test_name', 'None')
        }
    
    def _validation_worker(self):
        """Main validation worker thread"""
        self.validation_logger.info("🧵 Validation worker thread started")
        
        consecutive_anomalies = {}  # Track consecutive anomalies per test
        
        while self.is_running:
            try:
                # Get next event (with timeout to allow periodic checks)
                event = self.event_queue.get(timeout=self.config['validation_interval'])
                
                if event.event_type == "shutdown":
                    break
                
                # Process validation event
                self._process_validation_event(event, consecutive_anomalies)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except queue.Empty:
                # Timeout - perform periodic validation checks
                self._periodic_validation_check()
            except Exception as e:
                self.validation_logger.error(f"Validation worker error: {e}")
                self.logger.log_error("validation_worker_error", e)
        
        self.validation_logger.info("🧵 Validation worker thread stopped")
    
    def _process_validation_event(self, event: ValidationEvent, consecutive_anomalies: Dict):
        """Process a single validation event"""
        self.validation_stats['tests_monitored'] += 1
        
        # Update current test context
        if event.event_type in ['test_start', 'test_step']:
            self.current_test_context = {
                'test_name': event.test_name,
                'timestamp': event.timestamp,
                'metadata': event.metadata
            }
        
        # Validate test data
        validation_result = physics_validator.validate_test_results(
            event.test_name, event.test_data
        )
        
        # Process validation results
        if validation_result.violations or validation_result.warnings:
            self._handle_validation_issues(event, validation_result, consecutive_anomalies)
        else:
            # Reset consecutive anomaly counter for this test
            consecutive_anomalies.pop(event.test_name, None)
        
        # Log validation step
        if self.config['real_time_logging']:
            self.logger.log_step(f"validation_{event.event_type}", {
                "test_name": event.test_name,
                "violations": len(validation_result.violations),
                "warnings": len(validation_result.warnings),
                "corrections": len(validation_result.corrections),
                "is_valid": validation_result.is_valid
            })
        
        # Store validation history
        self.validation_history.append({
            'timestamp': event.timestamp,
            'test_name': event.test_name,
            'event_type': event.event_type,
            'validation_result': asdict(validation_result)
        })
    
    def _handle_validation_issues(self, event: ValidationEvent, 
                                validation_result: ValidationResult, 
                                consecutive_anomalies: Dict):
        """Handle validation issues and anomalies"""
        test_name = event.test_name
        
        # Track consecutive anomalies
        consecutive_anomalies[test_name] = consecutive_anomalies.get(test_name, 0) + 1
        
        # Determine severity
        critical_violations = [v for v in validation_result.violations if "CRITICAL" in v]
        fatal_violations = [v for v in validation_result.violations if "FATAL" in v]
        
        if fatal_violations:
            severity = "FATAL"
        elif critical_violations:
            severity = "CRITICAL"
        else:
            severity = "WARNING"
        
        # Update statistics
        if severity == "CRITICAL":
            self.validation_stats['critical_violations'] += 1
        self.validation_stats['anomalies_detected'] += 1
        
        # Create anomaly report
        anomaly_report = self._create_anomaly_report(
            event, validation_result, severity, consecutive_anomalies[test_name]
        )
        
        # Store anomaly report
        self.anomaly_reports.append(anomaly_report)
        
        # Apply corrections if enabled
        if self.config['auto_correction'] and validation_result.corrections:
            self._apply_corrections(event, validation_result.corrections)
        
        # Notify callbacks
        self._notify_anomaly_callbacks(anomaly_report)
        
        # Log anomaly
        if severity in ["CRITICAL", "FATAL"]:
            self.validation_logger.error(f"🚨 {severity} ANOMALY in {test_name}: {anomaly_report.description}")
        else:
            self.validation_logger.warning(f"⚠️  {severity} in {test_name}: {anomaly_report.description}")
        
        # Check if we need to escalate (multiple consecutive anomalies)
        if consecutive_anomalies[test_name] >= self.config['anomaly_threshold']:
            self._escalate_anomaly(test_name, anomaly_report)
    
    def _create_anomaly_report(self, event: ValidationEvent, 
                             validation_result: ValidationResult,
                             severity: str, consecutive_count: int) -> AnomalyReport:
        """Create detailed anomaly report"""
        
        # Extract detected values that caused violations
        detected_values = {}
        for violation in validation_result.violations:
            # Parse violation to extract parameter and value
            if "=" in violation:
                parts = violation.split("=")
                if len(parts) >= 2:
                    param = parts[0].split(":")[-1].strip()
                    value_part = parts[1].split()[0]
                    try:
                        detected_values[param] = float(value_part)
                    except ValueError:
                        detected_values[param] = value_part
        
        # Assess impact
        impact_assessment = self._assess_anomaly_impact(
            validation_result, severity, consecutive_count
        )
        
        # Capture physics context
        physics_context = {}
        if self.config['physics_context_capture']:
            physics_context = self._capture_physics_context(event)
        
        return AnomalyReport(
            timestamp=event.timestamp,
            test_name=event.test_name,
            anomaly_type=f"{len(validation_result.violations)}V_{len(validation_result.warnings)}W",
            severity=severity,
            description=f"{len(validation_result.violations)} violations, {len(validation_result.warnings)} warnings",
            detected_values=detected_values,
            suggested_corrections=validation_result.corrections,
            physics_context=physics_context,
            impact_assessment=impact_assessment
        )
    
    def _assess_anomaly_impact(self, validation_result: ValidationResult, 
                             severity: str, consecutive_count: int) -> str:
        """Assess the impact of detected anomaly"""
        if severity == "FATAL":
            return f"FATAL: Simulation integrity compromised. Immediate intervention required."
        elif severity == "CRITICAL" and consecutive_count >= 3:
            return f"HIGH: {consecutive_count} consecutive critical violations detected. Test reliability at risk."
        elif severity == "CRITICAL":
            return f"MEDIUM: Critical physics violation. Monitor for recurrence."
        else:
            return f"LOW: Minor physics inconsistency. Monitoring recommended."
    
    def _capture_physics_context(self, event: ValidationEvent) -> Dict[str, Any]:
        """Capture current physics context for anomaly analysis"""
        context = {
            'event_type': event.event_type,
            'test_metadata': event.metadata,
            'timestamp': event.timestamp,
            'queue_size': self.event_queue.qsize(),
            'validation_history_size': len(self.validation_history)
        }
        
        # Add recent validation trends
        if len(self.validation_history) >= 5:
            recent_violations = [
                len(h['validation_result']['violations']) 
                for h in self.validation_history[-5:]
            ]
            context['recent_violation_trend'] = {
                'avg_violations': np.mean(recent_violations),
                'max_violations': max(recent_violations),
                'trend_direction': 'increasing' if recent_violations[-1] > recent_violations[0] else 'decreasing'
            }
        
        return context
    
    def _apply_corrections(self, event: ValidationEvent, corrections: Dict[str, Any]):
        """Apply physics corrections"""
        self.validation_stats['corrections_applied'] += len(corrections)
        
        self.validation_logger.info(f"🔧 Applying {len(corrections)} corrections to {event.test_name}")
        
        # Notify correction callbacks
        for callback in self.correction_callbacks:
            try:
                callback(event.test_name, corrections)
            except Exception as e:
                self.validation_logger.error(f"Correction callback error: {e}")
    
    def _escalate_anomaly(self, test_name: str, anomaly_report: AnomalyReport):
        """Escalate anomaly for immediate attention"""
        self.validation_logger.critical(
            f"🚨 ESCALATED ANOMALY: {test_name} has {self.config['anomaly_threshold']} "
            f"consecutive anomalies. Severity: {anomaly_report.severity}"
        )
        
        # Log detailed escalation report
        self.logger.log_step("anomaly_escalation", {
            "test_name": test_name,
            "severity": anomaly_report.severity,
            "consecutive_count": self.config['anomaly_threshold'],
            "impact": anomaly_report.impact_assessment,
            "suggested_corrections": anomaly_report.suggested_corrections
        })
    
    def _notify_anomaly_callbacks(self, anomaly_report: AnomalyReport):
        """Notify registered anomaly callbacks"""
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly_report)
            except Exception as e:
                self.validation_logger.error(f"Anomaly callback error: {e}")
    
    def _periodic_validation_check(self):
        """Perform periodic validation checks"""
        # Check validation thread health
        if not self.is_running:
            return
        
        # Monitor queue size
        queue_size = self.event_queue.qsize()
        if queue_size > self.config['max_queue_size'] * 0.8:
            self.validation_logger.warning(f"Validation queue nearly full: {queue_size}")
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        duration = time.time() - self.validation_stats['validation_start_time']
        
        # Categorize anomalies by severity
        anomaly_by_severity = {}
        for report in self.anomaly_reports:
            severity = report.severity
            anomaly_by_severity[severity] = anomaly_by_severity.get(severity, 0) + 1
        
        # Calculate validation efficiency
        validation_rate = self.validation_stats['tests_monitored'] / duration if duration > 0 else 0
        
        # Generate test-specific statistics
        test_stats = {}
        for report in self.anomaly_reports:
            test_name = report.test_name
            if test_name not in test_stats:
                test_stats[test_name] = {'anomalies': 0, 'severities': []}
            test_stats[test_name]['anomalies'] += 1
            test_stats[test_name]['severities'].append(report.severity)
        
        return {
            'duration': duration,
            'validation_rate': validation_rate,
            'total_tests_monitored': self.validation_stats['tests_monitored'],
            'total_anomalies': len(self.anomaly_reports),
            'anomalies_by_severity': anomaly_by_severity,
            'corrections_applied': self.validation_stats['corrections_applied'],
            'test_statistics': test_stats,
            'validation_efficiency': min(1.0, validation_rate / 100.0),  # Normalize to 0-1
            'most_problematic_tests': self._get_most_problematic_tests(test_stats)
        }
    
    def _get_most_problematic_tests(self, test_stats: Dict) -> List[Dict]:
        """Get tests with most anomalies"""
        problematic = []
        for test_name, stats in test_stats.items():
            critical_count = stats['severities'].count('CRITICAL')
            fatal_count = stats['severities'].count('FATAL')
            
            problematic.append({
                'test_name': test_name,
                'total_anomalies': stats['anomalies'],
                'critical_anomalies': critical_count,
                'fatal_anomalies': fatal_count,
                'risk_score': fatal_count * 10 + critical_count * 3 + stats['anomalies']
            })
        
        # Sort by risk score
        problematic.sort(key=lambda x: x['risk_score'], reverse=True)
        return problematic[:5]  # Top 5 most problematic


# Global background validator instance
background_validator = BackgroundValidator() 