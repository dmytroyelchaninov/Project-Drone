#!/usr/bin/env python3
"""
Test Logger System

Comprehensive logging system for drone simulation tests that generates
detailed, computer-readable logs for LLM analysis and issue detection.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import psutil
import os

class TestLogger:
    """Comprehensive test logging system for drone simulation tests"""
    
    def __init__(self, test_suite_name: str, log_dir: str = "logs"):
        """
        Initialize test logger
        
        Args:
            test_suite_name: Name of the test suite (e.g., "advanced_tests", "performance_benchmarks")
            log_dir: Directory to store log files
        """
        self.test_suite_name = test_suite_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamped log directory for this test run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_dir = self.log_dir / f"{test_suite_name}_{timestamp}"
        self.run_log_dir.mkdir(exist_ok=True)
        
        # Initialize logging components
        self.test_results = {}
        self.system_info = self._get_system_info()
        self.test_start_time = time.time()
        self.current_test = None
        self.test_logs = []
        
        # Setup file logging
        self._setup_file_logging()
        
        # Log test session start
        self._log_session_start()
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        log_file = self.run_log_dir / f"{self.test_suite_name}_detailed.log"
        
        # Create detailed logger
        self.logger = logging.getLogger(f"{self.test_suite_name}_logger")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_max = cpu_freq.max if cpu_freq else 0
        except (FileNotFoundError, AttributeError):
            # macOS compatibility
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency_max'], 
                                      capture_output=True, text=True)
                cpu_freq_max = int(result.stdout.strip()) / 1000000 if result.stdout.strip().isdigit() else 0
            except:
                cpu_freq_max = 0
        
        import sys
        import platform
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            },
            'python': {
                'version': sys.version,
                'version_info': list(sys.version_info),
                'executable': sys.executable,
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq_max': cpu_freq_max,
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3,
                'disk_usage_gb': psutil.disk_usage('/').total / 1024**3,
            },
            'environment': {
                'cwd': os.getcwd(),
                'path': os.environ.get('PATH', ''),
                'pythonpath': os.environ.get('PYTHONPATH', ''),
            }
        }
    
    def _log_session_start(self):
        """Log test session start information"""
        self.logger.info(f"=== TEST SESSION START: {self.test_suite_name} ===")
        self.logger.info(f"Session ID: {self.run_log_dir.name}")
        self.logger.info(f"System: {self.system_info['system']['platform']}")
        self.logger.info(f"Python: {self.system_info['python']['version_info']}")
        self.logger.info(f"CPU: {self.system_info['hardware']['cpu_count']} cores")
        self.logger.info(f"Memory: {self.system_info['hardware']['memory_total_gb']:.1f} GB")
        self.logger.info("=" * 60)
    
    def start_test(self, test_name: str, test_config: Optional[Dict] = None):
        """Start logging for a specific test"""
        self.current_test = {
            'name': test_name,
            'start_time': time.time(),
            'config': test_config or {},
            'logs': [],
            'metrics': {},
            'warnings': [],
            'errors': [],
            'system_state': self._capture_system_state(),
        }
        
        self.logger.info(f">>> STARTING TEST: {test_name}")
        if test_config:
            self.logger.debug(f"Test config: {json.dumps(test_config, indent=2, default=str)}")
    
    def log_step(self, step_name: str, data: Dict, level: str = "INFO"):
        """Log a test step with data"""
        if not self.current_test:
            self.logger.warning("No active test - call start_test() first")
            return
        
        timestamp = time.time()
        step_data = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.current_test['start_time'],
            'step': step_name,
            'data': self._serialize_data(data),
            'level': level,
            'system_state': self._capture_system_state() if level == "ERROR" else None
        }
        
        self.current_test['logs'].append(step_data)
        
        # Log to file
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"  {step_name}: {json.dumps(data, default=str)}")
    
    def log_metric(self, metric_name: str, value: Union[float, int, str], unit: str = ""):
        """Log a performance metric"""
        if not self.current_test:
            return
        
        metric_data = {
            'value': value,
            'unit': unit,
            'timestamp': time.time(),
            'relative_time': time.time() - self.current_test['start_time']
        }
        
        self.current_test['metrics'][metric_name] = metric_data
        self.logger.debug(f"  METRIC {metric_name}: {value} {unit}")
    
    def log_warning(self, message: str, details: Optional[Dict] = None):
        """Log a warning"""
        if not self.current_test:
            return
        
        warning_data = {
            'message': message,
            'details': details or {},
            'timestamp': time.time(),
            'relative_time': time.time() - self.current_test['start_time']
        }
        
        self.current_test['warnings'].append(warning_data)
        self.logger.warning(f"  WARNING: {message}")
        if details:
            self.logger.debug(f"  Warning details: {json.dumps(details, default=str)}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None, details: Optional[Dict] = None):
        """Log an error"""
        if not self.current_test:
            return
        
        error_data = {
            'message': message,
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None,
            'details': details or {},
            'timestamp': time.time(),
            'relative_time': time.time() - self.current_test['start_time'],
            'system_state': self._capture_system_state()
        }
        
        self.current_test['errors'].append(error_data)
        self.logger.error(f"  ERROR: {message}")
        if exception:
            self.logger.error(f"  Exception: {exception}")
            self.logger.debug(f"  Traceback: {traceback.format_exc()}")
    
    def end_test(self, status: str = "PASSED", result_data: Optional[Dict] = None):
        """End logging for current test"""
        if not self.current_test:
            self.logger.warning("No active test to end")
            return
        
        end_time = time.time()
        duration = end_time - self.current_test['start_time']
        
        # Finalize test data
        self.current_test.update({
            'end_time': end_time,
            'duration': duration,
            'status': status,
            'result_data': self._serialize_data(result_data or {}),
            'final_system_state': self._capture_system_state(),
        })
        
        # Store test results
        self.test_results[self.current_test['name']] = self.current_test.copy()
        
        # Log completion
        self.logger.info(f"<<< COMPLETED TEST: {self.current_test['name']}")
        self.logger.info(f"    Status: {status}")
        self.logger.info(f"    Duration: {duration:.3f}s")
        self.logger.info(f"    Warnings: {len(self.current_test['warnings'])}")
        self.logger.info(f"    Errors: {len(self.current_test['errors'])}")
        
        # Save individual test log
        self._save_test_log(self.current_test)
        
        # Clear current test
        self.current_test = None
    
    def _capture_system_state(self) -> Dict:
        """Capture current system state for debugging"""
        try:
            return {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'process_count': len(psutil.pids()),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage"""
        if isinstance(data, np.ndarray):
            return {
                '__type__': 'numpy_array',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'data': data.tolist()
            }
        elif isinstance(data, np.number):
            return float(data)
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif hasattr(data, '__dict__'):
            return {
                '__type__': type(data).__name__,
                '__dict__': self._serialize_data(data.__dict__)
            }
        else:
            return data
    
    def _save_test_log(self, test_data: Dict):
        """Save individual test log"""
        test_log_file = self.run_log_dir / f"test_{test_data['name'].replace(' ', '_').lower()}.json"
        
        with open(test_log_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
    
    def finalize_session(self):
        """Finalize test session and save all logs"""
        session_end_time = time.time()
        session_duration = session_end_time - self.test_start_time
        
        # Create session summary
        session_summary = {
            'session_info': {
                'test_suite': self.test_suite_name,
                'session_id': self.run_log_dir.name,
                'start_time': self.test_start_time,
                'end_time': session_end_time,
                'duration': session_duration,
            },
            'system_info': self.system_info,
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len([t for t in self.test_results.values() if t['status'] == 'PASSED']),
                'failed_tests': len([t for t in self.test_results.values() if t['status'] == 'FAILED']),
                'total_warnings': sum(len(t['warnings']) for t in self.test_results.values()),
                'total_errors': sum(len(t['errors']) for t in self.test_results.values()),
            },
            'test_results': self.test_results
        }
        
        # Save session summary
        session_file = self.run_log_dir / "session_summary.json"
        with open(session_file, 'w') as f:
            json.dump(session_summary, f, indent=2, default=str)
        
        # Save detailed results for LLM analysis
        analysis_file = self.run_log_dir / "llm_analysis_data.json"
        llm_data = self._prepare_llm_analysis_data()
        with open(analysis_file, 'w') as f:
            json.dump(llm_data, f, indent=2, default=str)
        
        # Create README for the log directory
        readme_file = self.run_log_dir / "README.md"
        self._create_log_readme(readme_file, session_summary)
        
        self.logger.info("=== TEST SESSION COMPLETE ===")
        self.logger.info(f"Session duration: {session_duration:.2f}s")
        self.logger.info(f"Tests: {session_summary['test_summary']['total_tests']} total, "
                        f"{session_summary['test_summary']['passed_tests']} passed, "
                        f"{session_summary['test_summary']['failed_tests']} failed")
        self.logger.info(f"Logs saved to: {self.run_log_dir}")
        
        return str(self.run_log_dir)
    
    def _prepare_llm_analysis_data(self) -> Dict:
        """Prepare structured data specifically for LLM analysis"""
        analysis_data = {
            'analysis_metadata': {
                'purpose': 'LLM analysis of drone simulation test results',
                'test_suite': self.test_suite_name,
                'timestamp': datetime.now().isoformat(),
                'data_structure_version': '1.0'
            },
            'issues_detected': [],
            'performance_anomalies': [],
            'error_patterns': [],
            'warning_patterns': [],
            'test_failures': [],
            'recommendations': []
        }
        
        # Analyze each test for issues
        for test_name, test_data in self.test_results.items():
            test_analysis = {
                'test_name': test_name,
                'status': test_data['status'],
                'duration': test_data['duration'],
                'issues': []
            }
            
            # Check for errors
            if test_data['errors']:
                for error in test_data['errors']:
                    issue = {
                        'type': 'ERROR',
                        'message': error['message'],
                        'timestamp': error['timestamp'],
                        'details': error.get('details', {}),
                        'exception': error.get('exception'),
                        'severity': 'HIGH'
                    }
                    test_analysis['issues'].append(issue)
                    analysis_data['issues_detected'].append(issue)
            
            # Check for warnings
            if test_data['warnings']:
                for warning in test_data['warnings']:
                    issue = {
                        'type': 'WARNING',
                        'message': warning['message'],
                        'timestamp': warning['timestamp'],
                        'details': warning.get('details', {}),
                        'severity': 'MEDIUM'
                    }
                    test_analysis['issues'].append(issue)
            
            # Check for performance anomalies
            if test_data['metrics']:
                for metric_name, metric_data in test_data['metrics'].items():
                    # Add performance analysis logic here
                    if isinstance(metric_data.get('value'), (int, float)):
                        value = metric_data['value']
                        # Example: detect unusually high values
                        if 'time' in metric_name.lower() and value > 10.0:
                            analysis_data['performance_anomalies'].append({
                                'test': test_name,
                                'metric': metric_name,
                                'value': value,
                                'threshold': 10.0,
                                'severity': 'MEDIUM'
                            })
            
            # Store test analysis
            analysis_data[f'test_analysis_{test_name.replace(" ", "_").lower()}'] = test_analysis
        
        return analysis_data
    
    def _create_log_readme(self, readme_file: Path, session_summary: Dict):
        """Create README file for log directory"""
        readme_content = f"""# Test Log Directory

## Session Information
- **Test Suite**: {session_summary['session_info']['test_suite']}
- **Session ID**: {session_summary['session_info']['session_id']}
- **Date**: {datetime.fromtimestamp(session_summary['session_info']['start_time']).strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {session_summary['session_info']['duration']:.2f} seconds

## System Information
- **Platform**: {session_summary['system_info']['system']['platform']}
- **Python**: {session_summary['system_info']['python']['version_info']}
- **CPU**: {session_summary['system_info']['hardware']['cpu_count']} cores
- **Memory**: {session_summary['system_info']['hardware']['memory_total_gb']:.1f} GB

## Test Results Summary
- **Total Tests**: {session_summary['test_summary']['total_tests']}
- **Passed**: {session_summary['test_summary']['passed_tests']}
- **Failed**: {session_summary['test_summary']['failed_tests']}
- **Warnings**: {session_summary['test_summary']['total_warnings']}
- **Errors**: {session_summary['test_summary']['total_errors']}

## Files in this Directory

### Core Files
- `session_summary.json` - Complete session data with all test results
- `llm_analysis_data.json` - Structured data for LLM analysis and issue detection
- `{session_summary['session_info']['test_suite']}_detailed.log` - Detailed text logs
- `README.md` - This file

### Individual Test Files
Each test has its own JSON file with detailed logs:
"""
        
        for test_name in session_summary['test_results'].keys():
            filename = f"test_{test_name.replace(' ', '_').lower()}.json"
            readme_content += f"- `{filename}` - Detailed logs for {test_name}\n"
        
        readme_content += f"""
## Usage for LLM Analysis

To analyze these logs with an LLM:

1. **Primary Analysis File**: `llm_analysis_data.json`
   - Contains structured issue detection data
   - Pre-processed for LLM consumption
   - Includes error patterns and performance anomalies

2. **Detailed Investigation**: Individual test JSON files
   - Contains step-by-step execution logs
   - System state snapshots at error points
   - Complete metric data

3. **Human Readable**: `{session_summary['session_info']['test_suite']}_detailed.log`
   - Chronological text log
   - Easy to read for debugging

## Data Structure

Each test log contains:
- Test configuration and parameters
- Step-by-step execution logs
- Performance metrics with timestamps
- Warning and error details
- System state at key points
- Final results and status

Generated by TestLogger v1.0
"""
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)

# Convenience functions for easy integration
def create_test_logger(test_suite_name: str) -> TestLogger:
    """Create a new test logger instance"""
    return TestLogger(test_suite_name)

def log_test_execution(logger: TestLogger, test_name: str, test_function, *args, **kwargs):
    """Execute a test function with comprehensive logging"""
    logger.start_test(test_name, {'args': args, 'kwargs': kwargs})
    
    try:
        result = test_function(*args, **kwargs)
        logger.end_test("PASSED", result)
        return result
    except Exception as e:
        logger.log_error(f"Test failed with exception: {e}", e)
        logger.end_test("FAILED")
        raise 