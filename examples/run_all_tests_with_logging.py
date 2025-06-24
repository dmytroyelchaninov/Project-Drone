#!/usr/bin/env python3
"""
Run All Tests with Comprehensive Logging

This script runs all test suites with comprehensive logging enabled,
generating detailed logs for both human analysis and LLM consumption.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import sys

def run_test_suite(script_name: str, suite_name: str):
    """Run a test suite and capture its output"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING {suite_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, f"examples/{script_name}"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {suite_name} completed successfully in {duration:.2f}s")
            return {
                'status': 'SUCCESS',
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ {suite_name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return {
                'status': 'FAILED',
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {suite_name} timed out after 5 minutes")
        return {
            'status': 'TIMEOUT',
            'duration': 300,
            'error': 'Test suite timed out'
        }
    except Exception as e:
        print(f"💥 {suite_name} crashed: {e}")
        return {
            'status': 'CRASHED',
            'duration': time.time() - start_time,
            'error': str(e)
        }

def collect_log_directories():
    """Collect all log directories created during testing"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    log_dirs = []
    for item in logs_dir.iterdir():
        if item.is_dir():
            log_dirs.append({
                'name': item.name,
                'path': str(item),
                'created_time': item.stat().st_ctime,
                'size_mb': sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / 1024 / 1024
            })
    
    # Sort by creation time (newest first)
    log_dirs.sort(key=lambda x: x['created_time'], reverse=True)
    return log_dirs

def analyze_log_directories(log_dirs):
    """Analyze log directories for issues and patterns"""
    analysis = {
        'total_log_dirs': len(log_dirs),
        'total_size_mb': sum(d['size_mb'] for d in log_dirs),
        'test_suites': [],
        'issues_found': [],
        'performance_metrics': {}
    }
    
    for log_dir in log_dirs:
        log_path = Path(log_dir['path'])
        
        # Read LLM analysis data if available
        llm_data_file = log_path / "llm_analysis_data.json"
        if llm_data_file.exists():
            try:
                with open(llm_data_file, 'r') as f:
                    llm_data = json.load(f)
                
                suite_analysis = {
                    'name': llm_data.get('analysis_metadata', {}).get('test_suite', 'unknown'),
                    'timestamp': llm_data.get('analysis_metadata', {}).get('timestamp', ''),
                    'issues_detected': len(llm_data.get('issues_detected', [])),
                    'performance_anomalies': len(llm_data.get('performance_anomalies', [])),
                    'test_failures': len(llm_data.get('test_failures', [])),
                    'log_dir': log_dir['name']
                }
                
                analysis['test_suites'].append(suite_analysis)
                
                # Collect issues
                for issue in llm_data.get('issues_detected', []):
                    analysis['issues_found'].append({
                        'suite': suite_analysis['name'],
                        'type': issue.get('type', 'unknown'),
                        'message': issue.get('message', ''),
                        'severity': issue.get('severity', 'unknown')
                    })
                    
            except Exception as e:
                print(f"Warning: Could not analyze {llm_data_file}: {e}")
        
        # Read session summary if available
        session_file = log_path / "session_summary.json"
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                suite_name = session_data.get('session_info', {}).get('test_suite', 'unknown')
                test_summary = session_data.get('test_summary', {})
                
                analysis['performance_metrics'][suite_name] = {
                    'total_tests': test_summary.get('total_tests', 0),
                    'passed_tests': test_summary.get('passed_tests', 0),
                    'failed_tests': test_summary.get('failed_tests', 0),
                    'total_warnings': test_summary.get('total_warnings', 0),
                    'total_errors': test_summary.get('total_errors', 0),
                    'duration': session_data.get('session_info', {}).get('duration', 0)
                }
                
            except Exception as e:
                print(f"Warning: Could not read session summary {session_file}: {e}")
    
    return analysis

def generate_master_report(test_results, log_analysis):
    """Generate a master test report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"logs/master_test_report_{timestamp}.json"
    
    master_report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'comprehensive_test_analysis',
            'version': '1.0'
        },
        'execution_summary': {
            'test_suites_run': len(test_results),
            'successful_suites': len([r for r in test_results.values() if r['status'] == 'SUCCESS']),
            'failed_suites': len([r for r in test_results.values() if r['status'] != 'SUCCESS']),
            'total_execution_time': sum(r['duration'] for r in test_results.values())
        },
        'test_suite_results': test_results,
        'log_analysis': log_analysis,
        'recommendations': []
    }
    
    # Generate recommendations based on analysis
    if log_analysis['issues_found']:
        master_report['recommendations'].append({
            'type': 'ISSUES_DETECTED',
            'priority': 'HIGH',
            'message': f"Found {len(log_analysis['issues_found'])} issues across test suites",
            'details': log_analysis['issues_found']
        })
    
    # Check for performance issues
    for suite, metrics in log_analysis['performance_metrics'].items():
        if metrics['failed_tests'] > 0:
            master_report['recommendations'].append({
                'type': 'TEST_FAILURES',
                'priority': 'HIGH',
                'message': f"{suite} has {metrics['failed_tests']} failed tests",
                'suite': suite
            })
        
        if metrics['total_errors'] > 0:
            master_report['recommendations'].append({
                'type': 'ERRORS_DETECTED',
                'priority': 'MEDIUM',
                'message': f"{suite} has {metrics['total_errors']} errors",
                'suite': suite
            })
    
    # Save master report
    Path("logs").mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(master_report, f, indent=2, default=str)
    
    return report_file, master_report

def print_summary_report(master_report):
    """Print a human-readable summary of the master report"""
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    exec_summary = master_report['execution_summary']
    print(f"🏃 Test Suites Executed: {exec_summary['test_suites_run']}")
    print(f"✅ Successful: {exec_summary['successful_suites']}")
    print(f"❌ Failed: {exec_summary['failed_suites']}")
    print(f"⏱️  Total Time: {exec_summary['total_execution_time']:.2f}s")
    
    print(f"\n📋 Test Suite Details:")
    for suite_name, result in master_report['test_suite_results'].items():
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"   {status_emoji} {suite_name}: {result['status']} ({result['duration']:.2f}s)")
    
    log_analysis = master_report['log_analysis']
    print(f"\n📁 Log Analysis:")
    print(f"   📂 Log Directories: {log_analysis['total_log_dirs']}")
    print(f"   💾 Total Log Size: {log_analysis['total_size_mb']:.2f} MB")
    print(f"   ⚠️  Issues Found: {len(log_analysis['issues_found'])}")
    
    if log_analysis['performance_metrics']:
        print(f"\n🎯 Performance Metrics:")
        for suite, metrics in log_analysis['performance_metrics'].items():
            print(f"   {suite}:")
            print(f"      Tests: {metrics['passed_tests']}/{metrics['total_tests']} passed")
            if metrics['total_warnings'] > 0:
                print(f"      Warnings: {metrics['total_warnings']}")
            if metrics['total_errors'] > 0:
                print(f"      Errors: {metrics['total_errors']}")
    
    if master_report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in master_report['recommendations']:
            priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡"
            print(f"   {priority_emoji} {rec['type']}: {rec['message']}")

def main():
    """Main execution function"""
    print("🚁 COMPREHENSIVE DRONE SIMULATION TEST SUITE")
    print("=" * 60)
    print("Running all test suites with comprehensive logging...")
    
    # Define test suites to run
    test_suites = [
        ("advanced_tests.py", "Advanced Tests"),
        ("performance_benchmarks.py", "Performance Benchmarks"),
        ("maneuver_tests.py", "Maneuver Tests"),
        ("environmental_tests.py", "Environmental Tests")
    ]
    
    test_results = {}
    
    # Run each test suite
    for script_name, suite_name in test_suites:
        result = run_test_suite(script_name, suite_name)
        test_results[suite_name] = result
        
        # Brief pause between test suites
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("📊 ANALYZING GENERATED LOGS")
    print(f"{'='*60}")
    
    # Collect and analyze logs
    log_dirs = collect_log_directories()
    log_analysis = analyze_log_directories(log_dirs)
    
    print(f"Found {len(log_dirs)} log directories")
    print(f"Total log size: {log_analysis['total_size_mb']:.2f} MB")
    
    # Generate master report
    report_file, master_report = generate_master_report(test_results, log_analysis)
    
    # Print summary
    print_summary_report(master_report)
    
    print(f"\n📄 Master report saved to: {report_file}")
    
    # Print log directory information
    print(f"\n📁 Generated Log Directories:")
    for log_dir in log_dirs:
        print(f"   📂 {log_dir['name']} ({log_dir['size_mb']:.2f} MB)")
        print(f"      Path: {log_dir['path']}")
    
    print(f"\n🎯 TESTING COMPLETE!")
    print("=" * 60)
    print("All test suites have been executed with comprehensive logging.")
    print("Check the logs/ directory for detailed analysis data.")
    print("\nFor LLM analysis, use the following files:")
    print("• logs/*/llm_analysis_data.json - Structured data for AI analysis")
    print("• logs/*/session_summary.json - Complete test session data")
    print("• logs/*/test_*.json - Individual test detailed logs")
    print(f"• {report_file} - Master analysis report")

if __name__ == "__main__":
    main() 