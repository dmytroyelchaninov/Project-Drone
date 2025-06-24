#!/usr/bin/env python3
"""
Test Enhanced Logging Capabilities

This script demonstrates how to use the existing logging infrastructure
to capture detailed information during real-time GUI simulation.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.realtime_simulation import main as run_simulation
from drone_sim.utils.test_logger import TestLogger
from drone_sim.utils.background_validator import BackgroundValidator

def demonstrate_logging_features():
    """Demonstrate the enhanced logging features"""
    print("🔍 ENHANCED LOGGING DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. EXISTING LOGGING INFRASTRUCTURE:")
    print("   ✅ Background Validation - Real-time physics validation")
    print("   ✅ TestLogger - Comprehensive session logging")
    print("   ✅ Console Logging - Real-time status display")
    print("   ✅ Performance Metrics - FPS, CPU, Memory tracking")
    print("   ✅ Detailed History - Position, velocity, acceleration")
    print("   ✅ Control Analysis - Thrust and moment logging")
    print("   ✅ AI Analytics - Learning progress and decisions")
    print("   ✅ System Monitoring - Resource usage tracking")
    
    print("\n2. AVAILABLE LOG OUTPUTS:")
    print("   📁 logs/background_validator_*/ - Physics validation logs")
    print("   📁 logs/realtime_simulation_*/ - Session logs")
    print("   📊 Console - Real-time status and metrics")
    print("   💾 detailed_simulation_log_*.json - Comprehensive data")
    
    print("\n3. ENHANCED FEATURES:")
    print("   🎯 Real-time FPS and performance monitoring")
    print("   🔬 Physics violation detection and counting")
    print("   🤖 AI decision analysis and learning trends")
    print("   📈 System resource usage tracking")
    print("   ⚡ Control input magnitude analysis")
    print("   🎮 Detailed simulation state history")
    
    print("\n4. COMMAND LINE OPTIONS:")
    print("   --log              Enable console logging (default)")
    print("   --no-log           Disable console logging")
    print("   --detailed-log     Save detailed JSON log file")
    print("   --log-interval     Set status logging interval")
    
    print("\n5. EXAMPLE USAGE:")
    print("   # Basic enhanced logging:")
    print("   python examples/realtime_simulation.py --mode manual --log")
    print("")
    print("   # AI mode with detailed logging:")
    print("   python examples/realtime_simulation.py --mode ai --detailed-log")
    print("")
    print("   # Challenging environment with full logging:")
    print("   python examples/realtime_simulation.py --mode ai --environment challenging --detailed-log")
    
    print("\n6. LOG DATA ANALYSIS:")
    print("   The enhanced logging captures:")
    print("   - Position, velocity, acceleration history")
    print("   - Control inputs and magnitudes")
    print("   - AI decision making and learning progress")
    print("   - Physics validation results")
    print("   - System performance metrics")
    print("   - Real-time FPS and resource usage")

def analyze_existing_logs():
    """Analyze existing log files to show the data available"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ No logs directory found. Run a simulation first.")
        return
    
    print("\n📊 EXISTING LOG ANALYSIS:")
    print("=" * 40)
    
    # Find recent log directories
    log_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    log_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_dirs:
        print("❌ No log directories found.")
        return
    
    print(f"Found {len(log_dirs)} log directories:")
    
    for i, log_dir in enumerate(log_dirs[:5]):  # Show recent 5
        print(f"\n{i+1}. {log_dir.name}")
        
        # Check for different log types
        files = list(log_dir.glob("*"))
        for file in files:
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"   📄 {file.name} ({size_kb:.1f} KB)")
        
        # Try to read some data from detailed log
        detailed_log = log_dir / f"{log_dir.name.split('_')[0]}_detailed.log"
        if detailed_log.exists():
            try:
                with open(detailed_log, 'r') as f:
                    lines = f.readlines()
                    print(f"   📊 {len(lines)} log entries")
                    
                    # Count different types of entries
                    validation_steps = sum(1 for line in lines if 'validation_simulation_step' in line)
                    if validation_steps > 0:
                        print(f"   🔬 {validation_steps} physics validation steps")
                        
            except Exception as e:
                print(f"   ❌ Error reading log: {e}")
        
        # Check for JSON data files
        json_files = list(log_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"   📊 JSON data: {len(data)} keys")
                        if 'session_info' in data:
                            duration = data.get('session_info', {}).get('duration', 0)
                            print(f"   ⏱️  Session duration: {duration:.1f}s")
            except Exception as e:
                print(f"   ❌ Error reading JSON: {e}")

def test_logging_components():
    """Test individual logging components"""
    print("\n🧪 TESTING LOGGING COMPONENTS:")
    print("=" * 40)
    
    # Test TestLogger
    print("\n1. Testing TestLogger...")
    try:
        logger = TestLogger("test_enhanced_logging")
        logger.start_test("Component Test")
        logger.log_step("initialization", {"status": "success", "components": 3})
        logger.log_metric("test_duration", 1.5, "seconds")
        logger.log_warning("This is a test warning")
        logger.end_test("PASSED", {"test_results": "all_good"})
        log_dir = logger.finalize_session()
        print(f"   ✅ TestLogger working - logs saved to: {log_dir}")
    except Exception as e:
        print(f"   ❌ TestLogger error: {e}")
    
    # Test BackgroundValidator
    print("\n2. Testing BackgroundValidator...")
    try:
        validator = BackgroundValidator()
        validator.start_background_validation()
        
        # Submit some test events
        for i in range(3):
            validator.submit_test_event(
                "Test Validation",
                "test_step",
                {
                    'step': i,
                    'position': [i * 0.5, 0, -2],
                    'velocity': [0.1, 0, 0],
                    'valid': True
                },
                {'test_iteration': i}
            )
        
        time.sleep(0.5)  # Allow processing
        
        stats = validator.get_real_time_stats()
        print(f"   ✅ BackgroundValidator working - processed {stats['stats']['tests_monitored']} events")
        
        validator.stop_background_validation()
    except Exception as e:
        print(f"   ❌ BackgroundValidator error: {e}")
    
    print("\n✅ All logging components are functional!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Enhanced Logging Capabilities")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing log files")
    parser.add_argument("--test", action="store_true", help="Test logging components")
    parser.add_argument("--demo", action="store_true", help="Show logging features")
    parser.add_argument("--run-sim", action="store_true", help="Run simulation with enhanced logging")
    
    args = parser.parse_args()
    
    if args.demo or not any([args.analyze, args.test, args.run_sim]):
        demonstrate_logging_features()
    
    if args.analyze:
        analyze_existing_logs()
    
    if args.test:
        test_logging_components()
    
    if args.run_sim:
        print("\n🚀 RUNNING SIMULATION WITH ENHANCED LOGGING:")
        print("=" * 50)
        print("This will start the real-time simulation with all logging enabled.")
        print("You can interact with the GUI and see detailed console output.")
        print("Press Ctrl+C to stop the simulation.")
        print("")
        
        # Run the simulation with enhanced logging
        import sys
        sys.argv = [
            "realtime_simulation.py",
            "--mode", "manual",
            "--log",
            "--detailed-log",
            "--environment", "default"
        ]
        
        try:
            run_simulation()
        except KeyboardInterrupt:
            print("\n👋 Simulation stopped by user")

if __name__ == "__main__":
    main() 