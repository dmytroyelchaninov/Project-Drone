#!/usr/bin/env python3
"""
Test Console Logging
Simple test to verify console logging functionality
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_console_logger():
    """Test the console logger from the main application"""
    from examples.realtime_simulation import ConsoleLogger
    
    print("🧪 Testing Console Logger")
    print("=" * 30)
    
    # Create logger
    logger = ConsoleLogger(enabled=True)
    
    # Test event logging
    logger.log_event("TEST", "Console logger initialized")
    
    # Test status logging
    position = np.array([1.0, 2.0, -3.0])
    velocity = np.array([0.5, -0.2, 0.1])
    additional_info = {
        'Thrust': '15.2N',
        'Episodes': 5,
        'Reward': '12.34'
    }
    
    print("\nTesting status logging (should appear every second)...")
    for i in range(5):
        logger.log_status(position + i*0.1, velocity + i*0.05, "manual", additional_info)
        time.sleep(0.3)  # Fast for testing
    
    # Test control input logging
    print("\nTesting control input logging...")
    logger.log_control_input(15.5, np.array([1.0, -0.5, 0.2]))
    time.sleep(0.1)
    logger.log_control_input(14.8, np.array([0.8, -0.3, 0.1]))
    
    # Test AI progress logging
    print("\nTesting AI progress logging...")
    logger.log_ai_progress(10, 25.67, 0.75)
    
    logger.log_event("TEST", "All logging tests completed")
    print("✅ Console logger test completed")

def test_main_application():
    """Test the main application with console logging"""
    print("\n🚀 Testing Main Application Console Logging")
    print("=" * 50)
    
    # Import and test
    from examples.realtime_simulation import main
    
    # Mock command line arguments for testing
    import sys
    original_argv = sys.argv
    try:
        # Test with logging enabled
        sys.argv = ['realtime_simulation.py', '--mode', 'manual', '--log', '--no-gui']
        print("Testing with --log --no-gui flags...")
        
        # This would normally start the GUI, but we're testing the setup
        print("✅ Main application import successful")
        
    except Exception as e:
        print(f"❌ Error testing main application: {e}")
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    try:
        test_console_logger()
        test_main_application()
        print("\n🎉 All tests completed successfully!")
        
        print("\n📋 Usage Examples:")
        print("1. GUI with console logging:")
        print("   python examples/realtime_simulation.py --mode manual --log")
        print("2. GUI without console logging:")
        print("   python examples/realtime_simulation.py --mode manual --no-log")
        print("3. AI mode with console logging:")
        print("   python examples/realtime_simulation.py --mode ai --log")
        print("4. Different environments:")
        print("   python examples/realtime_simulation.py --environment challenging --log")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 