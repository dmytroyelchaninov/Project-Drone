#!/usr/bin/env python3
"""
Verify Console Logging
Quick verification that console logging works with all modes
"""

import sys
import subprocess
import time
import signal
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from examples.realtime_simulation import ConsoleLogger, main
        from drone_sim.ui.realtime_interface import RealTimeInterface, SimulationParameters
        from drone_sim.physics.environment import Environment, EnvironmentConfig
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment_config():
    """Test that EnvironmentConfig works with correct parameters"""
    print("🧪 Testing EnvironmentConfig...")
    
    try:
        from drone_sim.physics.environment import EnvironmentConfig
        
        # Test with correct parameters
        config = EnvironmentConfig(
            gravity_magnitude=9.81,
            air_density=1.225
        )
        print(f"✅ EnvironmentConfig created: gravity={config.gravity_magnitude}, air_density={config.air_density}")
        return True
    except Exception as e:
        print(f"❌ EnvironmentConfig failed: {e}")
        return False

def test_simulation_parameters():
    """Test SimulationParameters creation"""
    print("🧪 Testing SimulationParameters...")
    
    try:
        from drone_sim.ui.realtime_interface import SimulationParameters
        
        params = SimulationParameters(
            mass=1.5,
            gravity=9.81,
            air_density=1.225,
            dt=0.002,
            real_time_factor=1.0
        )
        print(f"✅ SimulationParameters created: mass={params.mass}, gravity={params.gravity}")
        return True
    except Exception as e:
        print(f"❌ SimulationParameters failed: {e}")
        return False

def test_console_logger():
    """Test ConsoleLogger functionality"""
    print("🧪 Testing ConsoleLogger...")
    
    try:
        from examples.realtime_simulation import ConsoleLogger
        import numpy as np
        
        logger = ConsoleLogger(enabled=True)
        
        # Test event logging
        logger.log_event("TEST", "Testing console logger")
        
        # Test status logging
        position = np.array([1.0, 2.0, -3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        logger.log_status(position, velocity, "test", {"Test": "Value"})
        
        print("✅ ConsoleLogger working correctly")
        return True
    except Exception as e:
        print(f"❌ ConsoleLogger failed: {e}")
        return False

def test_interface_creation():
    """Test RealTimeInterface creation"""
    print("🧪 Testing RealTimeInterface creation...")
    
    try:
        from drone_sim.ui.realtime_interface import RealTimeInterface
        
        # This should not crash
        interface = RealTimeInterface()
        print("✅ RealTimeInterface created successfully")
        return True
    except Exception as e:
        print(f"❌ RealTimeInterface creation failed: {e}")
        return False

def run_verification():
    """Run all verification tests"""
    print("🚀 Console Logging Verification")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("EnvironmentConfig Test", test_environment_config),
        ("SimulationParameters Test", test_simulation_parameters),
        ("ConsoleLogger Test", test_console_logger),
        ("Interface Creation Test", test_interface_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All verification tests passed!")
        print("\n✅ Ready to use:")
        print("   python examples/realtime_simulation.py --mode manual --log")
        print("   python examples/realtime_simulation.py --mode ai --log")
        print("   python examples/realtime_simulation.py --mode hybrid --log --environment challenging")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1) 