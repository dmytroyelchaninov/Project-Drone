#!/usr/bin/env python3
"""
Test GUI Keyboard Event Capture
Simple test to verify tkinter is capturing keyboard events correctly
"""

import sys
import os
import tkinter as tk
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gui_keyboard():
    """Test GUI keyboard event capture"""
    print("🖥️ Testing GUI Keyboard Event Capture")
    print("=" * 50)
    print("⚡ Starting GUI test window...")
    print("🎮 Instructions:")
    print("   - A window will open")
    print("   - Press arrow keys and WASD")
    print("   - You should see key events in the console")
    print("   - Press ESC to close the window")
    print("=" * 50)
    
    # Create test window
    root = tk.Tk()
    root.title("Keyboard Test - Drone Simulation")
    root.geometry("400x300")
    
    # Create label for instructions
    label = tk.Label(root, text="🎮 Keyboard Test Window\n\nPress keys and watch the console!\n\nArrow Keys, WASD, Space, ESC", 
                     font=('Arial', 12), justify=tk.CENTER)
    label.pack(expand=True)
    
    # Track events
    events_captured = []
    
    def on_key_press(event):
        key_info = f"Key pressed: '{event.keysym}' (keycode: {event.keycode})"
        print(f"✅ {key_info}")
        events_captured.append(('press', event.keysym))
        
        # Handle special keys
        if event.keysym.lower() == 'escape':
            print("🛑 ESC pressed - closing window...")
            root.quit()
    
    def on_key_release(event):
        key_info = f"Key released: '{event.keysym}' (keycode: {event.keycode})"
        print(f"🔓 {key_info}")
        events_captured.append(('release', event.keysym))
    
    # Bind keyboard events - multiple ways for maximum compatibility
    root.bind('<KeyPress>', on_key_press)
    root.bind('<KeyRelease>', on_key_release)
    root.bind_all('<KeyPress>', on_key_press)
    root.bind_all('<KeyRelease>', on_key_release)
    
    # Ensure focus
    root.focus_set()
    root.focus_force()
    
    # Make sure window gets focus
    root.attributes('-topmost', True)
    root.after_idle(lambda: root.attributes('-topmost', False))
    
    print("🖥️ Window opened. Try pressing keys now...")
    
    try:
        # Run the GUI loop
        root.mainloop()
    except KeyboardInterrupt:
        print("🛑 Test interrupted by user")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS")
    print("=" * 50)
    
    if events_captured:
        print(f"✅ Captured {len(events_captured)} keyboard events")
        
        # Check for specific keys
        keys_tested = set()
        for event_type, key in events_captured:
            if event_type == 'press':
                keys_tested.add(key.lower())
        
        important_keys = ['up', 'down', 'left', 'right', 'w', 's', 'a', 'd', 'space']
        tested_important = [key for key in important_keys if key in keys_tested]
        
        print(f"🎮 Important keys tested: {', '.join(tested_important)}")
        
        if len(tested_important) >= 3:
            print("✅ GUI keyboard capture working correctly!")
            return True
        else:
            print("⚠️  Few keys tested - GUI might have focus issues")
            return False
    else:
        print("❌ No keyboard events captured!")
        print("🔍 Possible issues:")
        print("   - Window didn't get focus")
        print("   - Keyboard bindings not working")
        print("   - GUI system problems")
        return False

def main():
    """Run the GUI keyboard test"""
    print("🧪 GUI Keyboard Event Test")
    print("🎯 This will test if the GUI can capture keyboard events")
    print()
    
    try:
        success = test_gui_keyboard()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 GUI keyboard test PASSED!")
            print("💡 The GUI can capture keyboard events correctly.")
            print("   If manual control still doesn't work, the issue is")
            print("   in how the events are processed by the controller.")
        else:
            print("❌ GUI keyboard test FAILED!")
            print("💡 The GUI cannot capture keyboard events properly.")
            print("   This explains why manual control doesn't work.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 