# Physics-Based Quadcopter Control System Guide

## Overview

The drone simulation now features a complete physics-based control system where all movement is controlled through individual engine thrust commands. This provides realistic flight dynamics and enables both human pilots and AI to learn proper drone control.

## 🚁 **Control Architecture**

### 1. **Individual Engine Control**

- **4 Independent Engines**: Each engine has individual thrust control (0-10 Newtons)
- **Engine Layout**: Standard quadcopter "+" configuration:
  ```
      Engine 1 (Front)
          [0.225m]
            ↑
  Engine 4 ← + → Engine 2 (Right)
    (Left)   ↓    [0.225m]
      Engine 3 (Back)
  ```

### 2. **Physics Integration**

- **Force Calculation**: Σ F = Σ(engine_thrusts) + gravity + drag
- **Moment Calculation**: Σ M = Σ(r × F) for each engine position
- **Motion Integration**: F = ma, τ = Iα (Newton's laws + Euler equations)

### 3. **Control Layers**

```
User Input (Arrow Keys) → Controller Logic → Engine Thrusts → Physics → Motion
AI Actions (Direct)     → Engine Mixing   → Force/Moments → Integration → State
```

## 🎮 **User Controls**

### **Arrow Key Navigation**

- **↑ Up Arrow**: Increase throttle (all engines +thrust) → **Climb**
- **↓ Down Arrow**: Decrease throttle (all engines -thrust) → **Descend**
- **← Left Arrow**: Roll left (right engines > left engines) → **Roll Left**
- **→ Right Arrow**: Roll right (left engines > right engines) → **Roll Right**

### **WASD Advanced Movement**

- **W**: Pitch forward (back engines > front engines) → **Move Forward**
- **S**: Pitch backward (front engines > back engines) → **Move Backward**
- **A**: Yaw left (counter-clockwise rotation) → **Rotate Left**
- **D**: Yaw right (clockwise rotation) → **Rotate Right**

### **Special Controls**

- **Space**: Return to hover (all neutral)
- **Esc**: Emergency stop (cut all thrust)

## ⚡ **Physics Details**

### **Engine Thrust Mapping**

```python
# Hover thrust per engine (to counteract gravity)
hover_thrust = mass * gravity / 4 = 1.5kg * 9.81m/s² / 4 = 3.675N per engine

# Control mixing for movement:
throttle_effect = control_input * base_thrust
roll_effect = roll_input * sensitivity * base_thrust
pitch_effect = pitch_input * sensitivity * base_thrust
yaw_effect = yaw_input * sensitivity * base_thrust * 0.5

# Final engine thrusts:
engine[0] = base + throttle + pitch - yaw    # Front
engine[1] = base + throttle + roll + yaw     # Right
engine[2] = base + throttle - pitch - yaw    # Back
engine[3] = base + throttle - roll + yaw     # Left
```

### **Force & Moment Calculation**

```python
# Total force (world frame)
total_force = Σ(engine_thrust) * body_to_world_rotation + gravity_vector

# Moments about center of mass (body frame)
total_moment = Σ(engine_position × engine_force)

# Motion integration
acceleration = total_force / mass
velocity += acceleration * dt
position += velocity * dt

angular_acceleration = inertia_matrix⁻¹ * (total_moment - ω × (I*ω))
angular_velocity += angular_acceleration * dt
orientation = integrate_quaternion(angular_velocity * dt)
```

### **Command Delay System**

- **Realistic Response**: 50ms delay between command and execution
- **Command Buffer**: Stores delayed commands for realistic control lag
- **Training Benefit**: AI learns to anticipate and compensate for delays

## 🤖 **AI Control Integration**

### **Direct Engine Control Mode**

AI can directly control individual engine thrusts:

```python
action = [engine1_thrust, engine2_thrust, engine3_thrust, engine4_thrust]
# Actions normalized to [-1,1], mapped to [0, 10N] thrust range
```

### **High-Level Control Mode**

AI can use same control inputs as human:

```python
action = [throttle, roll, pitch, yaw]
# Converted to engine thrusts through controller mixing
```

### **Physics-Based Rewards**

- **Positive**: Distance reduction, waypoint reaching, stability
- **Negative**: Crashes, excessive control effort, constraint violations
- **Realistic**: Based on actual physics simulation, not artificial penalties

## 🔬 **Key Benefits**

### **1. Realistic Training**

- AI learns actual drone physics, not simplified control
- Control limits and delays teach proper planning
- Force/moment constraints prevent unrealistic maneuvers

### **2. Transferable Skills**

- Human pilots learn real quadcopter control principles
- AI trained on real physics can transfer to actual drones
- Both learn to work within physical constraints

### **3. Predictive Control**

- Physics simulation enables accurate trajectory prediction
- Collision avoidance based on actual future positions
- Control planning considers real response times

### **4. Unified Interface**

- Same physics engine for manual and AI control
- Consistent behavior between training and operation
- Fair comparison between human and AI performance

## 📊 **Performance Characteristics**

### **Hover Requirements**

- **Total Thrust**: 14.715N (to counteract 1.5kg × 9.81m/s²)
- **Per Engine**: 3.675N hover thrust
- **Control Authority**: ±6.325N per engine (0-10N range)

### **Control Sensitivity**

- **Throttle**: 2% per frame (0.02/frame at 500Hz = 10/sec response)
- **Rotation**: 1% per frame (0.01/frame = 5/sec response)
- **Return to Neutral**: 95% decay when no input (smooth transitions)

### **Physics Limits**

- **Max Thrust**: 40N total (10N × 4 engines)
- **Max Climb Rate**: ~17 m/s² acceleration (40N - 14.715N gravity) / 1.5kg
- **Angular Rates**: Limited by moment arm and inertia

## 🛠 **Implementation Files**

### **Core Physics**

- `drone_sim/physics/quadcopter_physics.py`: Complete physics simulation
- `drone_sim/control/quadcopter_controller.py`: Engine thrust controller
- `drone_sim/control/keyboard_controller.py`: Human interface
- `drone_sim/control/rl_controller.py`: AI integration

### **Testing & Examples**

- `examples/test_physics_based_control.py`: Comprehensive system test
- `examples/realtime_simulation.py`: Full simulation with new physics

## 🚀 **Getting Started**

### **1. Test the System**

```bash
python examples/test_physics_based_control.py
```

### **2. Run Manual Control**

```bash
python examples/realtime_simulation.py --mode manual --log
```

### **3. Train AI with Physics**

```bash
python examples/realtime_simulation.py --mode ai --log --rtf 5.0
```

## 💡 **Pro Tips**

### **For Human Pilots**

- Start with gentle inputs - the physics responds realistically
- Use Space to return to hover when in trouble
- Practice smooth control transitions
- Remember: Release keys to auto-stabilize

### **For AI Training**

- Higher real-time factor (5-10x) for faster training
- Monitor engine thrust patterns in debug output
- Focus on learning efficient control strategies
- Use physics prediction for better collision avoidance

### **For Developers**

- All control goes through engine thrusts - no shortcuts
- Command delays simulate real hardware response
- Physics state includes forces, moments, and individual engine states
- Extensive debugging information available in controller outputs

The physics-based control system provides the foundation for realistic drone simulation where both humans and AI learn proper flight control through authentic quadcopter physics.
