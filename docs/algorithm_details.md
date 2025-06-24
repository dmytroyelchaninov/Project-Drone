# Algorithm Details and Mathematical Foundations

## 1. Rigid Body Dynamics

### 1.1 State Vector Representation

The drone state is represented as a 13-DOF vector:

```
x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz, ωx, ωy, ωz]ᵀ
```

Where:

- **Position** `[px, py, pz]`: Inertial frame coordinates (North-East-Down)
- **Quaternion** `[qw, qx, qy, qz]`: Attitude representation (scalar-first convention)
- **Velocity** `[vx, vy, vz]`: Body frame linear velocity
- **Angular Velocity** `[ωx, ωy, ωz]`: Body frame angular velocity

### 1.2 Equations of Motion

#### Linear Dynamics (Newton's Second Law)

```
F = m(a + ω × v)
```

In body frame:

```
F_body = m(v̇_body + ω_body × v_body)
```

Where:

- `F_body`: Total forces in body frame
- `m`: Vehicle mass
- `v̇_body`: Linear acceleration in body frame
- `ω_body`: Angular velocity vector
- `v_body`: Linear velocity in body frame

#### Angular Dynamics (Euler's Equation)

```
M = Iω̇ + ω × (Iω)
```

Where:

- `M`: Total moments about center of mass
- `I`: Inertia matrix (3×3)
- `ω̇`: Angular acceleration
- `ω`: Angular velocity vector

#### Kinematic Equations

**Position Kinematics**:

```
ṗ = R(q) · v_body
```

Where `R(q)` is the rotation matrix from quaternion:

```
R(q) = [1-2(qy²+qz²)   2(qxqy-qwqz)   2(qxqz+qwqy)  ]
       [2(qxqy+qwqz)   1-2(qx²+qz²)   2(qyqz-qwqx)  ]
       [2(qxqz-qwqy)   2(qyqz+qwqx)   1-2(qx²+qy²)  ]
```

**Quaternion Kinematics**:

```
q̇ = ½ q ⊗ [0, ωx, ωy, ωz]ᵀ
```

Expanded form:

```
q̇w = -½(qx·ωx + qy·ωy + qz·ωz)
q̇x = ½(qw·ωx + qy·ωz - qz·ωy)
q̇y = ½(qw·ωy + qz·ωx - qx·ωz)
q̇z = ½(qw·ωz + qx·ωy - qy·ωx)
```

### 1.3 State Derivative Computation

The complete state derivative vector:

```
ẋ = [ṗx, ṗy, ṗz, q̇w, q̇x, q̇y, q̇z, v̇x, v̇y, v̇z, ω̇x, ω̇y, ω̇z]ᵀ
```

**Implementation Algorithm**:

```python
def compute_derivatives(state, forces, moments):
    # Extract state components
    pos = state[0:3]
    quat = state[3:7]
    vel = state[7:10]
    omega = state[10:13]

    # Normalize quaternion
    quat = quat / norm(quat)

    # Position derivatives (velocity in inertial frame)
    R = quaternion_to_rotation_matrix(quat)
    pos_dot = R @ vel

    # Quaternion derivatives
    omega_quat = [0, omega[0], omega[1], omega[2]]
    quat_dot = 0.5 * quaternion_multiply(quat, omega_quat)

    # Linear acceleration (body frame)
    gravity_body = R.T @ [0, 0, 9.81]
    vel_dot = forces/mass + gravity_body - cross(omega, vel)

    # Angular acceleration
    omega_dot = inv(I) @ (moments - cross(omega, I @ omega))

    return concatenate([pos_dot, quat_dot, vel_dot, omega_dot])
```

---

## 2. Numerical Integration (RK4)

### 2.1 Runge-Kutta 4th Order Method

For the differential equation `ẋ = f(x, t)`, the RK4 update is:

```
k₁ = f(xₙ, tₙ)
k₂ = f(xₙ + ½h·k₁, tₙ + ½h)
k₃ = f(xₙ + ½h·k₂, tₙ + ½h)
k₄ = f(xₙ + h·k₃, tₙ + h)

xₙ₊₁ = xₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

Where:

- `h`: Time step (typically 2ms)
- `xₙ`: Current state
- `f(x,t)`: State derivative function

### 2.2 Quaternion Normalization

After each integration step, quaternions must be normalized:

```python
def normalize_quaternion(q):
    norm = sqrt(qw² + qx² + qy² + qz²)
    return q / norm
```

**Why Normalization is Critical**:

- Quaternions represent rotations only when normalized
- Numerical integration introduces small errors
- Unnormalized quaternions cause attitude drift

### 2.3 Integration Stability

**Time Step Selection**:

- Too large: Numerical instability
- Too small: Computational overhead
- Optimal: 2ms for typical drone dynamics

**Stability Criterion**:

```
h < 2/ωₘₐₓ
```

Where `ωₘₐₓ` is the highest frequency in the system.

---

## 3. Propeller Aerodynamics

### 3.1 Momentum Theory

**Basic Thrust Equation**:

```
T = CT · ρ · n² · D⁴
```

Where:

- `T`: Thrust force
- `CT`: Thrust coefficient (function of advance ratio)
- `ρ`: Air density
- `n`: Rotational speed (rev/s)
- `D`: Propeller diameter

**Power and Torque**:

```
P = CP · ρ · n³ · D⁵
Q = P / (2πn) = CP · ρ · n² · D⁵ / (2π)
```

### 3.2 Advance Ratio

The advance ratio determines propeller efficiency:

```
J = V / (n · D)
```

Where `V` is the advance velocity.

**Coefficient Models**:

```python
def thrust_coefficient(J):
    # Polynomial fit to experimental data
    return c0 + c1*J + c2*J² + c3*J³

def power_coefficient(J):
    # Similar polynomial model
    return d0 + d1*J + d2*J² + d3*J³
```

### 3.3 Propeller Array Dynamics

For a quadcopter with propellers at positions `rᵢ`:

**Total Force**:

```
F_total = Σᵢ Fᵢ
```

**Total Moment**:

```
M_total = Σᵢ (rᵢ × Fᵢ + Qᵢ)
```

Where:

- `Fᵢ`: Thrust force from propeller i
- `Qᵢ`: Reaction torque from propeller i
- `rᵢ`: Position vector to propeller i

---

## 4. Control System Algorithms

### 4.1 Three-Layer PID Architecture

#### Position Controller (Outer Loop)

```
eₚ = r_pos - x_pos
u_pos = Kₚ·eₚ + Kᵢ·∫eₚdt + Kd·ėₚ
```

**Output**: Desired attitude angles

```
φ_des = (u_pos_x · sin(ψ) - u_pos_y · cos(ψ)) / g
θ_des = (u_pos_x · cos(ψ) + u_pos_y · sin(ψ)) / g
ψ_des = r_yaw  # Direct yaw command
```

#### Attitude Controller (Middle Loop)

**Quaternion Error**:

```
q_error = q_desired ⊗ q_current⁻¹
```

**PID Update**:

```
e_att = 2 · sign(q_error_w) · [q_error_x, q_error_y, q_error_z]
u_att = Kₚ·e_att + Kᵢ·∫e_att dt + Kd·ė_att
```

#### Rate Controller (Inner Loop)

```
e_rate = ω_desired - ω_current
u_rate = Kₚ·e_rate + Kᵢ·∫e_rate dt + Kd·ė_rate
```

### 4.2 Motor Mixing

Convert control outputs to individual motor commands:

**Quadcopter X-Configuration**:

```
[m₁]   [1   1   1   1] [T_total]
[m₂] = [1  -1  -1   1] [τ_roll ]
[m₃]   [1   1  -1  -1] [τ_pitch]
[m₄]   [1  -1   1  -1] [τ_yaw  ]
```

Where:

- `mᵢ`: Motor i command
- `T_total`: Total thrust
- `τ`: Torque commands

### 4.3 Anti-Windup Protection

**Integral Clamping**:

```python
if abs(output) > limit:
    if sign(error) == sign(integral):
        integral += 0  # Don't accumulate
    else:
        integral += error * dt
else:
    integral += error * dt
```

**Back-Calculation Method**:

```python
output_saturated = clamp(output, -limit, limit)
integral -= (output - output_saturated) / Ki
```

---

## 5. Acoustic Modeling

### 5.1 Ffowcs Williams-Hawkings Equation

The fundamental equation for aeroacoustic noise:

```
□²p'(x,t) = ∂/∂t[ρ₀vₙδ(f)] - ∂/∂xᵢ[lᵢδ(f)] + ∂²/∂xᵢ∂xⱼ[Tᵢⱼ H(f)]
```

Where:

- `p'`: Acoustic pressure
- `f = 0`: Surface equation (propeller blade)
- `vₙ`: Normal velocity
- `lᵢ`: Surface force per unit area
- `Tᵢⱼ`: Lighthill stress tensor
- `H(f)`: Heaviside function

### 5.2 Thickness Noise

From blade volume displacement:

```
p₁(x,t) = (ρ₀/4π) ∂²/∂t² ∫∫ [Vₙ]/|x - y| dS
```

**Discrete Implementation**:

```python
def thickness_noise(blade_elements, observer_pos, time):
    pressure = 0
    for element in blade_elements:
        r_vec = observer_pos - element.position
        r_mag = norm(r_vec)

        # Retarded time
        tau = time - r_mag / c_sound

        # Volume displacement rate
        V_dot = element.thickness * element.normal_velocity(tau)

        # Add contribution
        pressure += rho_0 * V_dot / (4 * pi * r_mag)

    return pressure
```

### 5.3 Loading Noise

From aerodynamic forces:

```
p₂(x,t) = (1/4π) ∂/∂t ∫∫ [Fᵢ · r̂ᵢ]/c|x - y| dS
```

**Implementation**:

```python
def loading_noise(blade_elements, observer_pos, time):
    pressure = 0
    for element in blade_elements:
        r_vec = observer_pos - element.position
        r_mag = norm(r_vec)
        r_hat = r_vec / r_mag

        # Retarded time
        tau = time - r_mag / c_sound

        # Force projection
        F_radial = dot(element.force(tau), r_hat)

        # Add contribution
        pressure += F_radial / (4 * pi * c_sound * r_mag)

    return pressure
```

### 5.4 Broadband Noise

Empirical model for turbulent noise:

```python
def broadband_noise(frequency, tip_speed, chord, thickness):
    # Boundary layer noise
    St_1 = frequency * chord / tip_speed
    G1 = 10 * log10(St_1³ / ((St_1² + 0.25) * (St_1² + 0.0625)))

    # Tip vortex noise
    St_2 = frequency * thickness / tip_speed
    G2 = 10 * log10(St_2² / (St_2² + 0.0625))

    return G1 + G2
```

### 5.5 A-Weighting Filter

Human hearing correction:

```python
def a_weighting(frequency):
    f = frequency
    f2 = f**2

    numerator = 12194**2 * f2**2
    denominator = ((f2 + 20.6**2) *
                   sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) *
                   (f2 + 12194**2))

    return 20 * log10(numerator / denominator) + 2.0
```

---

## 6. State Validation and Numerical Stability

### 6.1 Physical Bounds Checking

**Position Bounds**:

```python
if norm(position) > MAX_POSITION:
    # Clamp to sphere
    position = position * MAX_POSITION / norm(position)
```

**Velocity Bounds**:

```python
if norm(velocity) > MAX_VELOCITY:
    velocity = velocity * MAX_VELOCITY / norm(velocity)
```

### 6.2 NaN and Infinity Detection

```python
def sanitize_state(state):
    # Replace NaN with zeros
    state = nan_to_num(state)

    # Check for infinities
    if any(isinf(state)):
        raise ValueError("Infinite values detected")

    # Normalize quaternion
    state[3:7] = state[3:7] / norm(state[3:7])

    return state
```

### 6.3 Numerical Conditioning

**Matrix Inversion Safety**:

```python
def safe_inverse(matrix, regularization=1e-12):
    # Add regularization to diagonal
    regularized = matrix + regularization * eye(matrix.shape[0])
    return inv(regularized)
```

**Quaternion Singularity Avoidance**:

```python
def quaternion_to_euler_safe(q):
    # Clamp sin(pitch) to avoid singularity
    sin_pitch = 2 * (q[0]*q[2] - q[3]*q[1])
    sin_pitch = clamp(sin_pitch, -0.99999, 0.99999)

    pitch = arcsin(sin_pitch)
    # ... continue with roll and yaw
```

---

## 7. Performance Optimization

### 7.1 Computational Complexity

**Per Simulation Step**:

- State derivative computation: O(1)
- RK4 integration: O(1)
- Control update: O(1)
- Propeller forces: O(n_props)
- Total: O(n_props)

### 7.2 Memory Management

**State History Buffer**:

```python
class CircularBuffer:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.index = 0
        self.size = 0
        self.max_size = max_size

    def append(self, item):
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
```

### 7.3 Real-Time Performance

**Adaptive Time Stepping**:

```python
def adaptive_step_size(error_estimate, tolerance):
    if error_estimate > tolerance:
        return dt * 0.8  # Reduce step size
    elif error_estimate < tolerance / 10:
        return dt * 1.2  # Increase step size
    else:
        return dt  # Keep current step size
```

This mathematical foundation provides the theoretical basis for understanding how the drone simulation achieves physics-accurate behavior while maintaining real-time performance.
