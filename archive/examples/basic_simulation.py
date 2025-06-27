#!/usr/bin/env python3
"""
Basic drone simulation example

This example demonstrates how to set up and run a basic drone simulation
with the drone_sim package.
"""

import numpy as np
import matplotlib.pyplot as plt
from drone_sim import (
    Simulator, SimulationConfig,
    StateManager, DroneState,
    RigidBody, RigidBodyConfig,
    Environment, EnvironmentConfig,
    Propeller, PropellerConfig, PropellerArray,
    PIDController, ControllerState, ControllerReference
)

def main():
    """Run basic simulation example"""
    
    print("Setting up drone simulation...")
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        dt=0.002,              # 2ms time step
        real_time_factor=1.0,  # Real time
        max_steps=50000        # 100 seconds
    )
    
    # Create simulator
    simulator = Simulator(sim_config)
    
    # Set up rigid body
    inertia_matrix = np.diag([0.02, 0.02, 0.04])  # kg⋅m²
    rigid_body_config = RigidBodyConfig(mass=1.5, inertia=inertia_matrix)
    rigid_body = RigidBody(rigid_body_config)
    
    # Set up propellers (quadcopter configuration)
    propeller_config = PropellerConfig(
        diameter=0.24,
        pitch=0.12,
        blades=2,
        ct_coefficient=1.1e-5,
        cq_coefficient=1.9e-6
    )
    
    propeller_array = PropellerArray()
    
    # Add 4 propellers in X configuration
    positions = [
        np.array([0.2, 0.2, 0.0]),    # Front right
        np.array([-0.2, -0.2, 0.0]),  # Rear left
        np.array([-0.2, 0.2, 0.0]),   # Front left
        np.array([0.2, -0.2, 0.0])    # Rear right
    ]
    
    for i, pos in enumerate(positions):
        propeller = Propeller(propeller_config)
        propeller_array.add_propeller(i, propeller, pos)
    
    # Set up environment
    environment = Environment()
    
    # Set up state manager
    state_manager = StateManager()
    
    # Set up PID controller
    controller = PIDController()
    
    # Register components with simulator
    simulator.register_physics_engine(rigid_body)
    simulator.register_environment(environment)
    
    # Set initial conditions
    initial_state = DroneState()
    initial_state.position = np.array([0.0, 0.0, -1.0])  # Start 1m above ground
    state_manager.set_state(initial_state)
    
    # Set reference trajectory (hover at 2m altitude)
    reference = ControllerReference()
    reference.position = np.array([0.0, 0.0, -2.0])
    reference.velocity = np.zeros(3)
    
    print("Starting simulation...")
    
    # Data logging
    time_history = []
    position_history = []
    velocity_history = []
    control_history = []
    
    # Simulation loop
    for step in range(sim_config.max_steps):
        current_time = step * sim_config.dt
        
        # Get current state
        current_state_data = state_manager.get_state()
        
        # Convert to controller format
        controller_state = ControllerState(
            position=current_state_data.position,
            quaternion=current_state_data.quaternion,
            velocity=current_state_data.velocity,
            angular_velocity=current_state_data.angular_velocity
        )
        
        # Update controller
        control_output = controller.update(reference, controller_state, sim_config.dt)
        
        # Simple motor mixing (equal thrust distribution)
        motor_thrust = control_output.thrust / 4.0
        motor_rpms = {
            0: np.sqrt(motor_thrust / (propeller_config.ct_coefficient * 1.225 * propeller_config.diameter**4)) * 60,
            1: np.sqrt(motor_thrust / (propeller_config.ct_coefficient * 1.225 * propeller_config.diameter**4)) * 60,
            2: np.sqrt(motor_thrust / (propeller_config.ct_coefficient * 1.225 * propeller_config.diameter**4)) * 60,
            3: np.sqrt(motor_thrust / (propeller_config.ct_coefficient * 1.225 * propeller_config.diameter**4)) * 60
        }
        
        # Calculate forces and moments
        forces_moments = propeller_array.calculate_total_forces_and_moments(motor_rpms)
        
        # Apply forces to rigid body
        rigid_body.clear_forces_and_moments()
        rigid_body.apply_force(forces_moments['force'])
        rigid_body.apply_moment(forces_moments['moment'])
        
        # Add gravity
        gravity_force = environment.get_gravity_force(rigid_body_config.mass)
        rigid_body.apply_force(gravity_force)
        
        # Update physics
        current_state_vector = state_manager.get_state_vector()
        derivatives = rigid_body.compute_derivatives(current_state_vector, current_time)
        
        # Simple Euler integration (for demonstration)
        new_state_vector = current_state_vector + derivatives * sim_config.dt
        state_manager.set_state_vector(new_state_vector, current_time)
        
        # Log data every 10 steps
        if step % 10 == 0:
            time_history.append(current_time)
            position_history.append(current_state_data.position.copy())
            velocity_history.append(current_state_data.velocity.copy())
            control_history.append(control_output.thrust)
            
        # Print progress
        if step % 1000 == 0:
            pos = current_state_data.position
            print(f"t={current_time:.2f}s: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m")
            
        # Break if simulation is complete
        if step > 10000:  # Run for 20 seconds
            break
    
    print("Simulation complete!")
    
    # Plot results
    plot_results(time_history, position_history, velocity_history, control_history)

def plot_results(time_history, position_history, velocity_history, control_history):
    """Plot simulation results"""
    
    time_array = np.array(time_history)
    position_array = np.array(position_history)
    velocity_array = np.array(velocity_history)
    control_array = np.array(control_history)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Position plot
    axes[0].plot(time_array, position_array[:, 0], 'r-', label='X')
    axes[0].plot(time_array, position_array[:, 1], 'g-', label='Y')
    axes[0].plot(time_array, position_array[:, 2], 'b-', label='Z')
    axes[0].axhline(y=-2.0, color='k', linestyle='--', alpha=0.5, label='Reference Z')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Drone Position')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(0, 1100)  # Set y-axis limits for position
    
    # Velocity plot
    axes[1].plot(time_array, velocity_array[:, 0], 'r-', label='VX')
    axes[1].plot(time_array, velocity_array[:, 1], 'g-', label='VY')
    axes[1].plot(time_array, velocity_array[:, 2], 'b-', label='VZ')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_title('Drone Velocity')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(0, 60)  # Set y-axis limits for velocity
    
    # Control plot
    axes[2].plot(time_array, control_array, 'k-', label='Thrust')
    axes[2].set_ylabel('Thrust (N)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Control Thrust')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylim(0, 30)  # Set y-axis limits for thrust
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 