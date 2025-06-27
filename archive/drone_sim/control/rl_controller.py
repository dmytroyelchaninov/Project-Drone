"""
Reinforcement Learning Controller for autonomous drone navigation
Supports obstacle avoidance, path planning, and adaptive control
"""

import numpy as np
import time
import json
import pickle
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. RL controller will use simplified learning.")

from .base_controller import BaseController, ControllerState, ControllerReference, ControllerOutput

class RLMode(Enum):
    """RL controller modes"""
    TRAINING = "training"
    INFERENCE = "inference"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"

class ActionType(Enum):
    """Types of actions the RL agent can take"""
    VELOCITY_CONTROL = "velocity"
    POSITION_CONTROL = "position"
    DIRECT_CONTROL = "direct"

@dataclass
class Obstacle:
    """Obstacle definition for navigation"""
    position: np.ndarray
    size: np.ndarray  # [width, height, depth]
    shape: str = "box"  # "box", "sphere", "cylinder"
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to obstacle"""
        if self.shape == "box":
            # Distance to box
            diff = np.abs(point - self.position) - self.size / 2
            return np.linalg.norm(np.maximum(diff, 0))
        elif self.shape == "sphere":
            # Distance to sphere
            return max(0, np.linalg.norm(point - self.position) - self.size[0])
        else:
            # Default to box
            return self.distance_to_point(point)

@dataclass
class Waypoint:
    """Waypoint for navigation"""
    position: np.ndarray
    tolerance: float = 0.5
    reached: bool = False
    
    def is_reached(self, current_position: np.ndarray) -> bool:
        """Check if waypoint is reached"""
        distance = np.linalg.norm(current_position - self.position)
        return distance <= self.tolerance

@dataclass
class RLConfig:
    """Configuration for RL controller"""
    # Network architecture
    state_dim: int = 15  # [pos, vel, ang_vel, target_pos, obstacles]
    action_dim: int = 4   # [engine1, engine2, engine3, engine4] thrust commands
    hidden_dim: int = 256
    num_layers: int = 3
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    
    # Reward parameters
    distance_reward_weight: float = 10.0
    obstacle_penalty_weight: float = -5.0
    velocity_penalty_weight: float = -0.05
    goal_reward: float = 200.0
    crash_penalty: float = -50.0
    progress_reward_weight: float = 5.0
    stability_reward_weight: float = 2.0
    
    # Physics parameters
    max_thrust_per_engine: float = 10.0  # Newtons per engine
    min_thrust_per_engine: float = 0.0   # Minimum thrust per engine
    hover_thrust_per_engine: float = 3.675  # Hover thrust (1.5kg * 9.81 / 4)
    
    # Safety parameters
    min_obstacle_distance: float = 1.0
    max_velocity: float = 5.0
    max_angular_velocity: float = 1.0
    
    # Training parameters
    max_episode_length: int = 1000
    save_frequency: int = 100
    evaluation_frequency: int = 50

class SimpleQLearning:
    """Simplified Q-learning implementation when PyTorch is not available"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.q_table = {}
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        
    def discretize_state(self, state: np.ndarray) -> str:
        """Discretize continuous state for Q-table"""
        # Simple discretization - round to nearest 0.5
        discrete_state = np.round(state * 2) / 2
        return str(discrete_state.tolist())
    
    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action using epsilon-greedy policy"""
        state_key = self.discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            # Random action
            action = np.random.uniform(-1, 1, self.config.action_dim)
        else:
            # Greedy action
            if state_key in self.q_table:
                # Find best action
                best_action_idx = np.argmax(list(self.q_table[state_key].values()))
                action_keys = list(self.q_table[state_key].keys())
                action = np.array(eval(action_keys[best_action_idx]))
            else:
                # Random action if state not seen
                action = np.random.uniform(-1, 1, self.config.action_dim)
        
        return action
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Update Q-table"""
        state_key = self.discretize_state(state)
        action_key = str(np.round(action * 2) / 2)
        next_state_key = self.discretize_state(next_state)
        
        # Initialize Q-values if not seen
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state_key in self.q_table:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q-learning update
        current_q = self.q_table[state_key][action_key]
        self.q_table[state_key][action_key] = current_q + self.learning_rate * (
            reward + self.gamma * max_next_q - current_q
        )
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

class DQNNetwork(nn.Module):
    """Deep Q-Network for RL controller"""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        input_dim = config.state_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = config.hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """Forward pass"""
        return self.network(state)

class ExperienceBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        # Filter out None values and sample indices
        valid_experiences = [exp for exp in self.buffer if exp is not None]
        if len(valid_experiences) < batch_size:
            batch_size = len(valid_experiences)
        
        indices = np.random.choice(len(valid_experiences), batch_size, replace=False)
        batch = [valid_experiences[i] for i in indices]
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

class RLController(BaseController):
    """
    Reinforcement Learning Controller for autonomous drone navigation
    Learns to navigate through obstacles and reach waypoints
    """
    
    def __init__(self, config: RLConfig = None):
        super().__init__("RLController")
        
        self.config = config or RLConfig()
        self.mode = RLMode.TRAINING
        self.action_type = ActionType.DIRECT_CONTROL
        
        # Environment
        self.obstacles: List[Obstacle] = []
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        
        # RL components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if PYTORCH_AVAILABLE else None
        
        if PYTORCH_AVAILABLE:
            self.q_network = DQNNetwork(self.config).to(self.device)
            self.target_network = DQNNetwork(self.config).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
            self.experience_buffer = ExperienceBuffer(self.config.buffer_size)
        else:
            self.simple_q = SimpleQLearning(self.config)
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.last_state = None
        self.last_action = None
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = 0.0
        self.collision_count = 0
        
        # Learning callbacks
        self.episode_end_callback: Optional[Callable] = None
        self.collision_callback: Optional[Callable] = None
        self.waypoint_reached_callback: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
        
        print(f"✅ RL Controller initialized with {'PyTorch' if PYTORCH_AVAILABLE else 'Simple Q-learning'}")
    
    def set_mode(self, mode: RLMode):
        """Set RL controller mode"""
        self.mode = mode
        print(f"🤖 RL mode set to: {mode.value}")
    
    def set_obstacles(self, obstacles: List[Obstacle]):
        """Set obstacles for navigation"""
        self.obstacles = obstacles
        print(f"🚧 Set {len(obstacles)} obstacles")
    
    def set_waypoints(self, waypoints: List[Waypoint]):
        """Set waypoints for navigation"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        for wp in self.waypoints:
            wp.reached = False
        print(f"📍 Set {len(waypoints)} waypoints")
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add single obstacle"""
        self.obstacles.append(obstacle)
    
    def add_waypoint(self, waypoint: Waypoint):
        """Add single waypoint"""
        self.waypoints.append(waypoint)
    
    def clear_environment(self):
        """Clear all obstacles and waypoints"""
        self.obstacles.clear()
        self.waypoints.clear()
        self.current_waypoint_idx = 0
    
    def _get_state_vector(self, current_state: ControllerState) -> np.ndarray:
        """Convert current state to RL state vector"""
        state_vector = []
        
        # Current position and velocity
        state_vector.extend(current_state.position)
        state_vector.extend(current_state.velocity)
        state_vector.extend(current_state.angular_velocity)
        
        # Target position (current waypoint)
        if self.current_waypoint_idx < len(self.waypoints):
            target_pos = self.waypoints[self.current_waypoint_idx].position
        else:
            target_pos = current_state.position  # Stay in place if no waypoints
        
        state_vector.extend(target_pos)
        
        # Distance to nearest obstacle
        min_obstacle_dist = float('inf')
        if self.obstacles:
            for obstacle in self.obstacles:
                dist = obstacle.distance_to_point(current_state.position)
                min_obstacle_dist = min(min_obstacle_dist, dist)
        else:
            min_obstacle_dist = 10.0  # Large distance if no obstacles
        
        state_vector.append(min_obstacle_dist)
        
        # Distance to target
        target_dist = np.linalg.norm(current_state.position - target_pos)
        state_vector.append(target_dist)
        
        # Progress (waypoint index / total waypoints)
        progress = self.current_waypoint_idx / max(1, len(self.waypoints))
        state_vector.append(progress)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _get_action(self, state_vector: np.ndarray) -> np.ndarray:
        """Get action from RL policy"""
        if PYTORCH_AVAILABLE:
            return self._get_dqn_action(state_vector)
        else:
            return self.simple_q.get_action(state_vector, self.mode == RLMode.TRAINING)
    
    def _get_dqn_action(self, state_vector: np.ndarray) -> np.ndarray:
        """Get action from DQN"""
        if self.mode == RLMode.TRAINING and np.random.random() < self._get_epsilon():
            # Exploration: random action
            action = np.random.uniform(-1, 1, self.config.action_dim)
        else:
            # Exploitation: use network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = torch.tanh(q_values).cpu().numpy()[0]
        
        return action
    
    def _get_epsilon(self) -> float:
        """Get current epsilon for exploration"""
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                 np.exp(-self.step_count / 10000)
        return epsilon
    
    def _calculate_reward(self, current_state: ControllerState, action: np.ndarray, 
                         next_state: ControllerState) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward for the current step"""
        reward = 0.0
        done = False
        info = {}
        
        # Base survival reward - positive for staying alive
        reward += 1.0  # Base reward for each step
        
        # Distance-based reward (primary signal)
        if self.current_waypoint_idx < len(self.waypoints):
            target_pos = self.waypoints[self.current_waypoint_idx].position
            current_distance = np.linalg.norm(current_state.position - target_pos)
            next_distance = np.linalg.norm(next_state.position - target_pos)
            
            # Progress reward - positive when getting closer
            distance_improvement = current_distance - next_distance
            reward += self.config.progress_reward_weight * distance_improvement
            
            # Distance reward - inverse distance (closer = better)
            max_distance = 20.0  # Normalize distance
            distance_reward = self.config.distance_reward_weight * (1.0 - min(next_distance / max_distance, 1.0))
            reward += distance_reward
            
            # Store previous distance for progress tracking
            if not hasattr(self, 'previous_distance'):
                self.previous_distance = current_distance
            
        # Waypoint reached reward
        if self.current_waypoint_idx < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_idx]
            if waypoint.is_reached(next_state.position):
                reward += self.config.goal_reward
                waypoint.reached = True
                self.current_waypoint_idx += 1
                info['waypoint_reached'] = True
                
                if self.waypoint_reached_callback:
                    self.waypoint_reached_callback(self.current_waypoint_idx - 1)
                
                # Check if all waypoints reached
                if self.current_waypoint_idx >= len(self.waypoints):
                    done = True
                    reward += self.config.goal_reward * 2  # Bonus for completing mission
                    info['mission_complete'] = True
        
        # Stability reward - reward for smooth, controlled flight
        velocity_magnitude = np.linalg.norm(next_state.velocity)
        angular_velocity_magnitude = np.linalg.norm(next_state.angular_velocity)
        
        # Reward moderate velocities (not too fast, not too slow)
        optimal_velocity = 2.0  # m/s
        velocity_factor = 1.0 - abs(velocity_magnitude - optimal_velocity) / optimal_velocity
        reward += self.config.stability_reward_weight * max(velocity_factor, 0.0)
        
        # Reward low angular velocity (stable orientation)
        angular_stability = 1.0 - min(angular_velocity_magnitude / self.config.max_angular_velocity, 1.0)
        reward += self.config.stability_reward_weight * angular_stability
        
        # Obstacle avoidance penalty (only when very close)
        min_obstacle_dist = float('inf')
        for obstacle in self.obstacles:
            dist = obstacle.distance_to_point(next_state.position)
            min_obstacle_dist = min(min_obstacle_dist, dist)
            
            if dist < self.config.min_obstacle_distance:
                # Collision or near collision
                penalty = self.config.obstacle_penalty_weight * (self.config.min_obstacle_distance - dist)
                reward += penalty
                
                if dist < 0.1:  # Actual collision
                    reward += self.config.crash_penalty
                    done = True
                    info['collision'] = True
                    self.collision_count += 1
                    
                    if self.collision_callback:
                        self.collision_callback(obstacle)
        
        # Velocity penalty (only for extreme velocities)
        if velocity_magnitude > self.config.max_velocity:
            reward += self.config.velocity_penalty_weight * (velocity_magnitude - self.config.max_velocity)
        
        # Angular velocity penalty (only for extreme angular velocities)
        if angular_velocity_magnitude > self.config.max_angular_velocity:
            reward += self.config.velocity_penalty_weight * (angular_velocity_magnitude - self.config.max_angular_velocity)
        
        # Reduce time penalty - small penalty for inefficiency
        reward -= 0.005  # Very small penalty for each step
        
        return reward, done, info
    
    def _update_policy(self, state: np.ndarray, action: np.ndarray, reward: float, 
                      next_state: np.ndarray, done: bool):
        """Update RL policy"""
        if PYTORCH_AVAILABLE:
            self._update_dqn(state, action, reward, next_state, done)
        else:
            self.simple_q.update(state, action, reward, next_state)
    
    def _update_dqn(self, state: np.ndarray, action: np.ndarray, reward: float, 
                   next_state: np.ndarray, done: bool):
        """Update DQN"""
        # Store experience
        self.experience_buffer.push(state, action, reward, next_state, done)
        
        # Update network if enough experiences
        if len(self.experience_buffer) >= self.config.batch_size:
            self._train_dqn()
        
        # Update target network
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _train_dqn(self):
        """Train DQN on batch of experiences"""
        # Sample batch
        batch = self.experience_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            target_q_values = rewards + (self.config.gamma * torch.max(next_q_values, dim=1)[0] * ~dones)
        
        # Compute loss
        loss = F.mse_loss(torch.max(current_q_values, dim=1)[0], target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
    def update(self, reference: ControllerReference, 
               current_state: ControllerState, 
               dt: float) -> ControllerOutput:
        """
        Update the RL controller
        
        Args:
            reference: Reference state (may be overridden by RL policy)
            current_state: Current drone state
            dt: Time step
            
        Returns:
            Control output from RL policy
        """
        if not self.enabled:
            return ControllerOutput()
        
        # Get state vector for RL
        state_vector = self._get_state_vector(current_state)
        
        # Get action from RL policy
        action = self._get_action(state_vector)
        
        # Calculate reward and update policy if training
        if self.mode == RLMode.TRAINING and self.last_state is not None:
            reward, done, info = self._calculate_reward(
                self._state_from_vector(self.last_state), 
                self.last_action, 
                current_state
            )
            
            self.total_reward += reward
            
            # Update policy
            self._update_policy(self.last_state, self.last_action, reward, state_vector, done)
            
            # Handle episode end
            if done or self.step_count % self.config.max_episode_length == 0:
                self._end_episode()
        
        # Store current state and action for next update
        self.last_state = state_vector.copy()
        self.last_action = action.copy()
        self.step_count += 1
        
        # Convert action to control output
        control_output = self._action_to_control_output(action, current_state)
        
        # Debug output every 100 steps
        if self.step_count % 100 == 0:
            # FIXED: Get target position using same logic as state vector
            if self.current_waypoint_idx < len(self.waypoints):
                target_pos = self.waypoints[self.current_waypoint_idx].position
            else:
                target_pos = current_state.position  # Stay in place if no waypoints
            
            distance_to_target = np.linalg.norm(current_state.position - target_pos)
            print(f"🤖 AI Step {self.step_count}: Pos={current_state.position}, Target={target_pos}, Dist={distance_to_target:.2f}m, Thrust={control_output.thrust:.2f}N, WP={self.current_waypoint_idx}/{len(self.waypoints)}")
        
        # Update metrics
        self._update_metrics(reference, current_state, control_output)
        
        return control_output
    
    def _state_from_vector(self, state_vector: np.ndarray) -> ControllerState:
        """Convert state vector back to ControllerState (simplified)"""
        return ControllerState(
            position=state_vector[0:3],
            velocity=state_vector[3:6],
            angular_velocity=state_vector[6:9],
            quaternion=np.array([1.0, 0.0, 0.0, 0.0])  # Simplified
        )
    
    def _action_to_control_output(self, action: np.ndarray, current_state: ControllerState) -> ControllerOutput:
        """Convert RL action to control output with individual engine thrusts"""
        if self.action_type == ActionType.VELOCITY_CONTROL:
            # Convert velocity commands to engine thrust commands through quadcopter controller
            from .quadcopter_controller import QuadcopterController, QuadcopterConfig
            
            # Create temporary quadcopter controller if needed
            if not hasattr(self, '_quad_controller'):
                config = QuadcopterConfig(mass=1.5)  # Use simulation mass
                self._quad_controller = QuadcopterController(config)
            
            # Map action to control inputs (throttle, roll, pitch, yaw)
            # Action is normalized to [-1, 1] range
            throttle = action[0]  # Vertical control
            roll = action[1]      # Roll control
            pitch = action[2]     # Pitch control
            yaw = action[3] if len(action) > 3 else 0.0  # Yaw control
            
            # Set control inputs to quadcopter controller
            self._quad_controller.set_control_inputs(
                throttle=throttle,
                roll=roll, 
                pitch=pitch,
                yaw=yaw
            )
            
            # Get physics-based control output
            from .base_controller import ControllerReference
            ref = ControllerReference()
            control_output = self._quad_controller.update(ref, current_state, 0.002)
            
            return control_output
        
        elif self.action_type == ActionType.DIRECT_CONTROL:
            # Direct engine thrust control (4 engines)
            if len(action) >= 4:
                # Map action [-1,1] to thrust range [0, max_thrust]
                max_thrust_per_engine = 10.0  # Newtons
                min_thrust_per_engine = 0.0
                
                # Normalize actions to [0, 1] then scale to thrust range
                normalized_action = (action[:4] + 1.0) / 2.0  # [-1,1] -> [0,1]
                engine_thrusts = (normalized_action * (max_thrust_per_engine - min_thrust_per_engine) + 
                                min_thrust_per_engine)
                
                # Calculate total thrust and moments from individual engines
                total_thrust = np.sum(engine_thrusts)
                
                # Calculate moments based on engine positions
                # Standard quadcopter layout: Front, Right, Back, Left
                arm_length = 0.225  # meters
                moments = np.zeros(3)
                
                # Engine positions (body frame)
                engine_positions = np.array([
                    [arm_length, 0.0, 0.0],    # Front
                    [0.0, arm_length, 0.0],    # Right
                    [-arm_length, 0.0, 0.0],   # Back  
                    [0.0, -arm_length, 0.0]    # Left
                ])
                
                for i, (pos, thrust) in enumerate(zip(engine_positions, engine_thrusts)):
                    force_vector = np.array([0.0, 0.0, thrust])
                    moment = np.cross(pos, force_vector)
                    moments += moment
                
                return ControllerOutput(
                    thrust=total_thrust,
                    moment=moments,
                    motor_commands=engine_thrusts
                )
            else:
                # Fallback to total thrust and moment control
                thrust = (action[0] + 1) * 25.0  # Map [-1,1] to [0,50]
                moment = action[1:4] * 2.0 if len(action) > 3 else np.zeros(3)
                
                return ControllerOutput(
                    thrust=thrust,
                    moment=moment
                )
        
        else:
            # Default fallback
            return ControllerOutput()
    
    def _end_episode(self):
        """Handle end of episode"""
        self.episode_rewards.append(self.total_reward)
        self.episode_lengths.append(self.step_count)
        self.episode_count += 1
        
        # Calculate success rate (last 100 episodes)
        recent_episodes = self.episode_rewards[-100:]
        success_threshold = 50.0  # Minimum reward for success
        successes = sum(1 for r in recent_episodes if r > success_threshold)
        self.success_rate = successes / len(recent_episodes)
        
        print(f"🤖 Episode {self.episode_count} complete: Reward={self.total_reward:.2f}, Steps={self.step_count}, Success Rate={self.success_rate:.2f}")
        
        # Save model periodically
        if self.episode_count % self.config.save_frequency == 0:
            self.save_model(f"rl_model_episode_{self.episode_count}.pth")
        
        # Reset for next episode
        self.total_reward = 0.0
        self.step_count = 0
        self.current_waypoint_idx = 0
        for wp in self.waypoints:
            wp.reached = False
        
        if self.episode_end_callback:
            self.episode_end_callback(self.episode_count, self.episode_rewards[-1], self.success_rate)
    
    def reset(self):
        """Reset the RL controller"""
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.step_count = 0
        self.current_waypoint_idx = 0
        
        for wp in self.waypoints:
            wp.reached = False
    
    def save_model(self, filename: str):
        """Save RL model"""
        if PYTORCH_AVAILABLE:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode_count': self.episode_count,
                'config': self.config
            }, filename)
        else:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'q_table': self.simple_q.q_table,
                    'epsilon': self.simple_q.epsilon,
                    'episode_count': self.episode_count,
                    'config': self.config
                }, f)
        
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load RL model"""
        if PYTORCH_AVAILABLE:
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_count = checkpoint['episode_count']
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.simple_q.q_table = data['q_table']
                self.simple_q.epsilon = data['epsilon']
                self.episode_count = data['episode_count']
        
        print(f"📂 Model loaded from {filename}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'episode_count': self.episode_count,
            'total_steps': sum(self.episode_lengths),
            'average_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'success_rate': self.success_rate,
            'collision_count': self.collision_count,
            'current_epsilon': self._get_epsilon() if PYTORCH_AVAILABLE else getattr(self.simple_q, 'epsilon', 0.0),
            'waypoints_completed': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints)
        }
    
    def set_episode_end_callback(self, callback: Callable):
        """Set callback for episode end"""
        self.episode_end_callback = callback
    
    def set_collision_callback(self, callback: Callable):
        """Set callback for collisions"""
        self.collision_callback = callback
    
    def set_waypoint_reached_callback(self, callback: Callable):
        """Set callback for waypoint reached"""
        self.waypoint_reached_callback = callback 