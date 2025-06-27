"""
Real-time interface for drone simulation
Supports both manual keyboard navigation and AI autonomous mode
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import numpy as np

# Optional imports with graceful degradation
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️  tkinter not available - GUI mode disabled")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - visualization disabled")

from ..core.simulator import Simulator, SimulationConfig
from ..core.state_manager import StateManager, DroneState
from ..physics.rigid_body import RigidBody, RigidBodyConfig
from ..physics.environment import Environment, EnvironmentConfig
from ..control.keyboard_controller import KeyboardController
from ..control.rl_controller import RLController, RLConfig, RLMode, Obstacle, Waypoint
from ..control.pid_controller import PIDController
from ..utils.background_validator import BackgroundValidator
from ..utils.test_logger import TestLogger

# Import movement logger
try:
    from ..logging.movement_logger import MovementLogger
except ImportError:
    MovementLogger = None


class SimulationMode(Enum):
    MANUAL = "manual"
    AI = "ai"
    HYBRID = "hybrid"

class InterfaceMode(Enum):
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class SimulationParameters:
    """User-configurable simulation parameters"""
    # Drone parameters
    mass: float = 1.5
    inertia_xx: float = 0.02
    inertia_yy: float = 0.02
    inertia_zz: float = 0.04
    
    # Environment parameters
    gravity: float = 9.81
    air_density: float = 1.225
    wind_velocity: List[float] = None
    
    # Simulation parameters
    dt: float = 0.002
    real_time_factor: float = 1.0
    physics_validation: bool = True
    
    # Initial conditions
    initial_position: List[float] = None
    initial_velocity: List[float] = None
    
    def __post_init__(self):
        if self.wind_velocity is None:
            self.wind_velocity = [0.0, 0.0, 0.0]
        if self.initial_position is None:
            self.initial_position = [0.0, 0.0, -2.0]
        if self.initial_velocity is None:
            self.initial_velocity = [0.0, 0.0, 0.0]

class RealTimeInterface:
    """
    Real-time interface for drone simulation
    Provides GUI for parameter configuration, manual control, and AI training
    """
    
    def __init__(self):
        self.simulation_mode = SimulationMode.MANUAL
        self.interface_mode = InterfaceMode.SETUP
        
        # Simulation components
        self.simulator: Optional[Simulator] = None
        self.keyboard_controller: Optional[KeyboardController] = None
        self.rl_controller: Optional[RLController] = None
        self.pid_controller: Optional[PIDController] = None
        self.background_validator: Optional[BackgroundValidator] = None
        self.test_logger: Optional[TestLogger] = None
        
        # Parameters
        self.sim_params = SimulationParameters()
        
        # Environment
        self.obstacles: List[Obstacle] = []
        self.waypoints: List[Waypoint] = []
        
        # UI components
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.figure: Optional[Figure] = None
        
        # Simulation data
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.control_history = []
        self.reward_history = []
        
        # Episode tracking for visualization (reset per episode)
        self.current_episode_positions = []
        self.current_episode_velocities = []
        self.current_episode_times = []
        self.current_episode_reward = 0.0
        self.episode_start_time = 0.0
        
        # Command delay system for realistic control
        self.command_delay = 0.05  # 50ms delay (realistic for drone systems)
        self.command_buffer = []  # Buffer for delayed commands
        self.last_executed_command = None
        
        # Threading
        self.simulation_thread: Optional[threading.Thread] = None
        self.running = False
        self.paused = False
        
        # Callbacks for external logging/monitoring
        self.status_callback: Optional[Callable] = None
        self.control_callback: Optional[Callable] = None
        self.ai_callback: Optional[Callable] = None
        self.event_callback: Optional[Callable] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize movement logger
        if MovementLogger:
            self.movement_logger = MovementLogger(enabled=True)
        else:
            print("⚠️  Movement logger not available")
            self.movement_logger = None
        
        # Initialize UI if available
        if TKINTER_AVAILABLE:
            self._initialize_ui()
        else:
            print("⚠️ Tkinter not available. Running in console mode.")
            self._initialize_console_mode()
        
        # Connect movement logger to keyboard controller
        if self.keyboard_controller and self.movement_logger:
            self.keyboard_controller.set_movement_logger(self.movement_logger)
    
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates (position, velocity, mode info)"""
        self.status_callback = callback
    
    def set_control_callback(self, callback: Callable):
        """Set callback for control input updates (thrust, moment)"""
        self.control_callback = callback
    
    def set_ai_callback(self, callback: Callable):
        """Set callback for AI progress updates"""
        self.ai_callback = callback
    
    def set_event_callback(self, callback: Callable):
        """Set callback for general events"""
        self.event_callback = callback
    
    def _trigger_event_callback(self, event_type: str, message: str):
        """Trigger event callback if set"""
        if self.event_callback:
            try:
                self.event_callback(event_type, message)
            except Exception as e:
                self.logger.warning(f"Event callback error: {e}")
    
    def _trigger_status_callback(self, position: np.ndarray, velocity: np.ndarray, mode_info: Dict):
        """Trigger status callback if set"""
        if self.status_callback:
            try:
                self.status_callback(position, velocity, mode_info)
            except Exception as e:
                self.logger.warning(f"Status callback error: {e}")
    
    def _trigger_control_callback(self, thrust: float, moment: np.ndarray):
        """Trigger control callback if set"""
        if self.control_callback:
            try:
                self.control_callback(thrust, moment)
            except Exception as e:
                self.logger.warning(f"Control callback error: {e}")
    
    def _trigger_ai_callback(self, stats: Dict):
        """Trigger AI callback if set"""
        if self.ai_callback:
            try:
                self.ai_callback(stats)
            except Exception as e:
                self.logger.warning(f"AI callback error: {e}")
    
    def _initialize_ui(self):
        """Initialize the graphical user interface"""
        self.root = tk.Tk()
        self.root.title("Drone Simulation - Real-time Interface")
        self.root.geometry("1400x900")
        
        # Create main frames
        self._create_control_panel()
        self._create_visualization_panel()
        self._create_status_panel()
        
        # Bind keyboard events to the main window
        if TKINTER_AVAILABLE and self.root:
            # Make sure the window can receive focus and keyboard events
            self.root.focus_set()  # Enable keyboard focus
            self.root.focus_force()  # Force focus 
            
            # Make the window focusable
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
            # Use bind_all to catch events even when other widgets have focus
            # This ensures keyboard controls work regardless of which widget is focused
            self.root.bind_all('<KeyPress>', self._on_key_press)
            self.root.bind_all('<KeyRelease>', self._on_key_release)
        
        print("✅ GUI initialized successfully")
    
    def _create_control_panel(self):
        """Create the control panel"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Simulation Mode")
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.mode_var = tk.StringVar(value=self.simulation_mode.value)
        ttk.Radiobutton(mode_frame, text="Manual Control", variable=self.mode_var, 
                       value="manual", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="AI Navigation", variable=self.mode_var, 
                       value="ai", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Hybrid Mode", variable=self.mode_var, 
                       value="hybrid", command=self._on_mode_change).pack(anchor=tk.W)
        
        # Clear Instructions Panel
        instructions_frame = ttk.LabelFrame(control_frame, text="📋 Quick Start Guide")
        instructions_frame.pack(fill=tk.X, pady=5)
        
        instructions_text = """
            🚀 WORKFLOW:
            1. Select simulation mode above
            2. Configure parameters below
            3. Click 'Start Simulation' to begin
            4. For AI mode: Use training controls

            ⚡ AI TRAINING:
            • Click 'Start Training' to begin episodes
            • Each episode runs for 12 seconds
            • Click 'Pause Training' to save progress
            • Click 'Resume Training' to continue
            • Use 'Reset Position' to restart drone
            • Training auto-saves every 10 episodes
        """
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                                     font=('Arial', 12), justify=tk.LEFT)
        instructions_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # Parameters
        params_frame = ttk.LabelFrame(control_frame, text="Parameters")
        params_frame.pack(fill=tk.X, pady=5)
        
        # Drone parameters
        ttk.Label(params_frame, text="Mass (kg):").grid(row=0, column=0, sticky=tk.W)
        self.mass_var = tk.DoubleVar(value=self.sim_params.mass)
        ttk.Entry(params_frame, textvariable=self.mass_var, width=10).grid(row=0, column=1)
        
        ttk.Label(params_frame, text="Real-time Factor:").grid(row=1, column=0, sticky=tk.W)
        self.rtf_var = tk.DoubleVar(value=self.sim_params.real_time_factor)
        ttk.Entry(params_frame, textvariable=self.rtf_var, width=10).grid(row=1, column=1)
        
        # Episode Length for AI Training
        ttk.Label(params_frame, text="Episode Length (s):").grid(row=2, column=0, sticky=tk.W)
        self.episode_length_var = tk.DoubleVar(value=12.0)
        ttk.Entry(params_frame, textvariable=self.episode_length_var, width=10).grid(row=2, column=1)
        
        # Command Delay for Realism
        ttk.Label(params_frame, text="Command Delay (ms):").grid(row=3, column=0, sticky=tk.W)
        self.command_delay_var = tk.DoubleVar(value=50.0)
        ttk.Entry(params_frame, textvariable=self.command_delay_var, width=10).grid(row=3, column=1)
        
        # Environment setup
        env_frame = ttk.LabelFrame(control_frame, text="Environment")
        env_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(env_frame, text="Add Obstacle", command=self._add_obstacle_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(env_frame, text="Add Waypoint", command=self._add_waypoint_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(env_frame, text="Clear Environment", command=self._clear_environment).pack(fill=tk.X, pady=2)
        
        # Main Control buttons
        control_buttons_frame = ttk.LabelFrame(control_frame, text="🎮 Simulation Control")
        control_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(control_buttons_frame, text="🚀 Start Simulation", 
                                     command=self._start_simulation, style="Accent.TButton")
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.pause_button = ttk.Button(control_buttons_frame, text="⏸️ Pause", 
                                     command=self._pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(control_buttons_frame, text="🛑 Stop", 
                                    command=self._stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        self.reset_button = ttk.Button(control_buttons_frame, text="🔄 Reset Position", 
                                     command=self._reset_drone_position)
        self.reset_button.pack(fill=tk.X, pady=2)
        
        # AI Training Controls (only visible in AI mode)
        self.ai_frame = ttk.LabelFrame(control_frame, text="🤖 AI Training Control")
        
        # Training session management
        training_session_frame = ttk.Frame(self.ai_frame)
        training_session_frame.pack(fill=tk.X, pady=2)
        
        self.start_training_button = ttk.Button(training_session_frame, text="🎯 Start Training", 
                                              command=self._start_training_session, 
                                              style="Accent.TButton")
        self.start_training_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.pause_training_button = ttk.Button(training_session_frame, text="⏸️ Pause Training", 
                                              command=self._pause_training_session, state=tk.DISABLED)
        self.pause_training_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        self.resume_training_button = ttk.Button(self.ai_frame, text="▶️ Resume Training", 
                                               command=self._resume_training_session, state=tk.DISABLED)
        self.resume_training_button.pack(fill=tk.X, pady=2)
        
        self.stop_training_button = ttk.Button(self.ai_frame, text="🛑 Stop Training & Save", 
                                             command=self._stop_training_session, state=tk.DISABLED)
        self.stop_training_button.pack(fill=tk.X, pady=2)
        
        # Model management
        model_frame = ttk.LabelFrame(self.ai_frame, text="Model Management")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(model_frame, text="📂 Load Model", command=self._load_ai_model).pack(fill=tk.X, pady=1)
        ttk.Button(model_frame, text="💾 Save Model", command=self._save_ai_model).pack(fill=tk.X, pady=1)
        ttk.Button(model_frame, text="🗑️ Reset Training", command=self._reset_training).pack(fill=tk.X, pady=1)
        
        # Training Status
        self.training_status_frame = ttk.LabelFrame(self.ai_frame, text="Training Status")
        self.training_status_frame.pack(fill=tk.X, pady=5)
        
        self.training_status_label = ttk.Label(self.training_status_frame, text="⏹️ Not Training", 
                                             font=('Arial', 10, 'bold'))
        self.training_status_label.pack(pady=5)
        
        # Manual control help
        self.manual_frame = ttk.LabelFrame(control_frame, text="Manual Controls")
        self.manual_frame.pack(fill=tk.X, pady=5)
        
        help_text = """
            🎮 ENHANCED MANUAL CONTROLS:

            ARROW KEYS (Primary):
            ↑/↓: Thrust up/down (smooth)
            ←/→: Roll left/right

            WASD KEYS (Secondary):
            W/S: Pitch forward/backward  
            A/D: Yaw left/right

            SPECIAL KEYS:
            Space: Return to hover
            ESC: Emergency thrust reduction

            CONTROLS FEATURE:
            • Hold keys for accelerating effect
            • Smooth return to neutral on release
            • Physics-based response curves
        """
        ttk.Label(self.manual_frame, text=help_text, font=('Courier', 12)).pack()
        
        # Initialize training state
        self.training_active = False
        self.training_paused = False
        self.current_episode_start_time = None
    
    def _create_visualization_panel(self):
        """Create the visualization panel"""
        if not MATPLOTLIB_AVAILABLE:
            viz_frame = ttk.Frame(self.root)
            viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            ttk.Label(viz_frame, text="Visualization not available\n(matplotlib required)").pack(expand=True)
            return
        
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure with improved layout
        self.figure = Figure(figsize=(14, 10))
        
        # LARGER 3D trajectory plot (takes up left half, full height)
        self.ax_3d = self.figure.add_subplot(2, 4, (1, 6), projection='3d')  # Spans 2 rows, 2 columns
        self.ax_3d.set_title('🚁 3D Drone Trajectory', fontsize=14, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # Top right: Engine Thrust Display
        self.ax_engines = self.figure.add_subplot(2, 4, 3)
        self.ax_engines.set_title('🔧 Engine Thrusts', fontsize=12, fontweight='bold')
        self.ax_engines.set_xlim(-1.5, 1.5)
        self.ax_engines.set_ylim(-1.5, 1.5)
        self.ax_engines.set_aspect('equal')
        self.ax_engines.set_xticks([])
        self.ax_engines.set_yticks([])
        
        # Top far right: Drone Angles Display
        self.ax_angles = self.figure.add_subplot(2, 4, 4)
        self.ax_angles.set_title('📐 Drone Orientation', fontsize=12, fontweight='bold')
        self.ax_angles.set_xlim(-1.2, 1.2)
        self.ax_angles.set_ylim(-1.2, 1.2)
        self.ax_angles.set_aspect('equal')
        self.ax_angles.set_xticks([])
        self.ax_angles.set_yticks([])
        
        # Bottom right: Position vs time (smaller)
        self.ax_pos = self.figure.add_subplot(2, 4, 7)
        self.ax_pos.set_title('Position vs Time', fontsize=10)
        self.ax_pos.set_xlabel('Time (s)')
        self.ax_pos.set_ylabel('Position (m)')
        
        # Bottom far right: Velocity/AI progress (smaller)
        self.ax_vel = self.figure.add_subplot(2, 4, 8)
        self.ax_vel.set_title('Velocity/AI Progress', fontsize=10)
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        
        # Adjust layout to give 3D plot more space
        self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, 
                                   wspace=0.3, hspace=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_status_panel(self):
        """Create status and information panel"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Status text area with scrollbar
        status_text_frame = ttk.Frame(status_frame)
        status_text_frame.pack(fill="both", expand=True)
        
        self.status_text = tk.Text(status_text_frame, height=8, width=80, wrap=tk.WORD)
        status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side="left", fill="both", expand=True)
        status_scrollbar.pack(side="right", fill="y")
        
        # Enhanced AI Training Status Panel
        ai_frame = ttk.LabelFrame(status_frame, text="🤖 AI Training Status", padding=10)
        ai_frame.pack(fill="x", pady=(10, 0))
        
        # AI metrics display
        ai_metrics_frame = ttk.Frame(ai_frame)
        ai_metrics_frame.pack(fill="x")
        
        # Episode info
        ttk.Label(ai_metrics_frame, text="Episode:").grid(row=0, column=0, sticky="w")
        self.episode_label = ttk.Label(ai_metrics_frame, text="0", font=("Arial", 10, "bold"))
        self.episode_label.grid(row=0, column=1, sticky="w", padx=(5, 20))
        
        ttk.Label(ai_metrics_frame, text="Reward:").grid(row=0, column=2, sticky="w")
        self.reward_label = ttk.Label(ai_metrics_frame, text="0.0", font=("Arial", 10, "bold"))
        self.reward_label.grid(row=0, column=3, sticky="w", padx=(5, 20))
        
        # Success rate and exploration
        ttk.Label(ai_metrics_frame, text="Success Rate:").grid(row=1, column=0, sticky="w")
        self.success_rate_label = ttk.Label(ai_metrics_frame, text="0%", font=("Arial", 10, "bold"))
        self.success_rate_label.grid(row=1, column=1, sticky="w", padx=(5, 20))
        
        ttk.Label(ai_metrics_frame, text="Exploration:").grid(row=1, column=2, sticky="w")
        self.exploration_label = ttk.Label(ai_metrics_frame, text="100%", font=("Arial", 10, "bold"))
        self.exploration_label.grid(row=1, column=3, sticky="w", padx=(5, 20))
        
        # Waypoint progress
        ttk.Label(ai_metrics_frame, text="Waypoints:").grid(row=2, column=0, sticky="w")
        self.waypoint_label = ttk.Label(ai_metrics_frame, text="0/0", font=("Arial", 10, "bold"))
        self.waypoint_label.grid(row=2, column=1, sticky="w", padx=(5, 20))
        
        ttk.Label(ai_metrics_frame, text="Collisions:").grid(row=2, column=2, sticky="w")
        self.collision_label = ttk.Label(ai_metrics_frame, text="0", font=("Arial", 10, "bold"))
        self.collision_label.grid(row=2, column=3, sticky="w", padx=(5, 20))
        
        # Progress bars
        progress_frame = ttk.Frame(ai_frame)
        progress_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(progress_frame, text="Training Progress:").pack(anchor="w")
        self.training_progress = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.training_progress.pack(fill="x", pady=(2, 5))
        
        ttk.Label(progress_frame, text="Episode Progress:").pack(anchor="w")
        self.episode_progress = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.episode_progress.pack(fill="x", pady=(2, 0))
        
        # Real-time Process Status
        process_frame = ttk.LabelFrame(status_frame, text="⚙️ System Processes", padding=10)
        process_frame.pack(fill="x", pady=(10, 0))
        
        process_metrics_frame = ttk.Frame(process_frame)
        process_metrics_frame.pack(fill="x")
        
        # Physics validation
        ttk.Label(process_metrics_frame, text="Physics Validation:").grid(row=0, column=0, sticky="w")
        self.physics_status_label = ttk.Label(process_metrics_frame, text="✅ Active", foreground="green")
        self.physics_status_label.grid(row=0, column=1, sticky="w", padx=(5, 20))
        
        # Background logging
        ttk.Label(process_metrics_frame, text="Background Logging:").grid(row=0, column=2, sticky="w")
        self.logging_status_label = ttk.Label(process_metrics_frame, text="✅ Active", foreground="green")
        self.logging_status_label.grid(row=0, column=3, sticky="w", padx=(5, 20))
        
        # Performance metrics
        ttk.Label(process_metrics_frame, text="FPS:").grid(row=1, column=0, sticky="w")
        self.fps_label = ttk.Label(process_metrics_frame, text="0.0")
        self.fps_label.grid(row=1, column=1, sticky="w", padx=(5, 20))
        
        ttk.Label(process_metrics_frame, text="CPU:").grid(row=1, column=2, sticky="w")
        self.cpu_label = ttk.Label(process_metrics_frame, text="0%")
        self.cpu_label.grid(row=1, column=3, sticky="w", padx=(5, 20))
    
    def _initialize_console_mode(self):
        """Initialize console-only mode"""
        print("🚀 Console mode initialized")
        print("Available commands:")
        print("  start - Start simulation")
        print("  stop  - Stop simulation")
        print("  mode <manual|ai|hybrid> - Change mode")
        print("  quit  - Exit")
    
    def _on_mode_change(self):
        """Handle simulation mode change"""
        new_mode = SimulationMode(self.mode_var.get())
        old_mode = self.simulation_mode
        self.simulation_mode = new_mode
        
        if TKINTER_AVAILABLE:
            # Show/hide relevant control panels
            if new_mode == SimulationMode.AI:
                self.ai_frame.pack(fill=tk.X, pady=5)
                self.manual_frame.pack_forget()
                # Update training button states
                self._update_training_buttons()
            elif new_mode == SimulationMode.HYBRID:
                self.ai_frame.pack(fill=tk.X, pady=5)
                self.manual_frame.pack(fill=tk.X, pady=5)
                # Update training button states
                self._update_training_buttons()
            else:  # MANUAL
                self.manual_frame.pack(fill=tk.X, pady=5)
                self.ai_frame.pack_forget()
        
        self._log_status(f"🎮 Mode changed: {old_mode.value} → {new_mode.value}")
    
    def _on_key_press(self, event):
        """Handle GUI key press events"""
        if self.keyboard_controller and self.simulation_mode in [SimulationMode.MANUAL, SimulationMode.HYBRID]:
            # Map tkinter key symbols to keyboard controller key names
            key_name = event.keysym.lower()
            
            # Handle special cases for arrow keys
            if event.keysym == 'Up':
                key_name = 'up'
            elif event.keysym == 'Down':
                key_name = 'down'
            elif event.keysym == 'Left':
                key_name = 'left'
            elif event.keysym == 'Right':
                key_name = 'right'
            elif event.keysym == 'space':
                key_name = 'space'
            elif event.keysym == 'Escape':
                key_name = 'escape'
            
            # Set key state in keyboard controller
            self.keyboard_controller.set_key_state(key_name, True)
    
    def _on_key_release(self, event):
        """Handle GUI key release events"""
        if self.keyboard_controller and self.simulation_mode in [SimulationMode.MANUAL, SimulationMode.HYBRID]:
            # Map tkinter key symbols to keyboard controller key names
            key_name = event.keysym.lower()
            
            # Handle special cases for arrow keys
            if event.keysym == 'Up':
                key_name = 'up'
            elif event.keysym == 'Down':
                key_name = 'down'
            elif event.keysym == 'Left':
                key_name = 'left'
            elif event.keysym == 'Right':
                key_name = 'right'
            elif event.keysym == 'space':
                key_name = 'space'
            elif event.keysym == 'Escape':
                key_name = 'escape'
            
            # Set key state in keyboard controller
            self.keyboard_controller.set_key_state(key_name, False)
    
    def _add_obstacle_dialog(self):
        """Show dialog to add obstacle"""
        if not TKINTER_AVAILABLE:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Obstacle")
        dialog.geometry("300x200")
        
        # Position
        ttk.Label(dialog, text="Position (x, y, z):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        pos_x = tk.DoubleVar(value=5.0)
        pos_y = tk.DoubleVar(value=0.0)
        pos_z = tk.DoubleVar(value=-2.0)
        ttk.Entry(dialog, textvariable=pos_x, width=8).grid(row=0, column=1, padx=2)
        ttk.Entry(dialog, textvariable=pos_y, width=8).grid(row=0, column=2, padx=2)
        ttk.Entry(dialog, textvariable=pos_z, width=8).grid(row=0, column=3, padx=2)
        
        # Size
        ttk.Label(dialog, text="Size (w, h, d):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        size_x = tk.DoubleVar(value=1.0)
        size_y = tk.DoubleVar(value=1.0)
        size_z = tk.DoubleVar(value=2.0)
        ttk.Entry(dialog, textvariable=size_x, width=8).grid(row=1, column=1, padx=2)
        ttk.Entry(dialog, textvariable=size_y, width=8).grid(row=1, column=2, padx=2)
        ttk.Entry(dialog, textvariable=size_z, width=8).grid(row=1, column=3, padx=2)
        
        def add_obstacle():
            obstacle = Obstacle(
                position=np.array([pos_x.get(), pos_y.get(), pos_z.get()]),
                size=np.array([size_x.get(), size_y.get(), size_z.get()])
            )
            self.obstacles.append(obstacle)
            self._log_status(f"🚧 Added obstacle at {obstacle.position}")
            dialog.destroy()
        
        ttk.Button(dialog, text="Add", command=add_obstacle).grid(row=2, column=1, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=2, column=2, pady=10)
    
    def _add_waypoint_dialog(self):
        """Show dialog to add waypoint"""
        if not TKINTER_AVAILABLE:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Waypoint")
        dialog.geometry("300x150")
        
        # Position
        ttk.Label(dialog, text="Position (x, y, z):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        pos_x = tk.DoubleVar(value=10.0)
        pos_y = tk.DoubleVar(value=0.0)
        pos_z = tk.DoubleVar(value=-2.0)
        ttk.Entry(dialog, textvariable=pos_x, width=8).grid(row=0, column=1, padx=2)
        ttk.Entry(dialog, textvariable=pos_y, width=8).grid(row=0, column=2, padx=2)
        ttk.Entry(dialog, textvariable=pos_z, width=8).grid(row=0, column=3, padx=2)
        
        # Tolerance
        ttk.Label(dialog, text="Tolerance (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        tolerance = tk.DoubleVar(value=0.5)
        ttk.Entry(dialog, textvariable=tolerance, width=8).grid(row=1, column=1, padx=2)
        
        def add_waypoint():
            waypoint = Waypoint(
                position=np.array([pos_x.get(), pos_y.get(), pos_z.get()]),
                tolerance=tolerance.get()
            )
            self.waypoints.append(waypoint)
            self._log_status(f"📍 Added waypoint at {waypoint.position}")
            dialog.destroy()
        
        ttk.Button(dialog, text="Add", command=add_waypoint).grid(row=2, column=1, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=2, column=2, pady=10)
    
    def _clear_environment(self):
        """Clear all obstacles and waypoints"""
        self.obstacles.clear()
        self.waypoints.clear()
        self._log_status("🧹 Environment cleared")
    
    def _start_simulation(self):
        """Start the simulation"""
        if self.interface_mode == InterfaceMode.RUNNING:
            return
        
        try:
            # Update simulation parameters from UI
            self.sim_params.mass = self.mass_var.get()
            self.sim_params.real_time_factor = self.rtf_var.get()
            
            # Update command delay from UI (convert ms to seconds)
            if hasattr(self, 'command_delay_var'):
                self.command_delay = self.command_delay_var.get() / 1000.0
            
            # Initialize simulation components
            self._initialize_simulation()
            
            # Start simulation thread
            self.running = True
            self.paused = False
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()
            
            # Update UI state
            self.interface_mode = InterfaceMode.RUNNING
            if TKINTER_AVAILABLE:
                self.start_button.config(state=tk.DISABLED)
                self.pause_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.NORMAL)
            
            self._log_status("🚀 Simulation started")
            self._trigger_event_callback("START", "Simulation started")
            
        except Exception as e:
            self._log_status(f"❌ Failed to start simulation: {e}")
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
    
    def _pause_simulation(self):
        """Pause/resume the simulation"""
        if self.interface_mode == InterfaceMode.RUNNING:
            self.paused = True
            self.interface_mode = InterfaceMode.PAUSED
            if TKINTER_AVAILABLE:
                self.pause_button.config(text="Resume")
            self._log_status("⏸️ Simulation paused")
            self._trigger_event_callback("PAUSE", "Simulation paused")
        elif self.interface_mode == InterfaceMode.PAUSED:
            self.paused = False
            self.interface_mode = InterfaceMode.RUNNING
            if TKINTER_AVAILABLE:
                self.pause_button.config(text="Pause")
            self._log_status("▶️ Simulation resumed")
            self._trigger_event_callback("RESUME", "Simulation resumed")
    
    def _stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.interface_mode = InterfaceMode.STOPPED
        
        if TKINTER_AVAILABLE:
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED, text="Pause")
            self.stop_button.config(state=tk.DISABLED)
        
        self._log_status("🛑 Simulation stopped")
        self._trigger_event_callback("STOP", "Simulation stopped")
    
    def _reset_simulation(self):
        """Reset the simulation"""
        self._stop_simulation()
        
        # Clear data
        self.time_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.control_history.clear()
        self.reward_history.clear()
        
        # Reset controllers
        if self.keyboard_controller:
            self.keyboard_controller.reset()
        if self.rl_controller:
            self.rl_controller.reset()
        
        self.interface_mode = InterfaceMode.SETUP
        self._log_status("🔄 Simulation reset")
    
    def _initialize_simulation(self):
        """Initialize all simulation components"""
        # Create simulation configuration
        sim_config = SimulationConfig(
            dt=self.sim_params.dt,
            real_time_factor=self.sim_params.real_time_factor,
            physics_validation=self.sim_params.physics_validation
        )
        
        # Create simulator
        self.simulator = Simulator(sim_config)
        
        # Create rigid body
        inertia_matrix = np.diag([
            self.sim_params.inertia_xx,
            self.sim_params.inertia_yy,
            self.sim_params.inertia_zz
        ])
        rigid_body_config = RigidBodyConfig(
            mass=self.sim_params.mass,
            inertia=inertia_matrix
        )
        rigid_body = RigidBody(rigid_body_config)
        
        # Create environment
        env_config = EnvironmentConfig(
            gravity_magnitude=self.sim_params.gravity,
            air_density=self.sim_params.air_density
        )
        environment = Environment(env_config)
        if self.sim_params.wind_velocity != [0.0, 0.0, 0.0]:
            environment.set_constant_wind(np.array(self.sim_params.wind_velocity))
        
        # Create controllers based on mode
        if self.simulation_mode in [SimulationMode.MANUAL, SimulationMode.HYBRID]:
            # Create quadcopter config for keyboard controller
            from ..control.quadcopter_controller import QuadcopterConfig
            keyboard_config = QuadcopterConfig(mass=self.sim_params.mass)
            self.keyboard_controller = KeyboardController(keyboard_config)
            # FIXED: Don't start pygame input monitoring in GUI mode to avoid threading issues
            # Only start monitoring if running in console mode
            if not TKINTER_AVAILABLE or not self.root:
                self.keyboard_controller.start_input_monitoring()
                self._log_status("🎮 Keyboard controller initialized for console mode")
            else:
                # For GUI mode, we'll use tkinter key events instead of pygame
                self._log_status("🎮 Keyboard controller initialized for GUI mode (using tkinter events)")
        
        if self.simulation_mode in [SimulationMode.AI, SimulationMode.HYBRID]:
            self.rl_controller = RLController()
            self.rl_controller.set_obstacles(self.obstacles)
            self.rl_controller.set_waypoints(self.waypoints)
            # Set to training mode so it learns and explores
            self.rl_controller.set_mode(RLMode.TRAINING)
            self._log_status("🤖 RL controller initialized in TRAINING mode")
        
        # Always create PID controller as fallback
        self.pid_controller = PIDController()
        
        # Register components
        self.simulator.register_physics_engine(rigid_body)
        self.simulator.register_environment(environment)
        
        # Set initial state
        initial_state = DroneState()
        initial_state.position = np.array(self.sim_params.initial_position)
        initial_state.velocity = np.array(self.sim_params.initial_velocity)
        self.simulator.state_manager.set_state(initial_state)
        
        # Initialize background validation and logging
        self.background_validator = BackgroundValidator()
        self.background_validator.start_background_validation()
        
        self.test_logger = TestLogger("realtime_simulation")
        self.test_logger.start_test("Real-time Simulation Session")
        
        self._log_status("✅ Simulation components initialized")
        self._trigger_event_callback("INIT", "Simulation components initialized")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        step_count = 0
        last_update_time = time.time()
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Get current state
                current_state = self.simulator.state_manager.get_state()
                
                # Select active controller based on mode
                active_controller = None
                if self.simulation_mode == SimulationMode.MANUAL:
                    # Always use keyboard controller in manual mode
                    active_controller = self.keyboard_controller
                elif self.simulation_mode == SimulationMode.AI:
                    active_controller = self.rl_controller
                elif self.simulation_mode == SimulationMode.HYBRID:
                    # In hybrid mode, check if keyboard has active inputs
                    if self.keyboard_controller and any(self.keyboard_controller.key_states.values()):
                        active_controller = self.keyboard_controller
                    elif self.rl_controller:
                        active_controller = self.rl_controller
                
                # Fallback to PID controller
                if active_controller is None:
                    active_controller = self.pid_controller
                
                # Update controller
                from ..control.base_controller import ControllerReference, ControllerState
                reference = ControllerReference()
                controller_state = ControllerState(
                    position=current_state.position,
                    quaternion=current_state.quaternion,
                    velocity=current_state.velocity,
                    angular_velocity=current_state.angular_velocity
                )
                
                # Get control command (with delay simulation)
                raw_control_output = active_controller.update(reference, controller_state, self.sim_params.dt)
                
                # Add command to delay buffer
                if raw_control_output:
                    command_time = time.time()
                    self.command_buffer.append((command_time, raw_control_output))
                
                # Process delayed commands (realistic 50ms delay)
                control_output = None
                current_time_real = time.time()
                
                # Remove old commands and get the one that should execute now
                while self.command_buffer:
                    cmd_time, cmd_output = self.command_buffer[0]
                    if current_time_real - cmd_time >= self.command_delay:
                        control_output = cmd_output
                        self.command_buffer.pop(0)
                        break
                    else:
                        break
                
                # Use last executed command if no new command is ready
                if control_output is None:
                    control_output = self.last_executed_command
                else:
                    self.last_executed_command = control_output
                
                # Apply control to simulation - FIXED: Actually integrate physics!
                if control_output:
                    # Update physics state using control output
                    new_state = self._integrate_physics(current_state, control_output, self.sim_params.dt)
                    self.simulator.state_manager.set_state(new_state)
                    
                    # Trigger control callback
                    thrust = control_output.thrust
                    moment = getattr(control_output, 'moment', np.zeros(3))
                    self._trigger_control_callback(thrust, moment)
                
                # Log data
                current_time = step_count * self.sim_params.dt
                self.time_history.append(current_time)
                self.position_history.append(current_state.position.copy())
                self.velocity_history.append(current_state.velocity.copy())
                self.control_history.append(control_output.thrust if control_output else 0.0)
                
                # Track current episode data (for visualization)
                episode_time = current_time - self.episode_start_time
                self.current_episode_times.append(episode_time)
                self.current_episode_positions.append(current_state.position.copy())
                self.current_episode_velocities.append(current_state.velocity.copy())
                
                # Log AI rewards if applicable
                if self.rl_controller and hasattr(self.rl_controller, 'total_reward'):
                    self.reward_history.append(self.rl_controller.total_reward)
                    self.current_episode_reward = self.rl_controller.total_reward
                
                # Trigger status callback
                mode_info = {
                    'thrust': control_output.thrust if control_output else 0.0,
                    'step': step_count,
                    'time': current_time
                }
                
                # Add AI-specific info
                if self.rl_controller:
                    stats = self.rl_controller.get_learning_stats()
                    mode_info.update({
                        'episode_count': stats.get('episode_count', 0),
                        'total_reward': getattr(self.rl_controller, 'total_reward', 0.0),
                        'success_rate': stats.get('success_rate', 0.0)
                    })
                    
                    # Trigger AI callback periodically
                    if step_count % 100 == 0:  # Every 100 steps
                        self._trigger_ai_callback(stats)
                        # Update UI displays
                        self._update_ai_status_display(stats)
                
                self._trigger_status_callback(current_state.position, current_state.velocity, mode_info)
                
                # Update visualization periodically
                if time.time() - last_update_time > 0.1:  # 10 Hz update
                    self._update_visualization()
                    # Update system status display
                    current_fps = 1.0 / (time.time() - last_update_time) if last_update_time > 0 else 0.0
                    self._update_system_status_display(fps=current_fps, cpu=0.0)  # CPU calculation can be added later
                    last_update_time = time.time()
                
                # Background validation
                if self.background_validator:
                    self.background_validator.submit_test_event(
                        "Real-time Simulation",
                        "simulation_step",
                        {
                            'time': current_time,
                            'position': current_state.position.tolist(),
                            'velocity': current_state.velocity.tolist(),
                            'control_thrust': control_output.thrust if control_output else 0.0
                        },
                        {'step': step_count}
                    )
                
                step_count += 1
                
                # Sleep to maintain real-time factor (only if real_time_factor > 0)
                if self.sim_params.real_time_factor > 0:
                    sleep_time = self.sim_params.dt / self.sim_params.real_time_factor
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
            except Exception as e:
                self._log_status(f"❌ Simulation error: {e}")
                break
        
        # Cleanup
        if self.keyboard_controller and hasattr(self.keyboard_controller, 'monitoring_thread'):
            if self.keyboard_controller.monitoring_thread and self.keyboard_controller.monitoring_thread.is_alive():
                self.keyboard_controller.stop_input_monitoring()
        if self.background_validator:
            self.background_validator.stop_background_validation()
        if self.test_logger:
            self.test_logger.end_test()
        
        # Save movement log
        if self.movement_logger:
            try:
                log_file = self.movement_logger.save_movement_log()
                if log_file:
                    self._log_status(f"💾 Movement log saved: {log_file}")
            except Exception as e:
                self._log_status(f"⚠️  Could not save movement log: {e}")
    
    def _integrate_physics(self, current_state, control_output, dt):
        """Integrate physics to update drone state based on control output"""
        from ..core.simulator import DroneState
        
        # Create new state based on current state
        new_state = DroneState()
        new_state.position = current_state.position.copy()
        new_state.velocity = current_state.velocity.copy()
        new_state.quaternion = current_state.quaternion.copy()
        new_state.angular_velocity = current_state.angular_velocity.copy()
        
        # Simple physics integration
        mass = self.sim_params.mass
        gravity = self.sim_params.gravity
        
        # STABILIZATION PHYSICS: Add position and velocity damping for realistic hover behavior
        # When no active input, drone should stabilize position and velocity
        position_damping_factor = 8.0  # Much stronger position restoration
        velocity_damping_factor = 0.75  # Strong velocity damping to prevent drift
        angular_damping_factor = 0.8   # Stronger angular velocity damping
        
        # Target position (hover position - current position)
        target_position = np.array(self.sim_params.initial_position)
        position_error = target_position - current_state.position
        
        # FIXED: Handle both thrust and velocity control properly
        if hasattr(control_output, 'desired_velocity') and control_output.desired_velocity is not None:
            # Direct velocity control (from AI velocity commands)
            new_state.velocity = control_output.desired_velocity.copy()
        else:
            # Thrust-based control with stabilization
            # Calculate forces in world frame
            thrust_force = np.array([0, 0, control_output.thrust])  # Thrust in body frame (up)
            gravity_force = np.array([0, 0, -mass * gravity])       # Gravity in world frame (down)
            
            # Add horizontal forces from moments (simplified coupling)
            if hasattr(control_output, 'moment') and control_output.moment is not None:
                # Convert moments to horizontal forces (simplified)
                horizontal_force = np.array([
                    control_output.moment[1] * mass * 0.5,  # Pitch moment -> X force (reduced)
                    -control_output.moment[0] * mass * 0.5, # Roll moment -> Y force (reduced)
                    0
                ])
                thrust_force += horizontal_force
            
            # STABILIZATION FORCES: Add automatic position and velocity stabilization
            # Position restoration force (proportional to distance from target)
            position_restoration_force = position_error * mass * position_damping_factor
            
            # Velocity damping force (opposes current velocity)
            velocity_damping_force = -current_state.velocity * mass * (1.0 - velocity_damping_factor)
            
            # Total force with stabilization
            total_force = thrust_force + gravity_force + position_restoration_force + velocity_damping_force
            
            # Update velocity using F = ma
            acceleration = total_force / mass
            new_state.velocity = current_state.velocity + acceleration * dt
            
            # Apply velocity damping directly (simulate air resistance)
            new_state.velocity *= velocity_damping_factor
        
        # Update position using v = dx/dt
        new_state.position = current_state.position + new_state.velocity * dt
        
        # Update angular velocity using moments with damping
        if hasattr(control_output, 'moment') and control_output.moment is not None:
            # Simple angular dynamics (ignoring inertia tensor for now)
            angular_acceleration = control_output.moment / 0.02  # Simplified inertia
            new_state.angular_velocity = current_state.angular_velocity + angular_acceleration * dt
        
        # Apply angular damping to prevent continuous rotation
        new_state.angular_velocity *= angular_damping_factor
        
        # Simple quaternion integration (for visualization)
        angular_vel_magnitude = np.linalg.norm(new_state.angular_velocity)
        if angular_vel_magnitude > 0.001:
            # For simplicity, just keep the quaternion normalized
            new_state.quaternion = current_state.quaternion / np.linalg.norm(current_state.quaternion)
        
        # Store current control output for visualization
        self.current_control_output = control_output
        
        # COLLISION DETECTION
        collision_detected = self._check_collisions(new_state.position)
        if collision_detected:
            # Reset to initial position and notify controllers
            self._handle_collision(new_state)
        
        return new_state
    
    def _check_collisions(self, position):
        """Check for collisions with obstacles"""
        for obstacle in self.obstacles:
            distance = obstacle.distance_to_point(position)
            if distance < 0.1:  # Collision threshold
                return True
        return False
    
    def _handle_collision(self, state):
        """Handle collision event"""
        # Reset to initial position
        state.position = np.array(self.sim_params.initial_position)
        state.velocity = np.array(self.sim_params.initial_velocity)
        state.angular_velocity = np.zeros(3)
        
        # Log collision
        self._log_status("💥 COLLISION DETECTED! Resetting drone to initial position")
        self._trigger_event_callback("COLLISION", "Drone collision detected, resetting position")
        
        # Notify AI controller if active
        if self.rl_controller and hasattr(self.rl_controller, 'collision_callback') and self.rl_controller.collision_callback:
            self.rl_controller.collision_callback(None)
        
        # Don't end episode - continue training with penalty
    
    def _update_visualization(self):
        """Update the visualization plots"""
        if not MATPLOTLIB_AVAILABLE or not self.canvas:
            return
        
        try:
            # Clear plots
            self.ax_3d.clear()
            self.ax_pos.clear()
            self.ax_vel.clear()
            self.ax_engines.clear()
            self.ax_angles.clear()
            
            if not self.position_history:
                return
            
            # 3D trajectory - show only current episode trajectory (LARGER VISUALIZATION)
            positions = np.array(self.current_episode_positions) if hasattr(self, 'current_episode_positions') and self.current_episode_positions else np.array(self.position_history[-500:])  # Last 500 points max
            self.ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=3, label='Current Episode', alpha=0.8)
            
            # Plot obstacles (larger for visibility)
            for obstacle in self.obstacles:
                pos = obstacle.position
                size = obstacle.size
                # Simple box representation
                self.ax_3d.scatter(pos[0], pos[1], pos[2], c='red', s=200, marker='s', label='Obstacle', alpha=0.7)
            
            # Plot waypoints (larger for visibility)
            for i, waypoint in enumerate(self.waypoints):
                pos = waypoint.position
                color = 'green' if waypoint.reached else 'orange'
                self.ax_3d.scatter(pos[0], pos[1], pos[2], c=color, s=200, marker='^', label=f'Waypoint {i+1}', alpha=0.8)
            
            # Current position (much larger drone representation)
            if positions.shape[0] > 0:
                current_pos = positions[-1]
                self.ax_3d.scatter(current_pos[0], current_pos[1], current_pos[2], c='blue', s=400, marker='o', label='Drone', edgecolors='darkblue', linewidth=2)
                
                # Add drone orientation arrows if available
                if hasattr(self, 'current_control_output') and self.current_control_output:
                    # Show thrust direction
                    thrust_magnitude = self.current_control_output.thrust / 25.0  # Normalize for visualization
                    self.ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2], 
                                     0, 0, thrust_magnitude, color='red', linewidth=3, alpha=0.8, label='Thrust Vector')
            
            self.ax_3d.set_xlabel('X (m)', fontsize=12)
            self.ax_3d.set_ylabel('Y (m)', fontsize=12)
            self.ax_3d.set_zlabel('Z (m)', fontsize=12)
            self.ax_3d.legend(loc='upper left', fontsize=10)
            self.ax_3d.set_title('🚁 3D Drone Trajectory', fontsize=14, fontweight='bold')
            
            # Position vs time - show only current episode (smaller plot)
            current_times = np.array(self.current_episode_times) if hasattr(self, 'current_episode_times') and self.current_episode_times else np.array(self.time_history[-500:])
            current_positions = positions
            
            if len(current_times) > 0 and len(current_positions) > 0:
                self.ax_pos.plot(current_times, current_positions[:, 0], 'r-', label='X', linewidth=2)
                self.ax_pos.plot(current_times, current_positions[:, 1], 'g-', label='Y', linewidth=2) 
                self.ax_pos.plot(current_times, current_positions[:, 2], 'b-', label='Z', linewidth=2)
                self.ax_pos.set_xlabel('Episode Time (s)')
                self.ax_pos.set_ylabel('Position (m)')
                self.ax_pos.legend(fontsize=8)
                self.ax_pos.grid(True)
                self.ax_pos.set_title('Position vs Time', fontsize=10)
            
            # Velocity vs time or AI progress (smaller plot)
            current_velocities = np.array(self.current_episode_velocities) if hasattr(self, 'current_episode_velocities') and self.current_episode_velocities else np.array(self.velocity_history[-500:])
            
            if len(current_times) > 0 and len(current_velocities) > 0:
                if self.simulation_mode in [SimulationMode.AI, SimulationMode.HYBRID] and self.reward_history:
                    # Show AI rewards
                    episodes = range(len(self.reward_history))
                    self.ax_vel.plot(episodes, self.reward_history, 'purple', linewidth=2)
                    self.ax_vel.set_xlabel('Episode')
                    self.ax_vel.set_ylabel('Reward')
                    self.ax_vel.set_title('AI Learning Progress', fontsize=10)
                else:
                    # Show velocities
                    self.ax_vel.plot(current_times, current_velocities[:, 0], 'r-', label='VX', linewidth=2)
                    self.ax_vel.plot(current_times, current_velocities[:, 1], 'g-', label='VY', linewidth=2)
                    self.ax_vel.plot(current_times, current_velocities[:, 2], 'b-', label='VZ', linewidth=2)
                    self.ax_vel.set_xlabel('Episode Time (s)')
                    self.ax_vel.set_ylabel('Velocity (m/s)')
                    self.ax_vel.legend(fontsize=8)
                    self.ax_vel.set_title('Velocity vs Time', fontsize=10)
                self.ax_vel.grid(True)
            
            # ENGINE THRUSTS DISPLAY
            if hasattr(self, 'current_control_output') and self.current_control_output:
                control_output = self.current_control_output
                
                # Calculate individual engine thrusts (simplified quadcopter layout)
                base_thrust = control_output.thrust / 4.0
                differential = 0.0
                
                if hasattr(control_output, 'moment') and control_output.moment is not None:
                    # Roll creates left/right differential
                    differential = control_output.moment[0] * 0.5  # Convert moment to thrust differential
                
                # Engine positions (quadcopter layout)
                engine_positions = np.array([
                    [-0.8, -0.8],  # Front-left
                    [0.8, -0.8],   # Front-right
                    [-0.8, 0.8],   # Rear-left
                    [0.8, 0.8]     # Rear-right
                ])
                
                # Engine thrusts
                engine_thrusts = np.array([
                    base_thrust - differential,  # Front-left
                    base_thrust + differential,  # Front-right
                    base_thrust - differential,  # Rear-left
                    base_thrust + differential   # Rear-right
                ])
                
                # Normalize for visualization (0-1 scale)
                max_thrust = 25.0  # Maximum possible thrust
                normalized_thrusts = np.clip(engine_thrusts / max_thrust, 0, 1)
                
                # Draw drone frame
                self.ax_engines.plot([-1, 1], [0, 0], 'k-', linewidth=3, alpha=0.5)  # X-axis
                self.ax_engines.plot([0, 0], [-1, 1], 'k-', linewidth=3, alpha=0.5)  # Y-axis
                
                # Draw engines with thrust visualization
                colors = ['red', 'green', 'blue', 'orange']
                for i, (pos, thrust, color) in enumerate(zip(engine_positions, normalized_thrusts, colors)):
                    # Engine position
                    circle = plt.Circle(pos, 0.15, color=color, alpha=0.7)
                    self.ax_engines.add_patch(circle)
                    
                    # Thrust vector (length represents thrust magnitude)
                    thrust_length = thrust * 0.5  # Scale for visualization
                    self.ax_engines.arrow(pos[0], pos[1], 0, -thrust_length, 
                                         head_width=0.1, head_length=0.05, fc=color, ec=color, alpha=0.8)
                    
                    # Thrust value text
                    self.ax_engines.text(pos[0], pos[1] + 0.25, f'{engine_thrusts[i]:.1f}N', 
                                        ha='center', va='center', fontsize=8, fontweight='bold')
                
                # Total thrust display
                self.ax_engines.text(0, -1.3, f'Total: {control_output.thrust:.1f}N', 
                                    ha='center', va='center', fontsize=11, fontweight='bold', 
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                self.ax_engines.set_xlim(-1.5, 1.5)
                self.ax_engines.set_ylim(-1.5, 1.5)
                self.ax_engines.set_aspect('equal')
                self.ax_engines.set_xticks([])
                self.ax_engines.set_yticks([])
                self.ax_engines.set_title('🔧 Engine Thrusts', fontsize=12, fontweight='bold')
            
            # DRONE ORIENTATION DISPLAY
            if hasattr(self, 'current_control_output') and self.current_control_output:
                control_output = self.current_control_output
                
                # Calculate angles from moments (simplified)
                roll_angle = 0.0
                pitch_angle = 0.0
                yaw_moment = 0.0
                
                if hasattr(control_output, 'moment') and control_output.moment is not None:
                    # Convert moments to visual angles (in degrees)
                    roll_angle = np.clip(control_output.moment[0] * 30, -45, 45)   # Roll
                    pitch_angle = np.clip(control_output.moment[1] * 30, -45, 45) # Pitch
                    yaw_moment = control_output.moment[2]  # Yaw moment
                
                # Draw drone frame (top view)
                frame_size = 0.8
                self.ax_angles.plot([-frame_size, frame_size], [0, 0], 'k-', linewidth=4, alpha=0.6)
                self.ax_angles.plot([0, 0], [-frame_size, frame_size], 'k-', linewidth=4, alpha=0.6)
                
                # Roll indicator (rotate frame around Z-axis)
                roll_rad = np.radians(roll_angle)
                rotated_x = frame_size * np.cos(roll_rad)
                rotated_y = frame_size * np.sin(roll_rad)
                self.ax_angles.plot([-rotated_x, rotated_x], [-rotated_y, rotated_y], 'r-', linewidth=3, label=f'Roll: {roll_angle:.1f}°')
                
                # Pitch indicator (forward/backward tilt)
                pitch_rad = np.radians(pitch_angle)
                pitch_x = frame_size * np.sin(pitch_rad)
                self.ax_angles.plot([0, pitch_x], [0, frame_size * np.cos(pitch_rad)], 'g-', linewidth=3, label=f'Pitch: {pitch_angle:.1f}°')
                
                # Yaw indicator (rotation direction)
                if abs(yaw_moment) > 0.01:
                    direction = 'CW' if yaw_moment > 0 else 'CCW'
                    self.ax_angles.annotate(f'Yaw: {direction}', xy=(0, -0.9), ha='center', fontsize=10, 
                                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
                # Center point (drone center)
                self.ax_angles.plot(0, 0, 'bo', markersize=12, markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=2)
                
                self.ax_angles.set_xlim(-1.2, 1.2)
                self.ax_angles.set_ylim(-1.2, 1.2)
                self.ax_angles.set_aspect('equal')
                self.ax_angles.set_xticks([])
                self.ax_angles.set_yticks([])
                self.ax_angles.legend(loc='upper right', fontsize=8)
                self.ax_angles.set_title('📐 Drone Orientation', fontsize=12, fontweight='bold')
            
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Visualization update error: {e}")
    
    def _start_ai_training(self):
        """Start AI training mode"""
        if self.rl_controller:
            self.rl_controller.mode = RLMode.TRAINING
            self._log_status("🤖 AI training mode activated")
        else:
            self._log_status("❌ RL controller not available")
    
    def _start_training_session(self):
        """Start a new AI training session"""
        if not self.rl_controller:
            self._log_status("❌ RL controller not available")
            return
        
        if not self.running:
            self._log_status("❌ Please start simulation first")
            return
        
        # Configure episode length based on user input
        episode_length_seconds = self.episode_length_var.get()
        episode_steps = int(episode_length_seconds / self.sim_params.dt)  # Convert seconds to steps
        
        # Update RL controller configuration
        self.rl_controller.config.max_episode_length = episode_steps
        self.rl_controller.config.save_frequency = 10  # Auto-save every 10 episodes
        
        # Set training mode and start
        self.rl_controller.mode = RLMode.TRAINING
        self.training_active = True
        self.training_paused = False
        self.current_episode_start_time = time.time()
        
        # Update UI
        self._update_training_buttons()
        self.training_status_label.config(text="🎯 Training Active", foreground="green")
        
        self._log_status(f"🚀 Training session started - Episodes: {episode_length_seconds}s ({episode_steps} steps)")
        self._log_status(f"📊 Auto-save every 10 episodes")
    
    def _pause_training_session(self):
        """Pause current training session (preserves weights and progress)"""
        if not self.training_active:
            return
        
        self.training_paused = True
        
        # Switch to inference mode to stop learning updates
        if self.rl_controller:
            self.rl_controller.mode = RLMode.INFERENCE
        
        # Save current state
        self._save_training_checkpoint()
        
        # Update UI
        self._update_training_buttons()
        self.training_status_label.config(text="⏸️ Training Paused", foreground="orange")
        
        self._log_status("⏸️ Training paused - progress saved")
    
    def _resume_training_session(self):
        """Resume paused training session"""
        if not self.training_paused:
            return
        
        self.training_paused = False
        
        # Resume training mode
        if self.rl_controller:
            self.rl_controller.mode = RLMode.TRAINING
        
        # Update UI
        self._update_training_buttons()
        self.training_status_label.config(text="🎯 Training Active", foreground="green")
        
        self._log_status("▶️ Training resumed")
    
    def _stop_training_session(self):
        """Stop training session and save final model"""
        if not self.training_active:
            return
        
        self.training_active = False
        self.training_paused = False
        
        # Switch to inference mode
        if self.rl_controller:
            self.rl_controller.mode = RLMode.INFERENCE
            
            # Save final model
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"rl_model_final_{timestamp}.pth"
            self.rl_controller.save_model(filename)
        
        # Update UI
        self._update_training_buttons()
        self.training_status_label.config(text="⏹️ Not Training", foreground="gray")
        
        self._log_status(f"🛑 Training session stopped - final model saved as {filename}")
    
    def _reset_training(self):
        """Reset all training progress (clear weights and statistics)"""
        if self.rl_controller:
            # Reset RL controller state
            self.rl_controller.reset()
            self.rl_controller.episode_count = 0
            self.rl_controller.episode_rewards = []
            self.rl_controller.episode_lengths = []
            self.rl_controller.success_rate = 0.0
            
            # Reinitialize networks if using PyTorch
            if hasattr(self.rl_controller, 'q_network'):
                self.rl_controller.q_network.apply(self._init_weights)
                self.rl_controller.target_network.load_state_dict(
                    self.rl_controller.q_network.state_dict()
                )
        
        # Reset training state
        self.training_active = False
        self.training_paused = False
        
        # Update UI
        self._update_training_buttons()
        self.training_status_label.config(text="🔄 Training Reset", foreground="blue")
        
        self._log_status("🗑️ Training progress reset - starting fresh")
    
    def _reset_drone_position(self):
        """Reset drone to initial position and clear current episode data"""
        if not self.simulator or not self.simulator.state_manager:
            self._log_status("❌ Simulator not initialized")
            return
        
        # Get initial state
        from ..core.simulator import DroneState
        initial_state = DroneState()
        initial_state.position = np.array(self.sim_params.initial_position)
        initial_state.velocity = np.array(self.sim_params.initial_velocity)
        initial_state.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        initial_state.angular_velocity = np.zeros(3)
        
        # Set state
        self.simulator.state_manager.set_state(initial_state)
        
        # Reset current episode tracking (clear trajectory visualization)
        self.current_episode_positions = []
        self.current_episode_velocities = []
        self.current_episode_times = []
        self.current_episode_reward = 0.0
        self.episode_start_time = len(self.time_history) * self.sim_params.dt if self.time_history else 0.0
        
        # Clear command buffer for fresh start
        self.command_buffer = []
        self.last_executed_command = None
        
        # Reset RL controller episode if active
        if self.rl_controller:
            self.rl_controller.total_reward = 0.0
            self.rl_controller.step_count = 0
            # Reset waypoint tracking
            self.rl_controller.current_waypoint_idx = 0
            for waypoint in self.rl_controller.waypoints:
                waypoint.reached = False
        
        self._log_status("🔄 Drone position reset - New episode started")
        self._trigger_event_callback("RESET", "Drone position reset to initial state")
    
    def _save_training_checkpoint(self):
        """Save training checkpoint"""
        if self.rl_controller:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"rl_checkpoint_{timestamp}.pth"
            self.rl_controller.save_model(filename)
            self._log_status(f"💾 Training checkpoint saved: {filename}")
    
    def _update_training_buttons(self):
        """Update training button states based on current training status"""
        if not TKINTER_AVAILABLE:
            return
        
        if self.training_active and not self.training_paused:
            # Training is active
            self.start_training_button.config(state=tk.DISABLED)
            self.pause_training_button.config(state=tk.NORMAL)
            self.resume_training_button.config(state=tk.DISABLED)
            self.stop_training_button.config(state=tk.NORMAL)
        elif self.training_paused:
            # Training is paused
            self.start_training_button.config(state=tk.DISABLED)
            self.pause_training_button.config(state=tk.DISABLED)
            self.resume_training_button.config(state=tk.NORMAL)
            self.stop_training_button.config(state=tk.NORMAL)
        else:
            # Training is not active
            self.start_training_button.config(state=tk.NORMAL)
            self.pause_training_button.config(state=tk.DISABLED)
            self.resume_training_button.config(state=tk.DISABLED)
            self.stop_training_button.config(state=tk.DISABLED)
    
    def _init_weights(self, m):
        """Initialize neural network weights"""
        if TORCH_AVAILABLE and isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def _load_ai_model(self):
        """Load AI model from file"""
        if not TKINTER_AVAILABLE or not self.rl_controller:
            return
        
        filename = filedialog.askopenfilename(
            title="Load AI Model",
            filetypes=[("Model files", "*.pth *.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.rl_controller.load_model(filename)
                self._log_status(f"📂 AI model loaded from {filename}")
            except Exception as e:
                self._log_status(f"❌ Failed to load model: {e}")
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def _save_ai_model(self):
        """Save AI model to file"""
        if not TKINTER_AVAILABLE or not self.rl_controller:
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save AI Model",
            defaultextension=".pth",
            filetypes=[("PyTorch models", "*.pth"), ("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.rl_controller.save_model(filename)
                self._log_status(f"💾 AI model saved to {filename}")
            except Exception as e:
                self._log_status(f"❌ Failed to save model: {e}")
                messagebox.showerror("Error", f"Failed to save model: {e}")
    
    def _log_status(self, message: str):
        """Log status message"""
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        if TKINTER_AVAILABLE and self.status_text:
            self.status_text.insert(tk.END, full_message + "\n")
            self.status_text.see(tk.END)
        else:
            print(full_message)
        
        # Trigger event callback for log messages instead of status callback
        if hasattr(self, 'event_callback') and self.event_callback:
            # Extract event type from message emojis
            if "🚀" in message or "START" in message:
                event_type = "START"
            elif "🛑" in message or "STOP" in message:
                event_type = "STOP"
            elif "❌" in message or "ERROR" in message:
                event_type = "ERROR"
            elif "⏸️" in message or "PAUSE" in message:
                event_type = "PAUSE"
            elif "▶️" in message or "RESUME" in message:
                event_type = "RESUME"
            else:
                event_type = "INFO"
            
            try:
                self.event_callback(event_type, message)
            except Exception:
                pass  # Don't let callback errors break the logging
    
    def run(self):
        """Run the interface"""
        if TKINTER_AVAILABLE and self.root:
            self._log_status("🎮 Starting real-time interface")
            self.root.mainloop()
        else:
            self._run_console_mode()
    
    def _run_console_mode(self):
        """Run in console mode"""
        print("🎮 Console mode active. Type 'help' for commands.")
        
        while True:
            try:
                command = input("drone_sim> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "start":
                    self._start_simulation()
                elif command == "stop":
                    self._stop_simulation()
                elif command.startswith("mode "):
                    mode = command.split()[1]
                    if mode in ["manual", "ai", "hybrid"]:
                        self.simulation_mode = SimulationMode(mode)
                        print(f"Mode set to: {mode}")
                    else:
                        print("Invalid mode. Use: manual, ai, or hybrid")
                elif command == "help":
                    print("Available commands:")
                    print("  start - Start simulation")
                    print("  stop  - Stop simulation")
                    print("  mode <manual|ai|hybrid> - Change mode")
                    print("  quit  - Exit")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        self._stop_simulation()
        print("👋 Goodbye!")

    def _update_ai_status_display(self, stats):
        """Update AI training status display in UI"""
        if not TKINTER_AVAILABLE or not hasattr(self, 'episode_label'):
            return
        
        try:
            # Update episode info
            self.episode_label.config(text=str(stats.get('episode_count', 0)))
            self.reward_label.config(text=f"{stats.get('average_reward', 0.0):.2f}")
            
            # Update success rate and exploration
            success_rate = stats.get('success_rate', 0.0) * 100
            self.success_rate_label.config(text=f"{success_rate:.1f}%")
            
            exploration = stats.get('current_epsilon', 1.0) * 100
            self.exploration_label.config(text=f"{exploration:.1f}%")
            
            # Update waypoint progress
            current_wp = stats.get('waypoints_completed', 0)
            total_wp = stats.get('total_waypoints', 0)
            self.waypoint_label.config(text=f"{current_wp}/{total_wp}")
            
            # Update collision count
            collisions = stats.get('collision_count', 0)
            self.collision_label.config(text=str(collisions))
            
            # Update progress bars
            if total_wp > 0:
                waypoint_progress = (current_wp / total_wp) * 100
                self.episode_progress['value'] = waypoint_progress
            
            # Training progress based on episodes (assuming 1000 episodes target)
            episode_count = stats.get('episode_count', 0)
            training_progress = min((episode_count / 1000) * 100, 100)
            self.training_progress['value'] = training_progress
            
        except Exception as e:
            print(f"Error updating AI status display: {e}")
    
    def _update_system_status_display(self, fps=0.0, cpu=0.0):
        """Update system process status display in UI"""
        if not TKINTER_AVAILABLE or not hasattr(self, 'fps_label'):
            return
        
        try:
            # Update performance metrics
            self.fps_label.config(text=f"{fps:.1f}")
            self.cpu_label.config(text=f"{cpu:.1f}%")
            
            # Update process status indicators
            if self.background_validator:
                self.physics_status_label.config(text="✅ Active", foreground="green")
            else:
                self.physics_status_label.config(text="❌ Inactive", foreground="red")
            
            if self.test_logger:
                self.logging_status_label.config(text="✅ Active", foreground="green")
            else:
                self.logging_status_label.config(text="❌ Inactive", foreground="red")
                
        except Exception as e:
            print(f"Error updating system status display: {e}")

def main():
    """Main entry point"""
    interface = RealTimeInterface()
    interface.run()

if __name__ == "__main__":
    main() 