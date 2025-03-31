import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple, Union
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm
from environment.integrator import ChemicalIntegrator, IntegratorConfig
from environment.combustion_problem import CombustionProblem, setup_problem, CombustionStage
import cantera as ct

class CombustionEnv(gym.Env):
    """Reinforcement learning environment for chemical kinetics integration."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self,
                 problem: 'CombustionProblem',
                 integrator: 'ChemicalIntegrator',
                 features_config: Optional[Dict] = None,
                 reward_weights: Optional[Dict] = None,
                 render_mode: Optional[str] = 'human'):
        """
        Initialize environment.
        
        Args:
            problem: CombustionProblem instance
            integrator: ChemicalIntegrator instance
            features_config: Feature calculation configuration
            reward_weights: Reward calculation weights
            render_mode: Mode for rendering ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.problem = problem
        self.integrator = integrator
        self.render_mode = render_mode
        
        # Validate and set default configurations
        self.features_config = self._validate_features_config(features_config)
        self.reward_weights = self._validate_reward_weights(reward_weights)
        
        # Extract parameters
        self.num_steps = problem.completed_steps
        self.species_to_track = problem.species_to_track
        self.timestep = problem.timestep
        self.window_size = self.features_config['window_size']
        
        # Initialize spaces
        self.action_space = spaces.Discrete(len(integrator.action_list))
        self._setup_observation_space()
        
        # Initialize feature buffers
        self._init_history_buffer()
        self.reset()
        
        
    def _validate_features_config(self, config: Optional[Dict]) -> Dict:
        """Validate and set default feature configuration."""
        default_config = {
            'temporal_features': True,
            'species_features': False,
            'basic_features': True,
            'include_time_step': True,
            'window_size': 5,
            'include_stage': True,
            'epsilon': 1e-6,
            'Etol': 1e-6,
            'time_threshold': 1e-3
        }
        
        if config is None:
            return default_config
            
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                
        return config
        
    def _validate_reward_weights(self, weights: Optional[Dict]) -> Dict:
        """Validate and set default reward weights."""
        default_weights = {
            'alpha': 0.8,  # time weight
            'beta': 0.2,   # error weight
            'gamma': 0.1,  # stability weight
            'delta': 0.05  # stage transition bonus
        }
        
        if weights is None:
            return default_weights
            
        for key in default_weights:
            if key not in weights:
                weights[key] = default_weights[key]
                
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _init_history_buffer(self):
        """Initialize history buffers."""
        self.history_buffer = {
            'temperature': deque(maxlen=self.window_size),
            'gradients': deque(maxlen=self.window_size),
            'species': {spec: deque(maxlen=self.window_size) 
                       for spec in self.species_to_track},
            'rewards': deque(maxlen=100),  # Track last 100 rewards
            'actions': deque(maxlen=self.window_size)
        }
        
    def _setup_observation_space(self):
        """Set up observation space based on enabled features."""
        obs_size = 0
        
        # Calculate observation size based on enabled features
        if self.features_config['include_time_step']:
            obs_size += 1
            
        if self.features_config['basic_features']:
            obs_size += len(self.species_to_track) + 1  # species + temperature
            
        if self.features_config['temporal_features']:
            obs_size += 4  # [max_rate, mean_rate, rate_variability, acceleration]
            
        if self.features_config['species_features']:
            obs_size += len(self.species_to_track) * 3  # [max_rate, mean_rate, variability] per species
            
        # Add current stage information
        if self.features_config['include_stage']:
            obs_size += 3  # One-hot encoding of current stage
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        self.integrator.reset_history()
        self._init_history_buffer()
        
        # Store initial conditions
        self.initial_state = self.integrator.y.copy()
        
        # Get initial observation
        observation = self._get_observation()
        
         # Initialize episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_errors = []
        self.episode_times = []
        self.episode_time_rewards = []
        self.episode_error_rewards = []
        
        # Track best performance
        self.best_episode_reward = float('-inf')
        self.best_episode_stats = None
        
        info = {
            'initial_temperature': self.integrator.y[0],
            'initial_species': {spec: self.integrator.y[self.integrator.gas.species_index(spec) + 1]
                              for spec in self.species_to_track}
        }
        
        self.action_distribution = {action: 0 for action in range(self.action_space.n)}
        self.action_history = []
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state and history.
        
        Returns:
            np.ndarray: Observation vector containing all enabled features
        """
        observation_parts = []
        
        # Basic features
        if self.features_config['basic_features']:
            basic_features = self._get_basic_features()
            observation_parts.append(basic_features)
        
        # Time step feature
        if self.features_config['include_time_step']:
            time_feature = self._get_time_feature()
            observation_parts.append(time_feature)
        
        # Temporal features
        if self.features_config['temporal_features']:
            temporal_features = self._get_temporal_features()
            observation_parts.append(temporal_features)
        
        # Species features
        if self.features_config['species_features']:
            species_features = self._get_species_features()
            observation_parts.append(species_features)
        
        # Add current stage information (one-hot encoded)
        if self.features_config['include_stage']:
            stage_feature = np.zeros(3)  # 3 stages: PREIGNITION, IGNITION, POSTIGNITION
            stage_feature[self.integrator.current_stage.value] = 1
            observation_parts.append(stage_feature)
        
        # Combine all features
        observation = np.concatenate(observation_parts)
        return observation.astype(np.float32)

    def _get_basic_features(self) -> np.ndarray:
        """Get basic state features."""
        # Temperature normalization
        T = self.integrator.y[0]
        T_normalized = np.log10(T)/3.0 
        
        # Pressure normalization
        P = self.problem.pressure
        P_normalized = P / ct.one_atm # normalized to 1 atm
        
        # Phi normalization
        phi = self.problem.phi
        phi_normalized = phi # normalized to 1
        
        
        # Species mass fractions (log-transformed)
        Y = np.array([
            self.integrator.y[self.integrator.gas.species_index(spec) + 1]
            for spec in self.species_to_track
        ])
        Y_normalized = np.clip(Y, 1e-20, None)
        Y_normalized = np.log10(Y_normalized) / 20 
        
        return np.concatenate([[T_normalized], Y_normalized])

    def _get_time_feature(self) -> np.ndarray:
        """Get time-related feature."""
        return np.array([-np.log10(self.timestep)])

    def _get_temporal_features(self) -> np.ndarray:
        """Get temporal evolution features."""
        if len(self.history_buffer['temperature']) < 2:
            return np.zeros(4)
        
        temp_array = np.array(list(self.history_buffer['temperature']))
        dT_dt = np.gradient(temp_array) / self.timestep
        d2T_dt2 = np.gradient(dT_dt) / self.timestep
        
        return np.array([
            np.log1p(np.max(np.abs(dT_dt))),
            np.log1p(np.mean(np.abs(dT_dt))),
            np.log1p(np.std(dT_dt)),
            np.log1p(np.mean(np.abs(d2T_dt2)))
        ])

    def _get_species_features(self) -> np.ndarray:
        """Get species evolution features."""
        features = []
        for spec in self.species_to_track:
            if len(self.history_buffer['species'][spec]) < 2:
                features.extend([0.0, 0.0, 0.0])
            else:
                spec_array = np.array(list(self.history_buffer['species'][spec]))
                dY_dt = np.gradient(spec_array) / self.timestep
                
                features.extend([
                    np.log1p(np.max(np.abs(dY_dt))),
                    np.log1p(np.mean(np.abs(dY_dt))),
                    np.log1p(np.std(dY_dt))
                ])
        return np.array(features)
    
    def step(self, action: int, timeout: Optional[float] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Integration method index
            
        Returns:
            observation: New observation
            reward: Reward value
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        #print(f"Step {self.current_step} - Action {action} - Integrator Step {self.integrator.step_count}")
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
            
        self.action_distribution[int(action)] += 1
        self.action_history.append(action)
        # Store current state for reward calculation
        previous_stage = self.integrator.current_stage
        # Perform integration step
        # print(f"Integrating with action {action} and step {self.current_step}")
        result = self.integrator.integrate_step(action, timeout=timeout)
        
        #print(f"Integrated with action {action} and step {self.current_step} - timed out: {result['timed_out']} - cpu time: {result['cpu_time']}")
        
        if result['timed_out']:
            print(f"[WARNING] Integration timed out after {result['cpu_time']} seconds with action {action} and step {self.current_step}")
            return (
                self._get_observation(),
                self._compute_failure_reward(),
                True,
                False,
                {
                    'error': 0,
                    'cpu_time': result['cpu_time'],
                    'message': result['message'],
                    'termination_reason': 'timed_out',
                    'timed_out': True
                }
            )
        
        if not result['success'] and result['message'] != 'Integration failed':
            print(f"[WARNING] Integration failed after {result['cpu_time']} seconds with action {action} and step {self.current_step}")
            return (
                self._get_observation(),
                self._compute_failure_reward(),
                True,
                False,
                {
                    'error': 0,
                    'cpu_time': result['cpu_time'],
                    'message': result['message'],
                    'termination_reason': 'integration_failure',
                    'timed_out': False
                }
            )
            
        
        
        # Update state
        
        self._update_history_buffer(result['y'], action)
        
        # Compute reward including stage transition bonus
        reward, time_reward, error_reward = self._compute_reward(
            result['cpu_time'], 
            result['error'],
        )
        

        # Check termination conditions
        self.current_step += 1
        terminated = self._check_termination(result)
        truncated = self.current_step >= self.problem.completed_steps
        
        # if terminated or truncated:
        #     print(f"Terminated or truncated at step {self.current_step}")
        
        
        #print(f"Integrator Step {self.integrator.step_count} - Step {self.current_step} - Reward: {reward}, Time: {result['cpu_time']}, Error: {result['error']} - Terminated: {terminated}, Truncated: {truncated}")
        
        # Update episode statistics
        self.episode_rewards.append(reward)
        self.episode_errors.append(result['error'])
        self.episode_times.append(result['cpu_time'])
        self.episode_time_rewards.append(time_reward)
        self.episode_error_rewards.append(error_reward)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_step_info(result, terminated or truncated)
        
        return observation, reward, terminated, truncated, info
    
    def _update_history_buffer(self, y: 
        np.ndarray, action: int):
        """
        Update history buffers with new state.
        
        Args:
            y: New state vector
            action: Action taken
        """
        # Store temperature
        T = y[0]
        self.history_buffer['temperature'].append(T)
        
        # Calculate and store temperature gradient
        if len(self.history_buffer['temperature']) > 1:
            dT_dt = ((self.history_buffer['temperature'][-1] - 
                      self.history_buffer['temperature'][-2]) / self.timestep)
            self.history_buffer['gradients'].append(dT_dt)
        
        # Store species concentrations
        for spec in self.species_to_track:
            idx = self.integrator.gas.species_index(spec)
            self.history_buffer['species'][spec].append(y[idx + 1])
        
        # Store action
        self.history_buffer['actions'].append(action)
    
    def _compute_failure_reward(self) -> float:
        """Compute reward for integration failure."""
        return -50.0  # Significant penalty for integration failure
    
    def _check_termination(self, result: Dict) -> bool:
        """Check if episode should terminate."""
        # Terminate if integration failed
        if not result['success'] and result['message'] != 'Integration failed':
            return True
            
        # Terminate if solution diverged
        if np.any(np.isnan(self.integrator.y)) or np.any(np.isinf(self.integrator.y)):
            return True
        
        if result['timed_out']:
            return True
        
        # Terminate if end simulation flag is set
        return result.get('end_simulation', False)
    
    def _compute_reward(self, cpu_time, error):
        """
        Compute reward prioritizing:
        1. Time efficiency as primary factor (linear scaling)
        2. Error only matters when below Etol
        3. Heightened importance during ignition/post-ignition stages
        
        Args:
            cpu_time: Computation time for the step
            error: Estimated local error
            previous_error: Error from previous step (if available)
            epsilon: Small number to prevent division by zero
            Etol: Error tolerance
            time_threshold: Target computation time
            stage: Current combustion stage
        """
        
        # 1. Linear time reward that heavily rewards speed
        # Scaled to be between -10 and 1
        normalized_time = cpu_time / self.features_config['time_threshold']
        time_reward = max(-5, 1 - 2 * normalized_time)
        
        # 2. Simple error reward that only matters when below Etol
        error_ratio = error / (self.features_config['Etol'] + self.features_config['epsilon'])
        if error_ratio <= 1.0:  # Below tolerance
            # Linear scaling from 1 (best) to 0 (at tolerance)
            error_reward = 1.0
        else:  # Above tolerance
            # Constant minimum value for being above tolerance
            error_reward = -np.log10(error_ratio + self.features_config['epsilon'])
        
        # 3. Stage-based scaling focusing on critical phases
        # Especially amplify time reward during ignition/post-ignition
        if self.integrator.current_stage == CombustionStage.IGNITION:
            time_multiplier = 2.0    # Double importance of time during ignition
            error_multiplier = 1.2   # Slightly higher accuracy importance
        elif self.integrator.current_stage == CombustionStage.POSTIGNITION:
            time_multiplier = 1.5    # 50% more importance for time
            error_multiplier = 1.0   # Normal accuracy importance
        else:  # PREIGNITION
            time_multiplier = 1.0    # Normal time importance
            error_multiplier = 1.0   # Normal accuracy importance
        
        # Base weights heavily favoring time
        w_time = self.reward_weights['alpha']  # 80% weight on time
        w_error = self.reward_weights['beta']  # 20% weight on error
        
        # Apply stage multipliers
        final_time_reward = time_reward * time_multiplier
        final_error_reward = error_reward * error_multiplier
        
        # Compute final reward
        reward = w_time * final_time_reward + w_error * final_error_reward
        
        tanh_reward = np.tanh(reward)
        self.reward_components = {
            'time_reward': time_reward,
            'error_reward': error_reward,
            'final_time_reward': final_time_reward,
            'final_error_reward': final_error_reward,
            'time_multiplier': time_multiplier,
            'error_multiplier': error_multiplier,
            'final_reward': reward,
            'tanh_reward': tanh_reward
        }
        
        return reward, time_reward, error_reward
    # def _compute_reward(self, cpu_time, error):
    #     """Compute reward based on CPU time and error"""
    #     epsilon = self.features_config['epsilon']
    
    #     # Base time reward
    #     time_reward = 1 / (1 + cpu_time / self.features_config['time_threshold'])
    #     # Error reward with dynamic scaling
    #     error_ratio = error / (self.features_config['Etol'] + epsilon)
        
    #     error_reward = -np.log1p(error_ratio) / np.log1p(100) # the 
    

    #     # Dynamic weight adjustment
    #     w_time = self.reward_weights['alpha']
    #     w_error = self.reward_weights['beta']
        

    #     # Combined reward
    #     reward = w_time * time_reward + w_error * error_reward
    #     #print(f"time_reward: {time_reward}, error_reward: {error_reward}, reward: {reward}, error: {error}, error_ratio: {error_ratio}, cpu_time: {cpu_time}")
        
    #     return reward, time_reward, error_reward
    
    
    def _get_step_info(self, result: Dict, is_terminal: bool) -> Dict:
        """Get step information dictionary."""
        info = {
            'cpu_time': result['cpu_time'],
            'error': result['error'],
            'current_stage': self.integrator.current_stage.name,
            'step': self.current_step,
            'action_history': list(self.history_buffer['actions']),
            'time_reward': self.episode_time_rewards[-1],
            'error_reward': self.episode_error_rewards[-1],
            'cummulative_reward': sum(self.episode_rewards),
            'cummulative_cpu_time': sum(self.episode_times),
            'cummulative_error': sum(self.episode_errors),
            'timed_out': False,
            'action_distribution': self.action_distribution
        }
        
        if is_terminal:
            # Add episode statistics
            episode_stats = self.integrator.get_statistics()
            info.update({
                'episode_total_reward': sum(self.episode_rewards),
                'episode_mean_reward': np.mean(self.episode_rewards),
                'episode_mean_error': np.mean(self.episode_errors),
                'episode_mean_time': np.mean(self.episode_times),
                'episode_statistics': episode_stats
            })
            
            # Update best episode if applicable
            total_reward = sum(self.episode_rewards)
            if total_reward > self.best_episode_reward:
                self.best_episode_reward = total_reward
                self.best_episode_stats = episode_stats
                info['new_best_episode'] = True
        
        return info
    
    def render(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Render the current state.
        
        Returns:
            np.ndarray: RGB array if render_mode is 'rgb_array'
        """
        if self.render_mode is None:
            return None
            
        if self.render_mode == "rgb_array":
            # Return RGB array of plot
            self.integrator.plot_history(save_path)
            return np.array(plt.gcf().canvas.renderer.buffer_rgba())
            
        elif self.render_mode == "human":
            self.integrator.plot_history(save_path)
            plt.show()
    
    def close(self):
        """Clean up environment resources."""
        plt.close('all')