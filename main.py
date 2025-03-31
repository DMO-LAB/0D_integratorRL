"""
Reorganized and optimized code for RL Integrator Selector
Contains modular components for data collection, training, evaluation and visualization
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse
import json
from datetime import datetime
import numpy as np
import cantera as ct
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from functools import partial
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rl_integrator")

# Import local modules (assuming these exist)
from config.default_config import Args
from agents.ppo_ import PPO
from environment.env_wrapper import EnvManager

#-------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------

@dataclass
class CollectionConfig:
    """Configuration for data collection process."""
    num_episodes: int = 200
    distance_threshold: float = 0.001
    metric: str = 'reward'
    timeout: float = 5
    verbose: bool = True

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    patience: int = 7
    test_size: float = 0.2

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    num_episodes: int = 10
    render: bool = False
    timeout: Optional[float] = None

@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_path: Optional[str] = None
    output_dir: str = 'results'
    save_model: bool = True
    mech_file: str = ''
    fuel: str = ''

#-------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------

class ExperimentTracker:
    """Manages experiment output directories and result tracking."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment tracker with given configuration."""
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{config.output_dir}/{self.timestamp}"
        self.model_dir = os.path.join(self.output_dir, "models")
        self.plot_dir = os.path.join(self.output_dir, "plots")
        self.data_dir = os.path.join(self.output_dir, "data")
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Results will be saved to: {self.output_dir}")
        
        # Save configuration
        self.save_config()
    
    def save_config(self):
        """Save configuration to JSON file."""
        config_dict = {}
        for category in ['collection', 'training', 'evaluation']:
            config_dict[category] = {}
            category_config = getattr(self.config, category)
            for key, value in category_config.__dict__.items():
                config_dict[category][key] = value
        
        config_dict['data_path'] = self.config.data_path
        config_dict['output_dir'] = self.config.output_dir
        config_dict['save_model'] = self.config.save_model
        
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def save_dict(self, data: Dict, filename: str):
        """Save dictionary data as JSON with proper serialization of numpy types."""
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.number):
                return float(obj)
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(item) for item in obj]
            return obj
        
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(serialize(data), f, indent=4)
    
    def save_numpy(self, data: np.ndarray, filename: str):
        """Save numpy array to the data directory."""
        path = os.path.join(self.data_dir, filename)
        np.save(path, data)
        logger.info(f"Data saved to {path}")
        return path
    
    def save_model(self, model_state_dict, filename: str):
        """Save model state dictionary."""
        path = os.path.join(self.model_dir, filename)
        torch.save(model_state_dict, path)
        logger.info(f"Model saved to {path}")
        return path
    
    def save_plot(self, fig, filename: str):
        """Save matplotlib figure."""
        path = os.path.join(self.plot_dir, filename)
        fig.savefig(path)
        logger.info(f"Plot saved to {path}")
        return path


class ChemicalUtils:
    """Utilities for working with chemical mechanisms and compositions."""
    
    @staticmethod
    def parse_composition(gas, X):
        """Create a dictionary of species:fraction pairs from the gas species names and X."""
        species_names = gas.species_names
        composition = {}
        for i, species in enumerate(species_names):
            composition[species] = X[i]
        return composition

    @staticmethod
    def load_chemical_data(data_path: str):
        """Load chemical data from file."""
        data = np.load(data_path)
        phis = data['phi'].reshape(-1, 1)
        Ts = data['T'].reshape(-1, 1)
        Ps = data['P'].reshape(-1, 1)
        Ys = data['Y']
        Zs = data.get('Z', np.zeros(len(Ts))).reshape(-1, 1)
        
        # Combine the phi, T, P, Z, Y into a single array
        initial_conditions = np.concatenate([phis, Ts, Ps, Zs, Ys], axis=1)

        # Create boolean mask for valid phi values
        phi_mask = np.logical_and(phis >= 0.0, phis <= 1000)
        phi_mask = phi_mask.flatten()  # Convert from (100000, 1) to (100000,)

        # Apply mask to filter initial conditions
        initial_conditions_filtered = initial_conditions[phi_mask]
        return initial_conditions_filtered

#-------------------------------------------------------------------------
# Data Collection
#-------------------------------------------------------------------------

class FilteringStrategy:
    """Strategy for filtering observations based on similarity."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize filtering strategy.
        
        Args:
            threshold: Distance threshold for filtering
        """
        self.threshold = threshold
        self.buffer = np.empty((0, 0))
    
    def reset(self, obs_dim: int):
        """Reset the filtering buffer."""
        self.buffer = np.empty((0, obs_dim))
    
    def should_keep(self, observation: np.ndarray) -> bool:
        """
        Determine if observation should be kept based on distance from buffered observations.
        
        Args:
            observation: The observation to evaluate
            
        Returns:
            bool: True if observation should be kept
        """
        obs_reshaped = observation.reshape(1, -1)
        
        if len(self.buffer) == 0:
            # First observation is always kept
            self.buffer = obs_reshaped
            return True
        
        # Calculate distances to previous observations
        distances = cdist(obs_reshaped, self.buffer, 'euclidean')[0]
        
        # Keep if minimum distance exceeds threshold
        if np.min(distances) >= self.threshold:
            self.buffer = np.vstack((self.buffer, obs_reshaped))
            return True
        
        return False


class IntegratorDataCollector:
    """Collects optimal integrator data by comparing integrators at each timestep."""
    
    def __init__(self, env_manager: EnvManager, config: CollectionConfig):
        """
        Initialize data collector.
        
        Args:
            env_manager: Manager for combustion environments
            config: Collection configuration
        """
        self.env_manager = env_manager
        self.config = config
        self.filtering_strategy = FilteringStrategy(threshold=config.distance_threshold)
        
        # Map metrics to their indices in the history array
        self.metric_indices = {
            'reward': 0,
            'error': 1,
            'cpu_time': 2,
            'time_reward': 3,
            'error_reward': 4,
        }
        
        if config.metric not in self.metric_indices:
            raise ValueError(f"Unknown metric: {config.metric}")
        
        self.metric_idx = self.metric_indices[config.metric]
        self.error_idx = self.metric_indices['error']
        self.minimize_metrics = ['error', 'cpu_time']
    
    def _run_with_all_integrators(self, env_dict, action_dim, seed=None, timeout=None):
        """
        Run a single step with all integrators and collect results.
        
        Args:
            env_dict: Dictionary of environments, one per integrator
            action_dim: Number of actions/integrators
            seed: Random seed for reproducibility
            timeout: Optional timeout for integration steps
            
        Returns:
            dict: Results for each action
            np.ndarray: Current observation
            bool: Whether any environment is done
        """
        timestep_results = {}
        current_obs = None
        any_done = False
        
        # Run one step with each integrator
        for action in range(action_dim):
            try:
                next_obs, reward, terminated, truncated, info = env_dict[action].step(
                    action, timeout=timeout
                )
                action_done = terminated or truncated
                
                # Collect metrics for this action
                timestep_results[action] = {
                    'obs': next_obs,
                    'reward': reward,
                    'error': info.get('error', float('inf')),
                    'cpu_time': info.get('cpu_time', float('inf')),
                    'time_reward': info.get('time_reward', -float('inf')),
                    'error_reward': info.get('error_reward', -float('inf')),
                    'done': action_done,
                    'current_stage': info.get('current_stage', 'UNKNOWN')
                }
                
                # Use any observation as current (they should be same state)
                current_obs = next_obs.copy()
                
                # Check if any environment is done
                if action_done:
                    any_done = True
                    
            except Exception as e:
                if self.config.verbose:
                    logger.error(f"Error with action {action}: {str(e)}")
                # Mark this action as invalid for this timestep
                timestep_results[action] = {
                    'obs': current_obs if current_obs is not None else None,
                    'reward': -float('inf'),
                    'error': float('inf'),
                    'cpu_time': float('inf'),
                    'time_reward': -float('inf'),
                    'error_reward': -float('inf'),
                    'done': True,
                    'current_stage': 'ERROR'
                }
                any_done = True
        
        return timestep_results, current_obs, any_done
    
    def _select_best_action(self, timestep_results):
        """
        Select best action based on chosen metric.
        
        Args:
            timestep_results: Results from running all integrators
            
        Returns:
            int: Best action
            dict: Result for the best action
        """
        metric = self.config.metric
        best_action = None
        best_value = float('inf') if metric in self.minimize_metrics else -float('inf')
        
        for action, result in timestep_results.items():
            metric_value = result[metric]
            error_value = result['error']
            
            # Skip invalid actions
            if (metric in self.minimize_metrics and metric_value == float('inf')) or \
               (metric not in self.minimize_metrics and metric_value == -float('inf')):
                continue
                
            # Check if better
            if metric in self.minimize_metrics:
                if metric_value < best_value:
                    best_value = metric_value
                    best_action = action
                # Tie-breaking using error when values are close
                elif abs(metric_value - best_value) / (abs(best_value) + 1e-10) < 0.05:
                    if error_value < timestep_results[best_action]['error']:
                        best_action = action
            else:  # Maximize
                if metric_value > best_value:
                    best_value = metric_value
                    best_action = action
                # Tie-breaking using error when values are close
                elif abs(metric_value - best_value) / (abs(best_value) + 1e-10) < 0.05:
                    if error_value < timestep_results[best_action]['error']:
                        best_action = action
        
        return best_action, timestep_results.get(best_action) if best_action is not None else None
    
    def _collect_single_episode(self, initial_condition=None, seed=None):
        """
        Collect data for a single episode.
        
        Args:
            initial_condition: Optional initial condition to use
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Combined history for this episode
            dict: Episode statistics
        """
        episode_start = time.time()
        
        # Generate seed for reproducibility
        if seed is None:
            seed = int(time.time() * 1000) % 10000
        
        # Select random initial condition if not provided
        if initial_condition is None:
            data_path = '/home/elo/CODES/SCI-ML/ember/initial_conditions_uniform_z_100.npz'
            initial_conditions = ChemicalUtils.load_chemical_data(data_path)
            initial_condition = initial_conditions[np.random.randint(0, len(initial_conditions))]
        
        # Extract initial condition parameters
        phi = initial_condition[0]
        T = initial_condition[1]
        P = initial_condition[2]
        Z = initial_condition[3]
        X = initial_condition[4:]
        
        # Get gas object and parse composition
        mech_file = '/home/elo/CODES/SCI-ML/RLIntegratorSelector/large_mechanism/large_mechanism/n-dodecane.yaml'
        gas = ct.Solution(mech_file)
        composition = ChemicalUtils.parse_composition(gas, X)
        
        # Get parameters for this problem
        fixed_temperature = T
        fixed_pressure = P/101325
        fixed_phi = phi
        initial_mixture = composition
        
        # Select time step based on temperature
        fixed_dt = np.random.choice(
            self.env_manager.args.min_time_steps_range if fixed_temperature > 1000 
            else self.env_manager.args.max_time_steps_range
        )
        
        if self.config.verbose:
            logger.info(f"Episode parameters: T={fixed_temperature}K, P={fixed_pressure}atm, " 
                        f"phi={fixed_phi}, dt={fixed_dt}s, Z={Z}")
        
        # Create temporary environment to get action space
        temp_env = self.env_manager.create_single_env(
            end_time=self.env_manager.args.end_time,
            fixed_temperature=fixed_temperature,
            fixed_pressure=fixed_pressure,
            fixed_phi=fixed_phi,
            fixed_dt=fixed_dt,
            initial_mixture=initial_mixture,
            randomize=False
        )
        action_dim = temp_env.action_space.n
        temp_env.close()
        
        # Create one environment per integrator with same parameters
        envs = {}
        for action in range(action_dim):
            envs[action] = self.env_manager.create_single_env(
                end_time=self.env_manager.args.end_time,
                fixed_temperature=fixed_temperature,
                fixed_pressure=fixed_pressure,
                fixed_phi=fixed_phi,
                fixed_dt=fixed_dt,
                initial_mixture=initial_mixture,
                randomize=False
            )
            # Reset environment
            obs, _ = envs[action].reset(seed=seed)
        
        # Dataset structure variables
        observations = []
        action_values = []  # Stores reward, error, etc. for each action
        selected_actions = []
        stage_counts = {}
        
        # Reset filtering strategy
        self.filtering_strategy.reset(obs_dim=len(obs))
        
        # Run episode loop
        max_steps = 1000  # Safety limit
        timestep = 0
        done = False
        
        try:
            while not done and timestep < max_steps:
                # Run step with all integrators
                timestep_results, current_obs, any_done = self._run_with_all_integrators(
                    envs, action_dim, seed, self.config.timeout
                )
                
                # Select best action
                best_action, best_result = self._select_best_action(timestep_results)
                
                # Store results for this timestep if we have valid data
                if best_action is not None and current_obs is not None and best_result is not None:
                    # Only include if observation passes filtering criteria
                    if self.filtering_strategy.should_keep(current_obs):
                        # Store observation and selected action
                        observations.append(current_obs)
                        selected_actions.append(best_action)
                        
                        # Store associated values for the selected action
                        action_values.append([
                            best_result['reward'],
                            best_result['error'],
                            best_result['cpu_time'],
                            best_result['time_reward'],
                            best_result['error_reward']
                        ])
                        
                        # Track stage information
                        stage = best_result.get('current_stage', 'UNKNOWN')
                        stage_counts[stage] = stage_counts.get(stage, 0) + 1
                
                # Update loop conditions
                done = any_done
                timestep += 1
        except Exception as e:
            logger.error(f"Episode error: {str(e)}")
        finally:
            # Close environments
            for env in envs.values():
                env.close()
        
        # Create combined history array
        if not observations:
            return None, {'valid': False, 'episode_time': time.time() - episode_start}
        
        # Convert lists to numpy arrays
        obs_array = np.array(observations)
        actions_array = np.array(selected_actions).reshape(-1, 1)
        values_array = np.array(action_values)
        
        # Combine into a single array
        # [observation, action, reward, error, cpu_time, time_reward, error_reward]
        combined_history = np.concatenate(
            [obs_array, actions_array, values_array], 
            axis=1
        )
        
        # Count by action
        action_counts = {}
        for action in selected_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate percentages
        total_count = len(selected_actions)
        action_percentages = {
            action: (count / total_count) * 100 
            for action, count in action_counts.items()
        }
        
        # Compile episode statistics
        episode_time = time.time() - episode_start
        episode_stats = {
            'valid': True,
            'filtered_observations': len(selected_actions),
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'stage_counts': stage_counts,
            'episode_time': episode_time,
            'temperature': fixed_temperature,
            'pressure': fixed_pressure,
            'phi': fixed_phi,
            'timestep': fixed_dt
        }
        
        return combined_history, episode_stats
        
    def collect_optimal_data(self):
        """
        Collect optimal integrator data across multiple episodes.
        
        Returns:
            list: List of combined history arrays
            dict: Overall statistics
        """
        all_combined_history = []
        overall_stats = {
            'episode_stats': [],
            'total_observations': 0,
            'filtered_observations': 0,
            'episodes_completed': 0,
            'computation_time': 0,
            'action_counts': {},
            'stage_counts': {},
        }
        
        start_time = time.time()
        
        for episode in tqdm(range(self.config.num_episodes), desc="Collecting optimal integrator data"):
            logger.info(f"\nCollecting data for episode {episode+1}/{self.config.num_episodes}")
            
            history, episode_stats = self._collect_single_episode()
            
            if history is not None and episode_stats['valid']:
                all_combined_history.append(history)
                print(f"Episode {episode+1} completed with {episode_stats['filtered_observations']} valid observations")
                
                # Update overall action counts
                for action, count in episode_stats['action_counts'].items():
                    overall_stats['action_counts'][action] = \
                        overall_stats['action_counts'].get(action, 0) + count
                
                # Update stage counts
                for stage, count in episode_stats.get('stage_counts', {}).items():
                    overall_stats['stage_counts'][stage] = \
                        overall_stats['stage_counts'].get(stage, 0) + count
                
                # Update episode stats
                episode_stats['episode'] = episode
                overall_stats['episode_stats'].append(episode_stats)
                overall_stats['filtered_observations'] += episode_stats['filtered_observations']
                overall_stats['episodes_completed'] += 1
                overall_stats['computation_time'] += episode_stats['episode_time']
                
                # Print episode stats
                if self.config.verbose or episode % 5 == 0:
                    logger.info(f"Episode {episode+1} results ({episode_stats['episode_time']:.2f}s):")
                    logger.info(f"  Filtered observations: {episode_stats['filtered_observations']}")
                    
                    # Print action distribution
                    for action, percentage in episode_stats['action_percentages'].items():
                        action_name = str(action)  # Get better name if available
                        logger.info(f"  {action_name}: {percentage:.1f}%")

            else:
                logger.info(f"[WARNING] Episode {episode+1} completed with {episode_stats['filtered_observations']} valid observations")
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        
        total_counts = sum(overall_stats['action_counts'].values())
        if total_counts > 0:
            overall_stats['action_percentages'] = {
                action: (count / total_counts) * 100 
                for action, count in overall_stats['action_counts'].items()
            }
        
        overall_stats['total_observations'] = sum(len(h) for h in all_combined_history)
        overall_stats['collection_time'] = elapsed_time
        overall_stats['observations_per_second'] = (overall_stats['total_observations'] / elapsed_time 
                                                   if elapsed_time > 0 else 0)
        
        # Print overall statistics
        if self.config.verbose:
            logger.info("\nData collection complete!")
            logger.info(f"Collected {overall_stats['total_observations']} unique observations from "
                      f"{overall_stats['episodes_completed']} episodes")
            logger.info(f"Total filtered observations: {overall_stats['filtered_observations']}")
            
            # Print action distribution
            logger.info("\nOverall action distribution:")
            for action, percentage in overall_stats.get('action_percentages', {}).items():
                logger.info(f"  Action {action}: {percentage:.1f}%")
            
            logger.info(f"\nTotal time: {elapsed_time:.1f}s (computation: {overall_stats['computation_time']:.1f}s)")
            logger.info(f"Processing speed: {overall_stats['observations_per_second']:.1f} obs/sec")
        
        return all_combined_history, overall_stats
        
    def save_dataset(self, all_combined_history, output_path='pretraining_dataset.npy'):
        """
        Concatenate and save all collected history into a single dataset.
        
        Args:
            all_combined_history: List of history arrays from collect_optimal_data
            output_path: Path to save the combined dataset
            
        Returns:
            numpy.ndarray: The concatenated dataset
        """
        if not all_combined_history:
            logger.warning("No data to save!")
            return None
            
        # Concatenate all episodes
        dataset = np.concatenate(all_combined_history, axis=0)
        logger.info(f"Final dataset shape: {dataset.shape}")
        
        # Save the dataset
        np.save(output_path, dataset)
        logger.info(f"Dataset saved to '{output_path}'")
        
        return dataset

#-------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------

class IntegratorDataset(Dataset):
    """Dataset for pretraining the PPO actor using optimal integrator choices."""
    
    def __init__(self, data, obs_dim):
        """
        Initialize dataset from collected optimal integrator data.
        
        Args:
            data: Combined history data with structure [obs, action, reward, error, cpu_time, time_reward, error_reward]
            obs_dim: Dimension of the observation space
        """
        self.data = data
        self.obs_dim = obs_dim
        
        # Extract observations and actions from the combined data
        self.states = data[:, :obs_dim].astype(np.float32)
        self.actions = data[:, obs_dim].astype(np.int64)
        
        # Calculate action distribution
        unique_actions, counts = np.unique(self.actions, return_counts=True)
        total = len(self.actions)
        
        logger.info("Action distribution in dataset:")
        for action, count in zip(unique_actions, counts):
            logger.info(f"  Action {int(action)}: {count} samples ({100 * count / total:.2f}%)")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class EarlyStopping:
    """Early stopping handler for training."""
    
    def __init__(self, patience=5, delta=0.001):
        """
        Initialize early stopping handler.
        
        Args:
            patience: Number of epochs to wait for improvement
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_state_dict = None
        self.counter = 0
        self.best_epoch = -1
        self.early_stop = False
    
    def __call__(self, epoch, val_loss, model_state_dict):
        """
        Check if training should be stopped.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            model_state_dict: Current model state dictionary
            
        Returns:
            bool: True if improved, False otherwise
        """
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = model_state_dict.copy()
            self.best_epoch = epoch
            return True
        
        if score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"  No improvement for {self.counter} epochs. "
                      f"Best: {-self.best_score:.4f} at epoch {self.best_epoch+1}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        
        logger.info(f"  Validation loss improved from {-self.best_score:.4f} to {-score:.4f}")
        self.best_score = score
        self.best_state_dict = model_state_dict.copy()
        self.best_epoch = epoch
        self.counter = 0
        return True


class PretrainTrainer:
    """Manages pretraining of PPO actor network."""
    
    def __init__(self, agent, training_config, device=None):
        """
        Initialize trainer for pretraining PPO actor.
        
        Args:
            agent: PPO agent with actor to pretrain
            training_config: Training configuration
            device: Device to use for training
        """
        self.agent = agent
        self.config = training_config
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def create_data_loaders(self, dataset, test_size=0.2):
        """
        Create train and test data loaders.
        
        Args:
            dataset: Full dataset
            test_size: Fraction to use for testing
            
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Split data
        train_indices, test_indices = train_test_split(
            np.arange(len(dataset)),
            test_size=test_size,
            stratify=dataset.actions,  # Stratify to maintain class distribution
            random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        logger.info(f"Training set: {len(train_dataset)} samples")
        logger.info(f"Testing set: {len(test_dataset)} samples")
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader):
        """
        Train the PPO actor with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for validation data
            
        Returns:
            dict: Training history and metrics
        """
        # Extract the actor network from the agent
        actor = self.agent.policy.actor
        actor.to(self.device)
        actor.train()
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(actor.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping handler
        early_stopping = EarlyStopping(patience=self.config.patience)
        
        # Training metrics
        train_loss_history = []
        test_loss_history = []
        train_acc_history = []
        test_acc_history = []
        
        logger.info(f"Starting pretraining for up to {self.config.epochs} epochs "
                  f"with early stopping (patience={self.config.patience})")
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training phase
            actor.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for states, actions in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{self.config.epochs}", leave=False):
                states = states.float().to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                action_probs = actor(states)
                loss = criterion(action_probs, actions)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(action_probs.data, 1)
                train_total += actions.size(0)
                train_correct += (predicted == actions).sum().item()
                
                train_loss += loss.item() * states.size(0)  # Weighted by batch size
            
            # Calculate average training metrics
            avg_train_loss = train_loss / train_total if train_total > 0 else 0
            train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
            
            # Validation phase
            actor.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for states, actions in tqdm(test_loader, desc=f"Test Epoch {epoch+1}/{self.config.epochs}", leave=False):
                    states = states.float().to(self.device)
                    actions = actions.to(self.device)
                    
                    # Forward pass
                    action_probs = actor(states)
                    loss = criterion(action_probs, actions)
                    
                    # Calculate metrics
                    _, predicted = torch.max(action_probs.data, 1)
                    test_total += actions.size(0)
                    test_correct += (predicted == actions).sum().item()
                    
                    test_loss += loss.item() * states.size(0)  # Weighted by batch size
            
            # Calculate average validation metrics
            avg_test_loss = test_loss / test_total if test_total > 0 else 0
            test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
            
            # Record history
            train_loss_history.append(avg_train_loss)
            test_loss_history.append(avg_test_loss)
            train_acc_history.append(train_accuracy)
            test_acc_history.append(test_accuracy)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            logger.info(f"  Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            
            # Check for improvement with early stopping
            improved = early_stopping(epoch, avg_test_loss, actor.state_dict())
            if improved:
                logger.info(f"  New best model! (Test Loss: {avg_test_loss:.4f})")
            
            if early_stopping.early_stop:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model
        actor.load_state_dict(early_stopping.best_state_dict)
        logger.info(f"Loaded best model from epoch {early_stopping.best_epoch+1}")
        
        # Collect all history
        history = {
            'train_loss': train_loss_history,
            'test_loss': test_loss_history,
            'train_acc': train_acc_history,
            'test_acc': test_acc_history,
            'best_epoch': early_stopping.best_epoch,
            'best_test_loss': -early_stopping.best_score,
            'best_test_acc': test_acc_history[early_stopping.best_epoch] if early_stopping.best_epoch < len(test_acc_history) else 0,
            'epochs_completed': len(train_loss_history)
        }
        
        return history, early_stopping.best_state_dict

#-------------------------------------------------------------------------
# Evaluation
#-------------------------------------------------------------------------

class MetricsCollector:
    """Collects and processes metrics during evaluation."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_distribution = {}
        self.stage_action_dist = {
            'PREIGNITION': {},
            'IGNITION': {},
            'POSTIGNITION': {}
        }
        self.integrator_metrics = {}
    
    def record_step(self, action, reward, info):
        """
        Record metrics from a single step.
        
        Args:
            action: Selected action
            reward: Reward received
            info: Information dictionary from environment
        """
        # Track action distribution
        self.action_distribution[action] = self.action_distribution.get(action, 0) + 1
        
        # Track stage-specific action choices
        if 'current_stage' in info:
            stage = info['current_stage']
            if stage in self.stage_action_dist:
                self.stage_action_dist[stage][action] = self.stage_action_dist[stage].get(action, 0) + 1
        
        # Track integrator-specific metrics
        if action not in self.integrator_metrics:
            self.integrator_metrics[action] = {'errors': [], 'cpu_times': []}
            
        if 'error' in info:
            self.integrator_metrics[action]['errors'].append(info['error'])
        if 'cpu_time' in info:
            self.integrator_metrics[action]['cpu_times'].append(info['cpu_time'])
    
    def record_episode(self, episode_reward, steps):
        """
        Record metrics from a complete episode.
        
        Args:
            episode_reward: Total reward for the episode
            steps: Number of steps in the episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
    
    def compute_statistics(self, action_names=None):
        """
        Compute overall statistics from collected metrics.
        
        Args:
            action_names: Optional mapping from action indices to names
            
        Returns:
            dict: Statistics dictionary
        """
        if action_names is None:
            action_names = {i: str(i) for i in self.action_distribution.keys()}
        
        # Calculate overall statistics
        mean_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        mean_steps = np.mean(self.episode_lengths)
        
        # Calculate action distribution percentages
        total_actions = sum(self.action_distribution.values())
        action_percentages = {k: (v / total_actions) * 100 for k, v in self.action_distribution.items()}
        
        # Calculate stage-specific action percentages
        stage_percentages = {}
        for stage, dist in self.stage_action_dist.items():
            stage_total = sum(dist.values())
            if stage_total > 0:
                stage_percentages[stage] = {k: (v / stage_total) * 100 for k, v in dist.items()}
            else:
                stage_percentages[stage] = {k: 0 for k in dist.keys()}
        
        # Calculate integrator-specific metrics
        for action in self.integrator_metrics:
            if self.integrator_metrics[action]['errors']:
                self.integrator_metrics[action]['mean_error'] = np.mean(self.integrator_metrics[action]['errors'])
                self.integrator_metrics[action]['median_error'] = np.median(self.integrator_metrics[action]['errors'])
            else:
                self.integrator_metrics[action]['mean_error'] = 0
                self.integrator_metrics[action]['median_error'] = 0
                
            if self.integrator_metrics[action]['cpu_times']:
                self.integrator_metrics[action]['mean_cpu_time'] = np.mean(self.integrator_metrics[action]['cpu_times'])
                self.integrator_metrics[action]['median_cpu_time'] = np.median(self.integrator_metrics[action]['cpu_times'])
            else:
                self.integrator_metrics[action]['mean_cpu_time'] = 0
                self.integrator_metrics[action]['median_cpu_time'] = 0
        
        # Compile all statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_steps': mean_steps,
            'action_distribution': self.action_distribution,
            'action_percentages': action_percentages,
            'stage_action_dist': self.stage_action_dist,
            'stage_percentages': stage_percentages,
            'integrator_metrics': self.integrator_metrics,
            'action_names': action_names
        }
        
        return stats


class Evaluator:
    """Evaluates pretrained policy on combustion environments."""
    
    def __init__(self, agent, env_manager, config, device=None):
        """
        Initialize evaluator.
        
        Args:
            agent: Pretrained PPO agent
            env_manager: Environment manager
            config: Evaluation configuration
            device: Device for evaluation
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def evaluate(self):
        """
        Evaluate the pretrained policy with detailed metrics.
        
        Returns:
            dict: Evaluation statistics
        """
        metrics_collector = MetricsCollector()
        
        logger.info(f"Evaluating pretrained policy for {self.config.num_episodes} episodes...")
        
        for episode in range(self.config.num_episodes):
            env = self.env_manager.create_single_env()
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                # Use deterministic policy (argmax) for evaluation
                with torch.no_grad():
                    action = self.agent.select_action(obs, deterministic=True)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(
                    action, timeout=self.config.timeout
                )
                
                # Record metrics
                metrics_collector.record_step(action, reward, info)
                
                # Update episode state
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
            # Record episode metrics
            metrics_collector.record_episode(episode_reward, steps)
            
            # Optional rendering
            if self.config.render:
                render_path = f"{os.getcwd()}/evaluation_episode_{episode+1}"
                os.makedirs(render_path, exist_ok=True)
                env.render(render_path)
            
            logger.info(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
            
            # Clean up
            env.close()
        
        # Get action names from environment
        env = self.env_manager.create_single_env()
        action_list = env.integrator.action_list
        action_names = {i: str(action) for i, action in enumerate(action_list)}
        env.close()
        
        # Compute and return statistics
        stats = metrics_collector.compute_statistics(action_names)
        
        # Print summary
        self._print_evaluation_summary(stats)
        
        return stats
    
    def _print_evaluation_summary(self, stats):
        """
        Print summary of evaluation results.
        
        Args:
            stats: Statistics dictionary
        """
        logger.info("\nEvaluation Results:")
        logger.info(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        logger.info(f"Mean Episode Length: {stats['mean_steps']:.1f} steps")
        logger.info(f"\nOverall Action Distribution:")
        
        for action, percentage in stats['action_percentages'].items():
            action_name = stats['action_names'][action]
            logger.info(f"{action_name}: {percentage:.1f}%")
        
        logger.info("\nStage-Specific Action Distribution:")
        for stage, percentages in stats['stage_percentages'].items():
            if sum(stats['stage_action_dist'][stage].values()) > 0:
                logger.info(f"  {stage}:")
                for action, percentage in percentages.items():
                    action_name = stats['action_names'][action]
                    logger.info(f"{action_name}: {percentage:.1f}%")
        
        logger.info("\nIntegrator Performance Metrics:")
        for action, metrics in stats['integrator_metrics'].items():
            action_name = stats['action_names'][action]
            if 'mean_error' in metrics:
                logger.info(f"  {action_name}: Mean Error: {metrics['mean_error']:.6f}, "
                          f"Mean CPU Time: {metrics['mean_cpu_time']:.6f}s")


#-------------------------------------------------------------------------
# Visualization
#-------------------------------------------------------------------------

class Visualizer:
    """Base class for visualization."""
    
    @staticmethod
    def show_and_save(fig, save_path=None):
        """
        Show figure and optionally save to file.
        
        Args:
            fig: Matplotlib figure
            save_path: Optional path to save the figure
        """
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.close(fig)


class TrainingVisualizer(Visualizer):
    """Creates visualizations for training results."""
    
    @staticmethod
    def plot_training_metrics(history, save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            history: Training history from pretrain_ppo_actor_with_validation
            save_path: Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss', marker='o', linestyle='-', markersize=4)
        plt.plot(history['test_loss'], label='Validation Loss', marker='s', linestyle='-', markersize=4)
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', 
                    label=f'Best Model (Epoch {history["best_epoch"]+1})')
        plt.title('Loss During Training', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(history['train_acc'], label='Train Accuracy', marker='o', linestyle='-', markersize=4)
        plt.plot(history['test_acc'], label='Validation Accuracy', marker='s', linestyle='-', markersize=4)
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', 
                    label=f'Best Model (Epoch {history["best_epoch"]+1})')
        plt.title('Accuracy During Training', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        TrainingVisualizer.show_and_save(fig, save_path)
        
        return fig


class DataCollectionVisualizer(Visualizer):
    """Creates visualizations for data collection results."""
    
    @staticmethod
    def plot_data_collection_stats(stats, save_path=None):
        """
        Plot statistics from data collection.
        
        Args:
            stats: Overall statistics from data collection
            save_path: Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot action distribution
        ax1 = plt.subplot(2, 2, 1)
        if 'action_percentages' in stats and stats['action_percentages']:
            actions = sorted(list(stats['action_percentages'].keys()))
            percentages = [stats['action_percentages'][a] for a in actions]
            
            ax1.bar(range(len(actions)), percentages, color='skyblue', edgecolor='navy')
            ax1.set_xticks(range(len(actions)))
            ax1.set_xticklabels([f"Action {a}" for a in actions], rotation=45)
            ax1.set_title('Action Distribution', fontsize=14)
            ax1.set_ylabel('Percentage (%)', fontsize=12)
            
            # Add percentage labels on top of bars
            for i, p in enumerate(percentages):
                ax1.text(i, p + 1, f"{p:.1f}%", ha='center', fontsize=9)
            
            ax1.grid(True, axis='y', alpha=0.3)
        
        # Plot observations per episode
        ax2 = plt.subplot(2, 2, 2)
        if 'episode_stats' in stats and stats['episode_stats']:
            episodes = [e.get('episode', i) for i, e in enumerate(stats['episode_stats'])]
            filtered_obs = [e.get('filtered_observations', 0) for e in stats['episode_stats']]
            
            ax2.plot(episodes, filtered_obs, 'g-o', label='Filtered Observations')
            ax2.set_title('Observations per Episode', fontsize=14)
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_ylabel('Number of Observations', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add average line
            avg_obs = np.mean(filtered_obs)
            ax2.axhline(avg_obs, color='r', linestyle='--', 
                       label=f'Average: {avg_obs:.1f}')
            ax2.legend(fontsize=10)
        
        # Plot stage-specific action distribution if available
        ax3 = plt.subplot(2, 2, 3)
        if 'stage_percentages' in stats and stats['stage_percentages']:
            stages = list(stats['stage_percentages'].keys())
            actions = sorted(list(stats['action_percentages'].keys()))
            x = np.arange(len(stages))
            width = 0.8 / len(actions)
            colors = plt.cm.viridis(np.linspace(0, 1, len(actions)))
            
            for i, action in enumerate(actions):
                percentages = [stats['stage_percentages'][stage].get(action, 0) for stage in stages]
                ax3.bar(x + i*width - 0.4 + width/2, percentages, width, label=f'Action {action}',
                       color=colors[i])
            
            ax3.set_title('Action Distribution by Combustion Stage', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(stages, rotation=45)
            ax3.set_ylabel('Percentage (%)', fontsize=12)
            ax3.legend(fontsize=9, loc='upper right')
            ax3.grid(True, axis='y', alpha=0.3)
        
        # Plot computation time
        ax4 = plt.subplot(2, 2, 4)
        if 'episode_stats' in stats and stats['episode_stats']:
            times = [e.get('episode_time', 0) for e in stats['episode_stats']]
            ax4.plot(episodes, times, 'b-o')
            ax4.set_title('Computation Time per Episode', fontsize=14)
            ax4.set_xlabel('Episode', fontsize=12)
            ax4.set_ylabel('Time (s)', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            # Add average time
            avg_time = np.mean(times)
            ax4.axhline(avg_time, color='r', linestyle='--')
            ax4.text(len(episodes)/2, avg_time*1.1, f"Average: {avg_time:.2f}s", 
                    ha='center', fontsize=10)
        
        plt.tight_layout()
        
        DataCollectionVisualizer.show_and_save(fig, save_path)
        
        return fig


class EvaluationVisualizer(Visualizer):
    """Creates visualizations for evaluation results."""
    
    @staticmethod
    def plot_integrator_comparison(stats, save_path=None):
        """
        Compare performance of different integrators from evaluation statistics.
        
        Args:
            stats: Evaluation statistics from evaluate method
            save_path: Path to save the comparison plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Extract integrator metrics and action names
        integrator_metrics = stats['integrator_metrics']
        action_names = stats['action_names']
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # 1. CPU Time comparison
        ax1 = plt.subplot(2, 2, 1)
        for action, metrics in integrator_metrics.items():
            if metrics.get('cpu_times', []):
                ax1.hist(metrics['cpu_times'], bins=20, alpha=0.7, 
                         label=action_names[action])
        
        ax1.set_xlabel('CPU Time (s)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('CPU Time Distribution by Integrator', fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Error comparison (log scale)
        ax2 = plt.subplot(2, 2, 2)
        for action, metrics in integrator_metrics.items():
            if metrics.get('errors', []):
                ax2.hist(metrics['errors'], bins=20, alpha=0.7, 
                         label=action_names[action])
        
        ax2.set_xlabel('Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Error Distribution by Integrator', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Use log scale if errors span multiple orders of magnitude
        if any('errors' in metrics and max(metrics['errors']) / (min(metrics['errors']) + 1e-10) > 100 
               for metrics in integrator_metrics.values() if 'errors' in metrics and metrics['errors']):
            ax2.set_xscale('log')
        
        # 3. Action distribution by stage
        ax3 = plt.subplot(2, 2, 3)
        stages = list(stats['stage_percentages'].keys())
        x = np.arange(len(stages))
        actions = list(integrator_metrics.keys())
        width = 0.8 / len(integrator_metrics)
        colors = plt.cm.viridis(np.linspace(0, 1, len(actions)))
        
        for i, action in enumerate(actions):
            percentages = [stats['stage_percentages'][stage].get(action, 0) 
                          for stage in stages]
            ax3.bar(x + i*width - 0.4 + width/2, percentages, width, 
                   label=action_names[action], color=colors[i])
        
        ax3.set_xlabel('Combustion Stage', fontsize=12)
        ax3.set_ylabel('Usage Percentage (%)', fontsize=12)
        ax3.set_title('Integrator Usage by Combustion Stage', fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels(stages, rotation=45)
        ax3.legend(fontsize=9)
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. Overall performance metrics comparison
        ax4 = plt.subplot(2, 2, 4)
        metrics_to_plot = ['mean_cpu_time', 'mean_error']
        metric_labels = ['Mean CPU Time (s)', 'Mean Error']
        x = np.arange(len(metrics_to_plot))
        
        # Normalize values for comparison
        max_values = {
            'mean_cpu_time': max(m.get('mean_cpu_time', 0) for m in integrator_metrics.values()),
            'mean_error': max(m.get('mean_error', 0) for m in integrator_metrics.values())
        }
        
        for i, (action, metrics) in enumerate(integrator_metrics.items()):
            values = []
            for metric in metrics_to_plot:
                if metric in metrics and max_values[metric] > 0:
                    values.append(metrics[metric] / max_values[metric])
                else:
                    values.append(0)
            
            ax4.bar(x + i*width - 0.4 + width/2, values, width, 
                   label=action_names[action], color=colors[i])
            
            # Add actual values as text
            for j, metric in enumerate(metrics_to_plot):
                if metric in metrics:
                    ax4.text(j + i*width - 0.4 + width/2, values[j] + 0.05, 
                             f"{metrics[metric]:.4f}", ha='center', va='bottom',
                             rotation=45, fontsize=8)
        
        ax4.set_xlabel('Metric', fontsize=12)
        ax4.set_ylabel('Normalized Value', fontsize=12)
        ax4.set_title('Normalized Performance Metrics', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(metric_labels)
        ax4.legend(fontsize=9)
        ax4.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        EvaluationVisualizer.show_and_save(fig, save_path)
        
        return fig

    @staticmethod
    def plot_reward_distribution(stats, save_path=None):
        """
        Plot distribution of rewards from evaluation.
        
        Args:
            stats: Evaluation statistics
            save_path: Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        rewards = stats['episode_rewards']
        
        fig = plt.figure(figsize=(10, 6))
        
        plt.hist(rewards, bins=10, color='green', alpha=0.7, edgecolor='black')
        plt.axvline(stats['mean_reward'], color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean: {stats["mean_reward"]:.2f}')
        
        plt.title('Distribution of Episode Rewards', fontsize=14)
        plt.xlabel('Reward', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        EvaluationVisualizer.show_and_save(fig, save_path)
        
        return fig


#-------------------------------------------------------------------------
# Main Function and Execution
#-------------------------------------------------------------------------

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='RL Integrator Selector Training and Evaluation')
    
    # Mode selection
    parser.add_argument('--collect_data', action='store_true', help='Collect new training data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--visualize_only', action='store_true', help='Only visualize existing results')
    
    # Data collection arguments
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes for data collection')
    parser.add_argument('--distance_threshold', type=float, default=0.001, help='Threshold for filtering similar observations')
    parser.add_argument('--metric', type=str, default='reward', 
                        choices=['reward', 'error', 'cpu_time', 'time_reward', 'error_reward'], 
                        help='Metric for selecting optimal integrator')
    parser.add_argument('--timeout', type=float, default=1, help='Timeout for integration steps')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of dataset to use for testing')
    
    # Evaluation arguments
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')
    
    # Path arguments
    parser.add_argument('--data_path', type=str, default=None, help='Path to existing dataset')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory for saving results')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    
    # Mechanism arguments
    parser.add_argument('--mech_file', type=str, 
                        default='/home/elo/CODES/SCI-ML/RLIntegratorSelector/large_mechanism/large_mechanism/n-dodecane.yaml',
                        help='Path to mechanism file')
    parser.add_argument('--fuel', type=str, default='nc12h26', help='Fuel species name')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Get arguments
    args = parser.parse_args()
    
    # If no mode selected, default to all
    if not any([args.collect_data, args.train, args.evaluate, args.visualize_only]):
        args.collect_data = True
        args.train = True
        args.evaluate = True
    
    return args


def create_configs_from_args(args):
    """Create configuration objects from parsed arguments."""
    # Collection config
    collection_config = CollectionConfig(
        num_episodes=args.num_episodes,
        distance_threshold=args.distance_threshold,
        metric=args.metric,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        test_size=args.test_size
    )
    
    # Evaluation config
    evaluation_config = EvaluationConfig(
        num_episodes=args.eval_episodes,
        render=args.render,
        timeout=args.timeout
    )
    
    # Overall experiment config
    experiment_config = ExperimentConfig(
        collection=collection_config,
        training=training_config,
        evaluation=evaluation_config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        save_model=args.save_model,
        mech_file=args.mech_file,
        fuel=args.fuel
    )
    
    return experiment_config


def main():
    """Main execution function with optimized organization."""
    # Parse arguments and create configs
    args = parse_arguments()
    config = create_configs_from_args(args)
    
    # Setup experiment tracker for results and visualization
    tracker = ExperimentTracker(config)
    
    # Create environment manager
    env_args = Args(mech_file=config.mech_file, fuel=config.fuel)
    env_args.timeout = config.collection.timeout
    env_manager = EnvManager(env_args)
    
    # Create a sample environment to get dimensions
    env = env_manager.create_single_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------------------------------------------------
    # Data Collection Phase
    # -------------------------------------------------------------------------
    if args.collect_data:
        logger.info("\n=== COLLECTING OPTIMAL INTEGRATOR DATA ===")
        
        data_collector = IntegratorDataCollector(
            env_manager=env_manager,
            config=config.collection
        )
        
        collected_data, collection_stats = data_collector.collect_optimal_data()
        
        # Save collection statistics
        tracker.save_dict(collection_stats, 'collection_stats.json')
        
        # Plot collection statistics
        DataCollectionVisualizer.plot_data_collection_stats(
            collection_stats,
            save_path=os.path.join(tracker.plot_dir, 'data_collection_stats.png')
        )
        
        # Concatenate and save all collected data
        if collected_data:
            pretraining_dataset = np.concatenate(collected_data, axis=0)
            dataset_path = tracker.save_numpy(pretraining_dataset, 'pretraining_dataset.npy')
        else:
            logger.error("No valid data collected. Exiting.")
            return
    else:
        # Load existing dataset
        if config.data_path is None:
            logger.error("No data path provided. Use --data_path or --collect_data.")
            return
            
        logger.info(f"\n=== LOADING DATASET FROM {config.data_path} ===")
        pretraining_dataset = np.load(config.data_path)
        
        # Copy dataset to output directory
        dataset_path = tracker.save_numpy(pretraining_dataset, 'pretraining_dataset.npy')
    
    logger.info(f"\nDataset shape: {pretraining_dataset.shape}")
    
    # -------------------------------------------------------------------------
    # Training Phase
    # -------------------------------------------------------------------------
    if args.train:
        logger.info("\n=== PREPARING DATASET FOR TRAINING ===")
        
        # Initialize PPO agent
        agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=config.training.learning_rate,
            lr_critic=config.training.learning_rate,
            gamma=0.99,
            K_epochs=10,
            eps_clip=0.2,
            has_continuous_action_space=False
        )
        
        # Create dataset
        dataset = IntegratorDataset(pretraining_dataset, state_dim)
        
        # Train the model
        logger.info("\n=== TRAINING MODEL ===")
        trainer = PretrainTrainer(agent, config.training, device=device)
        
        # Create data loaders
        train_loader, test_loader = trainer.create_data_loaders(
            dataset, 
            test_size=config.training.test_size
        )
        
        # Train the model
        history, best_state_dict = trainer.train(train_loader, test_loader)
        
        # Save training history
        tracker.save_dict(history, 'training_history.json')
        
        # Plot training metrics
        TrainingVisualizer.plot_training_metrics(
            history,
            save_path=os.path.join(tracker.plot_dir, 'training_metrics.png')
        )
        
        # Save the model
        if config.save_model:
            logger.info("\n=== SAVING MODEL ===")
            
            # Save final model
            tracker.save_model(agent.policy.actor.state_dict(), 'pretrained_actor.pth')
            
            # Save best model
            tracker.save_model(best_state_dict, 'best_pretrained_actor.pth')
            
            # Update policy_old with best weights
            agent.policy.actor.load_state_dict(best_state_dict)
            agent.policy_old.actor.load_state_dict(best_state_dict)
    else:
        # Load existing model if evaluation is requested
        if args.evaluate:
            # Initialize PPO agent
            agent = PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                lr_actor=config.training.learning_rate,
                lr_critic=config.training.learning_rate,
                gamma=0.99,
                K_epochs=10,
                eps_clip=0.2,
                has_continuous_action_space=False
            )
            
            # Load model
            if args.model_path:
                logger.info(f"\n=== LOADING MODEL FROM {args.model_path} ===")
                agent.policy.actor.load_state_dict(torch.load(args.model_path))
                agent.policy_old.actor.load_state_dict(torch.load(args.model_path))
            else:
                logger.error("No model path provided for evaluation. Use --model_path.")
                return
    
    # -------------------------------------------------------------------------
    # Evaluation Phase
    # -------------------------------------------------------------------------
    if args.evaluate:
        logger.info("\n=== EVALUATING MODEL ===")
        evaluator = Evaluator(agent, env_manager, config.evaluation, device=device)
        
        eval_stats = evaluator.evaluate()
        
        # Save evaluation statistics
        tracker.save_dict(eval_stats, 'evaluation_stats.json')
        
        # Plot evaluation results
        EvaluationVisualizer.plot_integrator_comparison(
            eval_stats,
            save_path=os.path.join(tracker.plot_dir, 'integrator_comparison.png')
        )
        
        EvaluationVisualizer.plot_reward_distribution(
            eval_stats,
            save_path=os.path.join(tracker.plot_dir, 'reward_distribution.png')
        )
    
    logger.info(f"\nAll results saved to: {tracker.output_dir}")


if __name__ == "__main__":
    main()