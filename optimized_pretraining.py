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

from config.default_config import Args
from agents.ppo_ import PPO
from environment.env_wrapper import EnvManager
import numpy as np
import cantera as ct

def parse_composition(gas, X):
    """create a dictionary of species:fraction pairs from the gas species names and X."""
    species_names = gas.species_names
    composition = {}
    for i, species in enumerate(species_names):
        composition[species] = X[i]
    return composition

data_path = '/home/elo/CODES/SCI-ML/ember/initial_conditions_uniform_z_100.npz'
data = np.load(data_path)
phis = data['phi']
Ts = data['T']
Ps = data['P']
Ys = data['Y']
Zs = data['Z']

# Reshape 1D arrays to 2D arrays with shape (n,1) before concatenating
phis = phis.reshape(-1,1) 
Ts = Ts.reshape(-1,1)
Ps = Ps.reshape(-1,1)
Zs = Zs.reshape(-1,1)

mech_file = '/home/elo/CODES/SCI-ML/RLIntegratorSelector/large_mechanism/large_mechanism/n-dodecane.yaml'
gas = ct.Solution(mech_file)
fuel = 'nc12h26'

# combine the phi, T, P, Y into a single array
initial_conditions = np.concatenate([phis, Ts, Ps, Zs, Ys], axis=1)

class IntegratorDataCollector:
    """Collects optimal integrator data by comparing integrators at each timestep."""
    
    def __init__(self, env_manager, distance_threshold=0.05, verbose=True):
        """
        Initialize data collector.
        
        Args:
            env_manager: Manager for combustion environments
            distance_threshold: Threshold for filtering similar observations
            verbose: Enable detailed logging
        """
        self.env_manager = env_manager
        self.distance_threshold = distance_threshold
        self.verbose = verbose
        
    def collect_optimal_data(self, num_episodes=10, metric='reward', timeout=None):
        """
        Collect optimal integrator data across multiple episodes.
        
        Args:
            num_episodes: Number of episodes to collect
            metric: Metric for selecting optimal integrator ('reward', 'error', 'cpu_time', etc.)
            timeout: Optional timeout for integration steps
            
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
        }
        
        start_time = time.time()
        
        # Map metrics to their indices in the history array
        metric_indices = {
            'reward': 0,
            'error': 1,
            'cpu_time': 2,
            'time_reward': 3,
            'error_reward': 4,
        }
        
        if metric not in metric_indices:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_idx = metric_indices[metric]
        error_idx = metric_indices['error']
        minimize_metrics = ['error', 'cpu_time']
        
        for episode in tqdm(range(num_episodes), desc="Collecting optimal integrator data"):
            if self.verbose:
                print(f"\nCollecting data for episode {episode+1}/{num_episodes}")
            
            try:
                episode_start = time.time()
                
                # Create identical environments for all integrators using the same random seed
                # This ensures they have the same initial conditions and reference solution
                seed = int(time.time() * 1000) % 10000  # Generate a seed for this episode

                condition = initial_conditions[np.random.randint(0, len(initial_conditions))]
                phi = condition[0]
                T = condition[1]
                P = condition[2]
                Z = condition[3]
                X = condition[4:]
                composition = parse_composition(gas, X)
                
                # Get parameters for this problem
                fixed_temperature = T
                fixed_pressure = 1
                fixed_phi = phi
                initial_mixture = composition
                fixed_dt = np.random.choice(
                    self.env_manager.args.min_time_steps_range if fixed_temperature > 1000 
                    else self.env_manager.args.max_time_steps_range
                )
                end_time = 1e-2
                
                if self.verbose:
                    print(f"Episode {episode+1} parameters: T={fixed_temperature}K, P={fixed_pressure}atm, " 
                          f"phi={fixed_phi}, dt={fixed_dt}s, Z={Z}")
                
                # Create environments with same parameters for all integrators
                envs = {}
                n_actions = None
                
                # First pass to determine number of actions and create environments
                env = self.env_manager.create_single_env(
                    end_time=self.env_manager.args.end_time,
                    fixed_temperature=fixed_temperature,
                    fixed_pressure=fixed_pressure,
                    fixed_phi=fixed_phi,
                    fixed_dt=fixed_dt,
                    initial_mixture=initial_mixture,
                    randomize=False  # Use fixed parameters
                )
                n_actions = env.action_space.n
                
                # Create one environment per integrator with same parameters
                for action in range(n_actions):
                    envs[action] = self.env_manager.create_single_env(
                        end_time=self.env_manager.args.end_time,
                        fixed_temperature=fixed_temperature,
                        fixed_pressure=fixed_pressure,
                        fixed_phi=fixed_phi,
                        fixed_dt=fixed_dt,
                        randomize=False  # Use fixed parameters
                    )
                    # Reset environment
                    obs, _ = envs[action].reset(seed=seed)
                
                # Dataset structure:
                # - observations
                # - selected_action 
                # - reward, error, cpu_time, time_reward, error_reward
                observations = []
                action_values = []  # Stores reward, error, etc. for each action
                selected_actions = []
                timestep = 0
                done = False
                
                # Dataset for filtered observations
                filtered_obs = []
                filtered_obs_buffer = np.empty((0, len(obs)))
                
                # Run episode using all integrators in parallel and select best at each step
                while not done and timestep < 10000:  # Safety limit
                    timestep_results = {}
                    current_obs = None
                    any_done = False
                    
                    # Run one step with each integrator
                    for action in range(n_actions):
                        try:
                            next_obs, reward, terminated, truncated, info = envs[action].step(
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
                                'done': action_done
                            }
                            
                            # Use any observation as current (they should be same state)
                            current_obs = next_obs.copy()
                            
                            # Check if any environment is done
                            if action_done:
                                any_done = True
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"Error with action {action} at timestep {timestep}: {e}")
                            # Mark this action as invalid for this timestep
                            timestep_results[action] = {
                                'obs': current_obs if current_obs is not None else None,
                                'reward': -float('inf'),
                                'error': float('inf'),
                                'cpu_time': float('inf'),
                                'time_reward': -float('inf'),
                                'error_reward': -float('inf'),
                                'done': True
                            }
                            any_done = True
                    
                    # Select best action based on chosen metric
                    best_action = None
                    best_value = float('inf') if metric in minimize_metrics else -float('inf')
                    
                    for action, result in timestep_results.items():
                        metric_value = result[metric]
                        error_value = result['error']
                        
                        # Skip invalid actions
                        if (metric in minimize_metrics and metric_value == float('inf')) or \
                           (metric not in minimize_metrics and metric_value == -float('inf')):
                            continue
                            
                        # Check if better
                        if metric in minimize_metrics:
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
                    
                    # Store results for this timestep
                    if best_action is not None and current_obs is not None:
                        # Only include if observation is significantly different from previous ones
                        current_obs_reshaped = current_obs.reshape(1, -1)
                        
                        if len(filtered_obs_buffer) > 0:
                            # Calculate distances to previous observations
                            distances = cdist(current_obs_reshaped, filtered_obs_buffer, 'euclidean')[0]
                            
                            # Keep if minimum distance exceeds threshold
                            if np.min(distances) >= self.distance_threshold:
                                # Store observation
                                observations.append(current_obs)
                                
                                # Store selected action
                                selected_actions.append(best_action)
                                
                                # Store associated values for the selected action
                                result = timestep_results[best_action]
                                action_values.append([
                                    result['reward'],
                                    result['error'],
                                    result['cpu_time'],
                                    result['time_reward'],
                                    result['error_reward']
                                ])
                                
                                # Update filtered buffer
                                filtered_obs_buffer = np.vstack((filtered_obs_buffer, current_obs_reshaped))
                                
                                # Count selected action
                                overall_stats['action_counts'][best_action] = overall_stats['action_counts'].get(best_action, 0) + 1
                        else:
                            # First observation is always kept
                            observations.append(current_obs)
                            selected_actions.append(best_action)
                            
                            result = timestep_results[best_action]
                            action_values.append([
                                result['reward'],
                                result['error'],
                                result['cpu_time'],
                                result['time_reward'],
                                result['error_reward']
                            ])
                            
                            filtered_obs_buffer = np.vstack((filtered_obs_buffer, current_obs_reshaped))
                            overall_stats['action_counts'][best_action] = overall_stats['action_counts'].get(best_action, 0) + 1
                    
                    # Check if all environments are done
                    done = any_done
                    timestep += 1
                
                # Episode complete - create combined history
                if observations:
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
                    
                    all_combined_history.append(combined_history)
                    
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
                    
                    # Calculate episode time
                    episode_time = time.time() - episode_start
                    overall_stats['computation_time'] += episode_time
                    
                    # Update episode stats
                    episode_stats = {
                        'episode': episode,
                        'filtered_observations': len(selected_actions),
                        'action_counts': action_counts,
                        'action_percentages': action_percentages,
                        'episode_time': episode_time,
                        'temperature': fixed_temperature,
                        'pressure': fixed_pressure,
                        'phi': fixed_phi,
                        'timestep': fixed_dt
                    }
                    
                    overall_stats['episode_stats'].append(episode_stats)
                    overall_stats['filtered_observations'] += len(selected_actions)
                    overall_stats['episodes_completed'] += 1
                    
                    # Print episode stats
                    if self.verbose or episode % 5 == 0:
                        print(f"Episode {episode+1} results ({episode_time:.2f}s):")
                        print(f"  Filtered observations: {len(selected_actions)}")
                        
                        # Print action distribution
                        for action, percentage in action_percentages.items():
                            action_name = str(envs[action].integrator.action_list[action])
                            print(f"  {action_name}: {percentage:.1f}%")
                
                # Close environments
                for env in envs.values():
                    env.close()
            
            except Exception as e:
                print(f"Error in episode {episode+1}: {e}")
                import traceback
                traceback.print_exc()
        
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
        if self.verbose:
            print("\nData collection complete!")
            print(f"Collected {overall_stats['total_observations']} unique observations from {overall_stats['episodes_completed']} episodes")
            print(f"Total filtered observations: {overall_stats['filtered_observations']}")
            
            # Print action distribution
            print("\nOverall action distribution:")
            for action, percentage in overall_stats.get('action_percentages', {}).items():
                print(f"  Action {action}: {percentage:.1f}%")
            
            print(f"\nTotal time: {elapsed_time:.1f}s (computation: {overall_stats['computation_time']:.1f}s)")
            print(f"Processing speed: {overall_stats['observations_per_second']:.1f} obs/sec")
        
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
            print("No data to save!")
            return None
            
        # Concatenate all episodes
        dataset = np.concatenate(all_combined_history, axis=0)
        print(f"Final dataset shape: {dataset.shape}")
        
        # Save the dataset
        np.save(output_path, dataset)
        print(f"Dataset saved to '{output_path}'")
        
        return dataset


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
        
        print("Action distribution in dataset:")
        for action, count in zip(unique_actions, counts):
            print(f"  Action {int(action)}: {count} samples ({100 * count / total:.2f}%)")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class PretrainTrainer:
    """Manages pretraining of PPO actor network."""
    
    def __init__(self, agent, device=None):
        """
        Initialize trainer for pretraining PPO actor.
        
        Args:
            agent: PPO agent with actor to pretrain
            device: Device to use for training
        """
        self.agent = agent
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def train(self, train_loader, test_loader, epochs=20, learning_rate=1e-4, patience=5):
        """
        Train the PPO actor with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for validation data
            epochs: Maximum number of epochs
            learning_rate: Learning rate for optimizer
            patience: Epochs to wait before early stopping
            
        Returns:
            dict: Training history and metrics
        """
        # Extract the actor network from the agent
        actor = self.agent.policy.actor
        actor.to(self.device)
        actor.train()
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        train_loss_history = []
        test_loss_history = []
        train_acc_history = []
        test_acc_history = []
        
        # Early stopping variables
        best_test_loss = float('inf')
        best_epoch = 0
        best_state_dict = None
        epochs_no_improve = 0
        
        print(f"Starting pretraining for up to {epochs} epochs with early stopping (patience={patience})")
        for epoch in range(epochs):
            # Training phase
            actor.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for states, actions in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False):
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
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for states, actions in tqdm(test_loader, desc=f"Test Epoch {epoch+1}/{epochs}", leave=False):
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
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            
            # Check for improvement
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_epoch = epoch
                best_state_dict = actor.state_dict().copy()
                epochs_no_improve = 0
                print(f"  New best model! (Test Loss: {best_test_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs. Best Test Loss: {best_test_loss:.4f} at epoch {best_epoch+1}")
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Load best model
        if best_state_dict is not None:
            actor.load_state_dict(best_state_dict)
            print(f"Loaded best model from epoch {best_epoch+1}")
        
        # Collect all history
        history = {
            'train_loss': train_loss_history,
            'test_loss': test_loss_history,
            'train_acc': train_acc_history,
            'test_acc': test_acc_history,
            'best_epoch': best_epoch,
            'best_test_loss': best_test_loss,
            'best_test_acc': test_acc_history[best_epoch] if best_epoch < len(test_acc_history) else 0,
            'epochs_completed': len(train_loss_history)
        }
        
        return history, best_state_dict


class Evaluator:
    """Evaluates pretrained policy on combustion environments."""
    
    def __init__(self, agent, env_manager, device=None):
        """
        Initialize evaluator.
        
        Args:
            agent: Pretrained PPO agent
            env_manager: Environment manager
            device: Device for evaluation
        """
        self.agent = agent
        self.env_manager = env_manager
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def evaluate(self, num_episodes=10, render=False, timeout=None, path=None):
        """
        Evaluate the pretrained policy with detailed metrics.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            timeout: Optional timeout for integration steps
            
        Returns:
            dict: Evaluation statistics
        """
        # Metrics to track
        episode_rewards = []
        episode_lengths = []
        action_distribution = {}
        stage_action_dist = {
            'PREIGNITION': {},
            'IGNITION': {},
            'POSTIGNITION': {}
        }
        
        # Track errors and CPU times for each integrator
        integrator_metrics = {}
        
        print(f"Evaluating pretrained policy for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            env = self.env_manager.create_single_env()
            
            # Initialize tracking for this episode
            action_list = env.integrator.action_list
            for i in range(env.action_space.n):
                action_distribution[i] = action_distribution.get(i, 0)
                integrator_metrics[i] = integrator_metrics.get(i, {'errors': [], 'cpu_times': []})
                
                for stage in stage_action_dist:
                    stage_action_dist[stage][i] = stage_action_dist[stage].get(i, 0)
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                # Use deterministic policy (argmax) for evaluation
                with torch.no_grad():
       
                    action = self.agent.select_action(obs, deterministic=True)
                
                # Track action distribution
                action_distribution[action] += 1
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action, timeout=timeout)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                # Track stage-specific action choices
                if 'current_stage' in info:
                    stage = info['current_stage']
                    if stage in stage_action_dist:
                        stage_action_dist[stage][action] += 1
                
                # Track integrator-specific metrics
                if 'error' in info:
                    integrator_metrics[action]['errors'].append(info['error'])
                if 'cpu_time' in info:
                    integrator_metrics[action]['cpu_times'].append(info['cpu_time'])
                
            if render:
                render_path = f"{path}/evaluation_episode_{episode+1}"
                os.makedirs(render_path, exist_ok=True)
                env.render(render_path)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
            print(f"Action distribution: {action_distribution}")
        
        # Calculate overall statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_steps = np.mean(episode_lengths)
        
        # Calculate action distribution percentages
        total_actions = sum(action_distribution.values())
        action_percentages = {k: (v / total_actions) * 100 for k, v in action_distribution.items()}
        
        # Calculate stage-specific action percentages
        stage_percentages = {}
        for stage, dist in stage_action_dist.items():
            stage_total = sum(dist.values())
            if stage_total > 0:
                stage_percentages[stage] = {k: (v / stage_total) * 100 for k, v in dist.items()}
            else:
                stage_percentages[stage] = {k: 0 for k in dist.keys()}
        
        # Calculate integrator-specific metrics
        for action in integrator_metrics:
            if integrator_metrics[action]['errors']:
                integrator_metrics[action]['mean_error'] = np.mean(integrator_metrics[action]['errors'])
                integrator_metrics[action]['median_error'] = np.median(integrator_metrics[action]['errors'])
            else:
                integrator_metrics[action]['mean_error'] = 0
                integrator_metrics[action]['median_error'] = 0
                
            if integrator_metrics[action]['cpu_times']:
                integrator_metrics[action]['mean_cpu_time'] = np.mean(integrator_metrics[action]['cpu_times'])
                integrator_metrics[action]['median_cpu_time'] = np.median(integrator_metrics[action]['cpu_times'])
            else:
                integrator_metrics[action]['mean_cpu_time'] = 0
                integrator_metrics[action]['median_cpu_time'] = 0
        
        # Compile all statistics
        env = self.env_manager.create_single_env()
        action_list = env.integrator.action_list
        action_names = {i: str(action) for i, action in enumerate(action_list)}
        
        stats = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_steps': mean_steps,
            'action_distribution': action_distribution,
            'action_percentages': action_percentages,
            'stage_action_dist': stage_action_dist,
            'stage_percentages': stage_percentages,
            'integrator_metrics': integrator_metrics,
            'action_names': action_names
        }
        
        # Print summary
        print("\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean Episode Length: {mean_steps:.1f} steps")
        print(f"\nOverall Action Distribution:")
        
        for action, percentage in action_percentages.items():
            action_name = action_names[action]
            print(f"{action_name}: {percentage:.1f}%")
        
        print("\nStage-Specific Action Distribution:")
        for stage, percentages in stage_percentages.items():
            if sum(stage_action_dist[stage].values()) > 0:
                print(f"  {stage}:")
                for action, percentage in percentages.items():
                    action_name = action_names[action]
                    print(f"{action_name}: {percentage:.1f}%")
        
        print("\nIntegrator Performance Metrics:")
        for action, metrics in integrator_metrics.items():
            action_name = action_names[action]
            if 'mean_error' in metrics:
                print(f"  {action_name}: Mean Error: {metrics['mean_error']:.6f}, Mean CPU Time: {metrics['mean_cpu_time']:.6f}s")
        
        return stats


class Visualizer:
    """Creates visualizations for pretraining and evaluation results."""
    
    @staticmethod
    def plot_training_metrics(history, save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            history: Training history from pretrain_ppo_actor_with_validation
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', marker='o')
        plt.plot(history['test_loss'], label='Validation Loss', marker='s')
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', 
                    label=f'Best Model (Epoch {history["best_epoch"]+1})')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(history['test_acc'], label='Validation Accuracy', marker='s')
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', 
                    label=f'Best Model (Epoch {history["best_epoch"]+1})')
        plt.title('Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Training metrics plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_data_collection_stats(stats, save_path=None):
        """
        Plot statistics from data collection.
        
        Args:
            stats: Overall statistics from data collection
            save_path: Path to save the plot
        """
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot action distribution
        if 'action_percentages' in stats and stats['action_percentages']:
            actions = list(stats['action_percentages'].keys())
            percentages = [stats['action_percentages'][a] for a in actions]
            
            ax1.bar(range(len(actions)), percentages)
            ax1.set_xticks(range(len(actions)))
            ax1.set_xticklabels([f"Action {a}" for a in actions])
            ax1.set_title('Action Distribution')
            ax1.set_ylabel('Percentage (%)')
            
            # Add percentage labels on top of bars
            for i, p in enumerate(percentages):
                ax1.text(i, p + 1, f"{p:.1f}%", ha='center')
        
        # Plot filtering efficiency
        if 'episode_stats' in stats and stats['episode_stats']:
            episodes = [e['episode'] for e in stats['episode_stats']]
            original = [sum(e['stats'].get('original_counts', {}).values()) for e in stats['episode_stats']]
            filtered = [e['stats'].get('total_observations', 0) for e in stats['episode_stats']]
            
            ax2.plot(episodes, original, 'b-', label='Original')
            ax2.plot(episodes, filtered, 'r-', label='Filtered')
            ax2.set_title('Filtering Efficiency')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Number of Observations')
            ax2.legend()
            
            # Add reduction percentage text
            if 'total_reduction_percentage' in stats:
                ax2.text(0.5, 0.9, f"Overall Reduction: {stats['total_reduction_percentage']:.1f}%", 
                         transform=ax2.transAxes, ha='center',
                         bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot stage-specific action distribution if available
        if 'stage_percentages' in stats and stats['stage_percentages']:
            stages = list(stats['stage_percentages'].keys())
            x = np.arange(len(stages))
            width = 0.8 / len(actions)
            
            for i, action in enumerate(actions):
                percentages = [stats['stage_percentages'][stage].get(action, 0) for stage in stages]
                ax3.bar(x + i*width - 0.4 + width/2, percentages, width, label=f'Action {action}')
            
            ax3.set_title('Action Distribution by Combustion Stage')
            ax3.set_xticks(x)
            ax3.set_xticklabels(stages)
            ax3.set_ylabel('Percentage (%)')
            ax3.legend()
        
        # Plot computation time
        if 'episode_stats' in stats and stats['episode_stats']:
            times = [e['episode_time'] for e in stats['episode_stats']]
            ax4.plot(episodes, times, 'g-')
            ax4.set_title('Computation Time per Episode')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Time (s)')
            
            # Add average time
            avg_time = np.mean(times)
            ax4.axhline(avg_time, color='r', linestyle='--')
            ax4.text(len(episodes)/2, avg_time*1.1, f"Average: {avg_time:.2f}s", ha='center')
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Data collection stats plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_integrator_comparison(stats, save_path=None):
        """
        Compare performance of different integrators from evaluation statistics.
        
        Args:
            stats: Evaluation statistics from evaluate method
            save_path: Path to save the comparison plot
        """
        # Extract integrator metrics and action names
        integrator_metrics = stats['integrator_metrics']
        action_names = stats['action_names']
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # 1. CPU Time comparison
        plt.subplot(2, 2, 1)
        for action, metrics in integrator_metrics.items():
            if metrics['cpu_times']:
                plt.hist(metrics['cpu_times'], bins=20, alpha=0.7, 
                         label=action_names[action])
        
        plt.xlabel('CPU Time (s)')
        plt.ylabel('Frequency')
        plt.title('CPU Time Distribution by Integrator')
        plt.legend()
        plt.grid(True)
        
        # 2. Error comparison
        plt.subplot(2, 2, 2)
        for action, metrics in integrator_metrics.items():
            if metrics['errors']:
                plt.hist(metrics['errors'], bins=20, alpha=0.7, 
                         label=action_names[action])
        
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution by Integrator')
        plt.legend()
        plt.grid(True)
        
        # 3. Action distribution by stage
        plt.subplot(2, 2, 3)
        stages = list(stats['stage_percentages'].keys())
        x = np.arange(len(stages))
        width = 0.8 / len(integrator_metrics)
        
        for i, action in enumerate(integrator_metrics.keys()):
            percentages = [stats['stage_percentages'][stage].get(action, 0) 
                          for stage in stages]
            plt.bar(x + i*width - 0.4 + width/2, percentages, width, 
                   label=action_names[action])
        
        plt.xlabel('Combustion Stage')
        plt.ylabel('Usage Percentage (%)')
        plt.title('Integrator Usage by Combustion Stage')
        plt.xticks(x, stages)
        plt.legend()
        plt.grid(True)
        
        # 4. Overall performance metrics comparison
        plt.subplot(2, 2, 4)
        metrics_to_plot = ['mean_cpu_time', 'mean_error']
        metric_labels = ['Mean CPU Time (s)', 'Mean Error']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(integrator_metrics)
        
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
            
            plt.bar(x + i*width - 0.4 + width/2, values, width, 
                   label=action_names[action])
            
            # Add actual values as text
            for j, metric in enumerate(metrics_to_plot):
                if metric in metrics:
                    plt.text(j + i*width - 0.4 + width/2, values[j] + 0.05, 
                             f"{metrics[metric]:.4f}", ha='center', va='bottom',
                             rotation=45, fontsize=8)
        
        plt.xlabel('Metric')
        plt.ylabel('Normalized Value')
        plt.title('Normalized Performance Metrics')
        plt.xticks(x, metric_labels)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Integrator comparison plot saved to {save_path}")
        
        plt.show()

def setup_argparse():
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(description='Pretrain RL integrator selector')
    
    # Data collection arguments
    parser.add_argument('--collect_data', action='store_true', help='Collect new training data')
    parser.add_argument('--num_episodes', type=int, default=200, help='Number of episodes for data collection')
    parser.add_argument('--distance_threshold', type=float, default=0.001, help='Threshold for filtering similar observations')
    parser.add_argument('--metric', type=str, default='reward', choices=['reward', 'error', 'cpu_time', 'time_reward', 'error_reward'], 
                        help='Metric for selecting optimal integrator')
    parser.add_argument('--timeout', type=float, default=0.3, help='Timeout for integration steps')
    
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
    parser.add_argument('--data_path', type=str, default=None, help='Path to existing dataset (if not collecting new data)')
    parser.add_argument('--output_dir', type=str, default='pretraining_results', help='Directory for saving results')
    parser.add_argument('--save_model', action='store_true', help='Save pretrained model')
    
    return parser


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{timestamp}"
    model_dir = os.path.join(output_dir, "models")
    plot_dir = os.path.join(output_dir, "plots")
    data_dir = os.path.join(output_dir, "data")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Create environment and agent
    mech_file = '/home/elo/CODES/SCI-ML/RLIntegratorSelector/large_mechanism/large_mechanism/n-dodecane.yaml'
    gas = ct.Solution(mech_file)
    fuel = 'nc12h26'
    env_args = Args(mech_file=mech_file, fuel=fuel)
    env_args.timeout = args.timeout
    env_manager = EnvManager(env_args)
    
    # Create a sample environment to get dimensions
    env = env_manager.create_single_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize PPO agent
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.learning_rate,
        lr_critic=args.learning_rate,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        has_continuous_action_space=False
    )
    
    # Set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Data collection or loading
    if args.collect_data:
        print("\n=== COLLECTING OPTIMAL INTEGRATOR DATA ===")
        data_collector = IntegratorDataCollector(
            env_manager=env_manager,
            distance_threshold=args.distance_threshold,
            verbose=True
        )
        
        collected_data, collection_stats = data_collector.collect_optimal_data(
            num_episodes=args.num_episodes,
            metric=args.metric,
            timeout=args.timeout
        )
        
        # Save collection statistics
        with open(os.path.join(data_dir, 'collection_stats.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_stats = {}
            for key, value in collection_stats.items():
                if key == 'episode_stats':
                    serializable_stats[key] = value
                elif isinstance(value, dict):
                    serializable_stats[key] = {k: float(v) if isinstance(v, np.number) else v 
                                              for k, v in value.items()}
                elif isinstance(value, np.number):
                    serializable_stats[key] = float(value)
                else:
                    serializable_stats[key] = value
            
            json.dump(serializable_stats, f, indent=4)
        
        # Plot collection statistics
        Visualizer.plot_data_collection_stats(
            collection_stats,
            save_path=os.path.join(plot_dir, 'data_collection_stats.png')
        )
        
        # Concatenate and save all collected data
        if collected_data:
            pretraining_dataset = np.concatenate(collected_data, axis=0)
            dataset_path = os.path.join(data_dir, 'pretraining_dataset.npy')
            np.save(dataset_path, pretraining_dataset)
            print(f"Dataset saved to {dataset_path}")
        else:
            print("No valid data collected. Exiting.")
            return
    else:
        # Load existing dataset
        if args.data_path is None:
            print("No data path provided. Use --data_path or --collect_data.")
            return
            
        print(f"\n=== LOADING DATASET FROM {args.data_path} ===")
        pretraining_dataset = np.load(args.data_path)
        
        # Copy dataset to output directory
        dataset_path = os.path.join(data_dir, 'pretraining_dataset.npy')
        np.save(dataset_path, pretraining_dataset)
    
    print(f"\nDataset shape: {pretraining_dataset.shape}")
    
    # Create dataset and split into train/test
    print("\n=== PREPARING DATASET FOR TRAINING ===")
    dataset = IntegratorDataset(pretraining_dataset, state_dim)
    
    # Split data
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=args.test_size,
        stratify=dataset.actions,  # Stratify to maintain class distribution
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Testing set: {len(test_dataset)} samples")
    
    # Train the model
    print("\n=== TRAINING MODEL ===")
    trainer = PretrainTrainer(agent, device=device)
    
    history, best_state_dict = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience
    )
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        # Convert values to standard Python types for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list) and all(isinstance(v, (int, float, np.number)) for v in value):
                serializable_history[key] = [float(v) if isinstance(v, np.number) else v for v in value]
            elif isinstance(value, np.number):
                serializable_history[key] = float(value)
            else:
                serializable_history[key] = value
        
        json.dump(serializable_history, f, indent=4)
    
    # Plot training metrics
    Visualizer.plot_training_metrics(
        history,
        save_path=os.path.join(plot_dir, 'training_metrics.png')
    )
    
    # Save the model
    if args.save_model:
        print("\n=== SAVING MODEL ===")
        
        # Save final model
        final_path = os.path.join(model_dir, 'pretrained_actor.pth')
        torch.save(agent.policy.actor.state_dict(), final_path)
        print(f"Final model saved to {final_path}")
        
        # Save best model
        best_path = os.path.join(model_dir, 'best_pretrained_actor.pth')
        torch.save(best_state_dict, best_path)
        print(f"Best model saved to {best_path}")
        
        # Update policy_old with best weights
        agent.policy.actor.load_state_dict(best_state_dict)
        agent.policy_old.actor.load_state_dict(best_state_dict)
    
    # Evaluate the model
    print("\n=== EVALUATING MODEL ===")
    evaluator = Evaluator(agent, env_manager, device=device)
    
    eval_stats = evaluator.evaluate(
        num_episodes=args.eval_episodes,
        render=args.render,
        timeout=args.timeout,
        path=plot_dir
    )
    
    # Save evaluation statistics
    with open(os.path.join(output_dir, 'evaluation_stats.json'), 'w') as f:
        # Convert numpy arrays/values to standard Python types for JSON serialization
        serializable_stats = {}
        for key, value in eval_stats.items():
            if key in ['episode_rewards', 'episode_lengths']:
                serializable_stats[key] = [float(v) if isinstance(v, np.number) else v for v in value]
            elif isinstance(value, dict):
                if key == 'integrator_metrics':
                    serializable_stats[key] = {}
                    for action, metrics in value.items():
                        serializable_stats[key][str(action)] = {
                            k: [float(v) if isinstance(v, np.number) else v for v in values] 
                              if isinstance(values, list) else (float(values) if isinstance(values, np.number) else values)
                            for k, values in metrics.items()
                        }
                else:
                    serializable_stats[key] = {k: float(v) if isinstance(v, np.number) else v 
                                             for k, v in value.items()}
            elif isinstance(value, np.number):
                serializable_stats[key] = float(value)
            else:
                serializable_stats[key] = value
                
        json.dump(serializable_stats, f, indent=4)
    
    # Plot integrator comparison
    Visualizer.plot_integrator_comparison(
        eval_stats,
        save_path=os.path.join(plot_dir, 'integrator_comparison.png')
    )
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
