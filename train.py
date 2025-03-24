import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional
import random
from config.default_config import Args
from agents.ppo_ import PPO
from environment.env_wrapper import EnvManager
from utils.logging_utils import Logger
from utils.evaluation_utils import evaluate_policy, run_episode

class SingleEnvTrainer:
    """Trainer class for PPO algorithm using a single combustion environment."""
    
    def __init__(self, args):
        """Initialize trainer with configuration."""
        self.args = args
        self.setup_experiment()
        
        # Initialize environment manager and create single environment
        self.env_manager = EnvManager(args)
        self.env = self.env_manager.create_single_env()
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Initialize PPO agent
        self.agent = PPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr_actor=args.learning_rate,
            lr_critic=args.learning_rate,
            gamma=args.gamma,
            K_epochs=args.update_epochs,
            eps_clip=args.clip_coef,
            has_continuous_action_space=False
        )
        
        # Training metrics
        self.global_step = 0
        self.start_time = time.time()
        self.episodes_completed = 0
        
    def setup_experiment(self):
        """Set up experiment directory and logging."""
        # Create unique experiment name
        self.run_name = f"{self.args.exp_name}_{self.args.seed}_{int(time.time())}"
        
        # Set up working directory
        self.work_dir = os.path.join("runs", self.run_name)
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger(self.args, self.run_name, self.work_dir)
        
        # Set random seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        
    def collect_rollout(self, next_obs: np.ndarray) -> Dict:
        """Collect a single rollout of experience."""
        episode_stats = {
            'rewards': [],
            'values': [],
            'dones': [],
            'env_infos': []
        }
        
        # Run episode
        done = False
        observation = next_obs
        
        while not done:
            # Select action
            action = self.agent.select_action(observation)
            
            # Execute environment step
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(done)
            
            # Update observation
            observation = next_observation
            
            # Collect statistics
            episode_stats['rewards'].append(reward)
            episode_stats['dones'].append(done)
            episode_stats['env_infos'].append(info)
            
            self.global_step += 1
            
            # Log step information
            if done:
                self.logger.log_episode_info(
                    self.global_step,
                    self.episodes_completed,
                    info,
                    end_of_episode=True
                )
            else:
                self.logger.log_episode_info(
                    self.global_step,
                    self.episodes_completed,
                    info,
                    end_of_episode=False
                )
            
        return episode_stats, observation
    
    def train_iteration(self) -> Dict:
        """Run a single training iteration."""
        iteration_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy_losses': [],
            'clipfracs': []
        }
        
        # Reset environment at the start of iteration
        next_obs, _ = self.env.reset()
        
        # Clear the buffer at the start of each iteration
        self.agent.buffer.clear()
        
        # Collect multiple episodes
        episodes_this_iter = 0
        while episodes_this_iter < self.args.min_episodes_before_update:
            episode_stats, next_obs = self.collect_rollout(next_obs)
            episodes_this_iter += 1
            self.episodes_completed += 1
            
            # Store episode statistics
            iteration_stats['episode_rewards'].append(sum(episode_stats['rewards']))
            
            # Check if we should update
            if len(self.agent.buffer.rewards) >= self.args.batch_size:
                # Perform PPO update
                self.agent.update()
                
                # Clear buffers
                self.agent.buffer.clear()
                
            # Reset environment for next episode
            next_obs, _ = self.env.reset()
        
        return iteration_stats
    
    def train(self):
        """Main training loop."""
        try:
            num_iterations = self.args.total_timesteps // self.args.batch_size
            
            for iteration in range(num_iterations):
                # Training iteration
                iteration_stats = self.train_iteration()
                
                # Periodic evaluation
                if iteration % self.args.eval_frequency == 0:
                    print(f"Evaluating policy at iteration {iteration}")
                    evaluate_policy(env=self.env, agent=self.agent, 
                                 num_episodes=self.args.num_eval_episodes, 
                                 work_dir=self.work_dir, iteration=iteration, 
                                 logger=self.logger)
                
                # Save checkpoint
                if iteration % self.args.save_frequency == 0:
                    print(f"Saving checkpoint at iteration {iteration}")
                    checkpoint_path = os.path.join(
                        self.work_dir, 
                        f"checkpoint_{iteration}.pth"
                    )
                    self.agent.save(checkpoint_path)
                    self.logger.save_checkpoint(checkpoint_path)
                
                # Regenerate environment periodically
                if iteration % self.args.env_reset_frequency == 0:
                    self.env = self.env_manager.create_single_env()
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Cleaning up...")
        finally:
            # Clean up
            self.env.close()
            self.logger.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint."""
        self.agent.load(checkpoint_path)

if __name__ == "__main__":
    # Create Args instance with default values
    args = Args()
    
    # Initialize and run trainer
    trainer = SingleEnvTrainer(args)
    trainer.train()