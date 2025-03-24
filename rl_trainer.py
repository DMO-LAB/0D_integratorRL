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
from utils.evaluation_utils import evaluate_policy,  run_episode
from tqdm import tqdm

class Trainer:
    """Trainer class for PPO algorithm on combustion environments."""
    
    def __init__(self, args):
        """Initialize trainer with configuration."""
        self.args = args
        self.setup_experiment()
        
        # Initialize environment
        self.env_manager = EnvManager(args, single_env=True)
        
        # Get environment dimensions
        env = self.env_manager.env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
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
        
        model_path = "pretraining_results/models/best_pretrained_actor.pth"
        actor_state_dict = torch.load(model_path)

        print(f"Loading actor state dict from {model_path}")
        self.agent.policy.actor.load_state_dict(actor_state_dict)

        
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
        # Run episode
        dones = False
        observation = next_obs
        
        action_selected = []
        
        while not dones:
            # Select action
            action = self.agent.select_action(observation)
            action_selected.append(action)
            # Execute environment step
            next_observation, rewards, terminateds, truncateds, infos = self.env_manager.env.step(action, timeout=self.args.timeout)
            dones = np.logical_or(terminateds, truncateds)
            
            if dones:
                print(f"[TRAINING] Terminated or truncated at step {self.env_manager.env.current_step} and global step {self.global_step}")
                # print the action distribution
                print(f"[TRAINING] Action distribution: {self.env_manager.env.action_distribution}")
            
            # Store experience for each environment
            self.agent.buffer.rewards.append(rewards)
            self.agent.buffer.is_terminals.append(dones)
            # Update observation
            observation = next_observation
            
            if len(self.agent.buffer.rewards) != len(self.agent.buffer.state_values):
                print(f"Reward Buffer size: {len(self.agent.buffer.rewards)} - State Value Buffer size: {len(self.agent.buffer.state_values)}")
                    
            self.global_step += 1
            
            # Log step information for each environment
            if 'final_info' in infos and infos['final_info'] is not None:
                env_info = infos['final_info']
                self.logger.log_episode_info(
                    self.global_step,
                    self.episodes_completed,
                    env_info,
                    end_of_episode=True
                )
            elif not dones:
                self.logger.log_episode_info(
                    self.global_step,
                    self.episodes_completed,
                    infos,
                    end_of_episode=False
                )
            
        action_selected = np.array(action_selected)
        return observation, action_selected
    
    def train_iteration(self) -> Dict:
        """Run a single training iteration."""
        
        # Reset environments at the start of iteration
        next_obs, _ = self.env_manager.env.reset()
        
        # Clear the buffer at the start of each iteration
        self.agent.buffer.clear()
        
        # Collect multiple episodes
        episodes_this_iter = 0
        while episodes_this_iter < self.args.min_episodes_before_update:
            next_obs, action_selected = self.collect_rollout(next_obs)
            episodes_this_iter += 1
            self.episodes_completed += 1
            # Check if we should update
            if len(self.agent.buffer.rewards) >= self.args.batch_size:
                # Perform PPO update
                self.agent.update()
                
                # Clear buffers
                self.agent.buffer.clear()
                
            next_obs, _ = self.env_manager.env.reset()
        return next_obs

    def train(self):
        """Main training loop."""
        try:
            num_iterations = self.args.total_timesteps // self.args.batch_size
            
            for iteration in tqdm(range(num_iterations), desc="Training", total=num_iterations, leave=False):
                # Training iteration
                next_obs = self.train_iteration()
                
                # Periodic evaluation
                if iteration % self.args.eval_frequency == 0:
                    print(f"[TRAINING] Evaluating policy at iteration {iteration}")
                    evaluate_policy(env=self.env_manager.env, agent=self.agent, num_episodes=self.args.num_eval_episodes, work_dir=self.work_dir, iteration=iteration, logger=self.logger)
                
                # Save checkpoint
                if iteration % self.args.save_frequency == 0:
                    print(f"Saving checkpoint at iteration {iteration}")
                    checkpoint_path = os.path.join(
                        self.work_dir, 
                        f"checkpoint_{iteration}.pth"
                    )
                    self.agent.save(checkpoint_path)
                    self.logger.save_checkpoint(checkpoint_path)
                
                # Regenerate environments periodically
                if iteration % self.args.env_reset_frequency == 0:
                    self.env_manager.generate_environments(single_env=True)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Cleaning up...")
        finally:
            # Clean up
            self.env_manager.close()
            self.logger.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint."""
        self.agent.load(checkpoint_path)
        
if __name__ == "__main__":
    from dataclasses import dataclass
    
    # Create Args instance with default values
    args = Args()
    
    # Initialize and run trainer
    trainer = Trainer(args)
    trainer.train()