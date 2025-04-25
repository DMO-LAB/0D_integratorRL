import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional
import random
from config.default_config import Args
from agents.balanced_ppo import BalancedPPO as PPO
from environment.env_wrapper import EnvManager
from utils.logging_utils import Logger
from utils.evaluation_utils import evaluate_policy,  run_episode
from tqdm import tqdm
from environment.combustion_problem import CombustionStage

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
        
    # Fixed collect_rollout method for rl_trainer.py

    def collect_rollout(self, next_obs: np.ndarray) -> Dict:
        """Collect a single rollout of experience with frame skipping and L2 norm filtering."""
        # Run episode
        dones = False
        observation = next_obs
        
        action_selected = []
        counter = 0
        
        # Keep track of skipped frames
        frame_skip = 1  # Default: don't skip
        pre_ignition_frame_skip = 5  # Skip frames during pre-ignition
        
        # Store the previous observation for L2 norm comparison
        previous_obs = observation.copy()
        previous_temperature = self.env_manager.env.integrator.y[0].copy()

        previous_Y = np.array([
            self.env_manager.env.integrator.y[self.env_manager.env.integrator.gas.species_index(spec) + 1]
            for spec in self.env_manager.env.species_to_track
        ])
        
        # L2 norm threshold for filtering observations (adjust as needed)
        l2_threshold = 0.1  # Minimum change required to consider a state different enough
        
        # Buffer for storing experiences that need to be filtered
        temp_buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'state_values': [],
            'rewards': [],
            'is_terminals': [],
            'stages': []
        }
        
        while not dones:
            # Determine if we should skip this frame based on the stage
            current_stage = self.env_manager.env.integrator.current_stage
            should_skip = False
            
            if current_stage.value == CombustionStage.PREIGNITION.value:
                frame_skip = pre_ignition_frame_skip
                should_skip = (counter % frame_skip != 0)
            else:
                frame_skip = 1  # Don't skip during ignition or post-ignition
                
            # Get current buffer sizes before we add to them
            current_state_size = len(self.agent.buffer.states)
            
            # Select action and store in temporary variables
            with torch.no_grad():
                state = torch.FloatTensor(observation).float()
                action, action_logprob, state_val = self.agent.policy_old.act(state)
                
            action_item = action.item()
            action_selected.append(action_item)

            # Execute environment step
            next_observation, rewards, terminateds, truncateds, infos = self.env_manager.env.step(action_item, timeout=self.args.timeout)
            dones = np.logical_or(terminateds, truncateds)
            #print(f"[{counter}] - TRAINING] Observation: {next_observation} - Rewards: {rewards} - Terminated: {terminateds} - Truncated: {truncateds} - Done: {dones}")
            if dones:
                print(f"[TRAINING] Terminated or truncated at step {self.env_manager.env.current_step} and global step {self.global_step}")
                print(f"[TRAINING] Action distribution: {self.env_manager.env.action_distribution}")
            
            # Calculate L2 norm between current and previous observation
            l2_norm = np.linalg.norm(next_observation - previous_obs)
            significant_change = l2_norm > l2_threshold
            
            # # Log significant changes in pre-ignition for debugging
            # if significant_change and current_stage.value == CombustionStage.PREIGNITION.value:
            #     print(f"[TRAINING] L2 norm: {l2_norm:.4f} - threshold: {l2_threshold} - significant change: {significant_change}")
            
            # Determine whether to store the current experience
            if current_stage.value == CombustionStage.PREIGNITION.value:
                # In pre-ignition: Only store if there's significant change OR it's a regular sample
                should_store = significant_change #or not should_skip
                
            else:
                # In ignition/post-ignition: Always store
                should_store = True
                
            # Always store terminal states
            if dones:
                should_store = True
            
            if should_store:
                # print the before and after obs
                print(f"Counter: {counter} - L2 norm: {l2_norm:.4f} - significant change: {significant_change} - current stage: {current_stage.value} \n previous temperature: {previous_temperature} - current temperature: {self.env_manager.env.integrator.y[0]}")
                current_Y = np.array([
                    self.env_manager.env.integrator.y[self.env_manager.env.integrator.gas.species_index(spec) + 1]
                    for spec in self.env_manager.env.species_to_track
                ])
                print(f"Action taken: {action_item} - Rewards: {rewards} - cpu time: {infos['cpu_time']} - error: {infos['error']}")
                # print(f"Previous Y: {previous_Y} - Current Y: {current_Y}")
                # Store experiences directly in the PPO buffer
                self.agent.buffer.states.append(state)
                self.agent.buffer.actions.append(action)
                self.agent.buffer.logprobs.append(action_logprob)
                self.agent.buffer.state_values.append(state_val)
                self.agent.buffer.rewards.append(rewards)
                self.agent.buffer.is_terminals.append(dones)
                self.agent.buffer.stages.append(current_stage.value)
                
                self.global_step += 1
                
                # Log step information
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
                    
                # # Log when we're storing data from pre-ignition
                # if current_stage.value == CombustionStage.PREIGNITION.value:
                #     if significant_change:
                #         print(f"[TRAINING - {counter}] Storing significant pre-ignition change - L2 norm: {l2_norm:.4f}")
                #     elif not should_skip:
                #         print(f"[TRAINING - {counter}] Storing regular pre-ignition sample (not skipped)")
            # else:
            #     print(f"[TRAINING] Skipping pre-ignition sample - L2 norm: {l2_norm:.4f} - frame: {counter}")
                
            # Always update observation (even when skipping)
            previous_obs = next_observation.copy()
            observation = next_observation
            previous_temperature = self.env_manager.env.integrator.y[0].copy()
            previous_Y = np.array([
            self.env_manager.env.integrator.y[self.env_manager.env.integrator.gas.species_index(spec) + 1]
            for spec in self.env_manager.env.species_to_track
        ])
            counter += 1
            
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
                print(f"[TRAINING] Performing PPO update at iteration {self.global_step} - with {len(self.agent.buffer.rewards)} rewards and buffer size {len(self.agent.buffer.states)}")
                # Perform PPO update
                self.agent.update()
                
                # Clear buffers
                self.agent.buffer.clear()
                print(f"[TRAINING] Buffer cleared at iteration {self.global_step} - with {len(self.agent.buffer.rewards)} rewards and buffer size {len(self.agent.buffer.states)}")
                
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
                    print(f"[TRAINING] Regenerating environments at iteration {iteration}")
                    # plot the reference solution
                    self.env_manager.env.problem.plot_reference_solution(save_path=os.path.join(self.work_dir, f"reference_solution_{iteration}.png"))
                    print(f"[TRAINING] Reference solution plotted at iteration {iteration}  saved at {self.work_dir}/reference_solution_{iteration}.png")
                
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