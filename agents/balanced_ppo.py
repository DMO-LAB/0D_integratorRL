# This file should be added as a new file in your agents directory, e.g., 'balanced_ppo.py'
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from environment.combustion_problem import CombustionStage

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.stages = []  # New field to store stage information
        
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.stages[:]  # Clear stage information


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, deterministic=False):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        if deterministic:
            action = action_probs.argmax(dim=-1)
            #print(f"[ACTOR INFO] - Action: {action} - Action probabilities: {action_probs} - state: {state}")
        else:
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class BalancedPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space=False):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.has_continuous_action_space = has_continuous_action_space
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim).float()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim).float()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, deterministic=False, store_in_buffer=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).float()
            action, action_logprob, state_val = self.policy_old.act(state, deterministic)
        if store_in_buffer:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
        
        return action.item()
    
    def update(self, current_stage=None):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()
        old_state_values = torch.stack(self.buffer.state_values, dim=0).squeeze().detach()

        # If stages are stored, create stage-balanced sampling
        if hasattr(self.buffer, 'stages') and len(self.buffer.stages) > 0:
            # Convert stages to numpy for easier manipulation
            stages = np.array(self.buffer.stages)
            
            # Get indices for each stage
            preignition_indices = np.where(stages == CombustionStage.PREIGNITION.value)[0]
            ignition_indices = np.where(stages == CombustionStage.IGNITION.value)[0]
            postignition_indices = np.where(stages == CombustionStage.POSTIGNITION.value)[0]
            
            # Print stage distribution info
            print(f"Stage distribution in buffer: Pre-ignition: {len(preignition_indices)}, "
                f"Ignition: {len(ignition_indices)}, Post-ignition: {len(postignition_indices)}")
            
            # Calculate weights for oversampling rare stages
            total_samples = len(self.buffer.stages)
            
            # Calculate sample weights - inverse frequency weighting
            weights = np.ones(total_samples)
            
            # Only apply weights if we have samples from at least two stages
            if len(preignition_indices) > 0 and (len(ignition_indices) > 0 or len(postignition_indices) > 0):
                # Set higher weights for ignition and post-ignition states (rare states)
                if len(ignition_indices) > 0:
                    # Weight = total / count (higher weight for rarer stages)
                    ignition_weight = min(100, total_samples / max(1, len(ignition_indices)))
                    weights[ignition_indices] = ignition_weight
                    print(f"Applying weight {ignition_weight:.2f} to ignition samples")
                
                if len(postignition_indices) > 0:
                    postignition_weight = min(50, total_samples / max(1, len(postignition_indices)))
                    weights[postignition_indices] = postignition_weight
                    print(f"Applying weight {postignition_weight:.2f} to post-ignition samples")
            
            # Normalize weights to create a probability distribution
            weights = weights / np.sum(weights)
            
            # Create batch indices with weighted sampling
            batch_indices = np.random.choice(
                np.arange(total_samples), 
                size=min(1024, total_samples),  # Limit batch size
                replace=True,
                p=weights
            )
            
            print(f"Using weighted sampling with {len(np.unique(batch_indices))} unique samples")
        else:
            # If stages not available, use all indices
            batch_indices = np.arange(len(self.buffer.states))
            np.random.shuffle(batch_indices)
            print("Stage information not available, using uniform sampling")
            
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Shuffle batch indices for each epoch
            np.random.shuffle(batch_indices)
            
            # Process in batches to avoid memory issues with very large buffers
            batch_size = 256
            for start_idx in range(0, len(batch_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(batch_indices))
                batch_idx = batch_indices[start_idx:end_idx]
                
                # Create minibatch
                states_batch = old_states[batch_idx]
                actions_batch = old_actions[batch_idx]
                logprobs_batch = old_logprobs[batch_idx]
                rewards_batch = rewards[batch_idx]
                state_values_batch = old_state_values[batch_idx]
                
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(states_batch, actions_batch)
                
                # Finding ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - logprobs_batch.detach())
                
                # Finding Surrogate Loss
                advantages = rewards_batch - state_values_batch.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                # Final loss of clipped objective PPO with higher entropy coefficient
                entropy_coef = 0.05  # Increased from 0.01 to promote exploration
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.squeeze(), rewards_batch) - entropy_coef*dist_entropy
                
                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
    

        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))