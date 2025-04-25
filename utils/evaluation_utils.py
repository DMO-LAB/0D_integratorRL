import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
import wandb
from datetime import datetime
import json
from utils.logging_utils import Logger

def evaluate_policy(env, agent, num_episodes: int = 4, work_dir: str = 'evaluation', 
                   fixed_actions: Optional[List[int]] = None, iteration: int = 0, logger: Logger = None, device: torch.device = None) -> Dict[str, Any]:
    """
    Evaluate a policy and compare it with fixed actions, generating comprehensive metrics and visualizations.
    """
    print(f"[EVALUATION INFO] - Temperature: {env.problem.temperature} - phi: {env.problem.phi} - pressure: {env.problem.pressure} -end time: {env.problem.end_time}")
    os.makedirs(work_dir, exist_ok=True)
    work_dir = work_dir + '/evaluation_plots'
    os.makedirs(work_dir, exist_ok=True)
    
    if fixed_actions is None:
        fixed_actions = [i for i in range(len(env.integrator.action_list))]
        fixed_actions.append(None)

    # Run comparison with fixed actions
    comparison_results, good_policy = compare_fixed_actions(env, agent, fixed_actions, device, work_dir, iteration)
    
    # Generate and save comparison plots
    plot_path = generate_comparison_plots(comparison_results, work_dir, iteration, logger, action_list=env.integrator.action_list)

    return good_policy

def run_episode(agent, env, default_action: Optional[int] = None, device: torch.device = None, work_dir: str = None, iteration: int = 0) -> Tuple[float, float, float, float, float, Any]:
    """
    Run a single episode with either a fixed action or the policy.
    """
    if default_action is None:
        print(f"[EVALUATION] Running episode with RL policy")
    else:
        print(f"[EVALUATION] Running episode with fixed action {default_action}")
        
    obs, _ = env.reset()
    done = False
    total_reward = 0
    total_error = 0
    total_time = 0
    total_time_reward = 0
    total_error_reward = 0
    
    while not done:
        if default_action is not None:
            action = default_action
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(obs_tensor, deterministic=True, store_in_buffer=False)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if info['success']:
            total_reward += reward
            total_error += info['error']
            total_time += info['cpu_time']
            total_time_reward += info['time_reward']
            total_error_reward += info['error_reward']
        else:
            print(f"Episode failed - {info}")
    
    try:
        print(f"Total reward: {total_reward:.2f} - Total error: {total_error:.4f} - Total time: {total_time:.2f} - Total time reward: {total_time_reward:.2f} - Total error reward: {total_error_reward:.2f}")
    except Exception as e:
        pass
    
    if default_action is None:
        print(f"[EVALUATION] RL POLICY EVALUATION - Reward: {total_reward:.2f} - CPU Time: {total_time:.2f} - Error: {total_error:.4f}")
        print(f"[EVALUATION] Action Distribution: {env.action_distribution}")
        # if work_dir is not None:
        #     plot_path = work_dir + f'/rl_policy_evaluation_{iteration}.png'
        #     env.render('human')
        #     plt.savefig(plot_path)
        #     plt.close()
        
    return total_reward, total_error, total_time, total_time_reward, total_error_reward, env

def compare_fixed_actions(env, agent, fixed_actions: List[int], device: torch.device, work_dir: str, iteration: int) -> Dict[str, Any]:
    """
    Run evaluation episodes with fixed actions for comparison.
    """
    results = {
        'actions': [],
        'rewards': [],
        'errors': [],
        'cpu_times': [],
        'time_rewards': [],
        'error_rewards': [],
        'action_distributions': [],
        'statistics': []
    }
    
    for action in fixed_actions:
        try:
            reward, error, cpu_time, time_reward, error_reward, env_state = run_episode(
                agent=agent,
                env=env,
                default_action=action,
                device=device, 
                work_dir=work_dir,
                iteration=iteration
            )
            
            results['actions'].append(action)
            results['rewards'].append(env_state.episode_rewards)
            results['errors'].append(error)
            results['cpu_times'].append(cpu_time)
            results['time_rewards'].append(env_state.episode_time_rewards)
            results['error_rewards'].append(env_state.episode_error_rewards)
            results['statistics'].append(env_state.integrator.get_statistics())
            results['action_distributions'].append(
                env_state.integrator.history.get('actions_taken', [])
            )
            
            # Save plot for this action
            if work_dir is not None :
                if action is None:
                    plot_path = work_dir + f'/rl_policy_evaluation_{iteration}.png'
                else:
                    plot_path = work_dir + f'/fixed_action_{action}_evaluation_{iteration}.png'
                env_state.render(plot_path)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running fixed action {action}: {e}")
            continue

    # print(f"Results: {results}")
    # check if the last action has the best cpu time
    min_cpu_time = min(results['cpu_times'])
    best_action = results['actions'][results['cpu_times'].index(min_cpu_time)]
    if best_action == fixed_actions[-1]:
        print(f"RL POLICY IS BETTER THAN FIXED ACTIONS - {min_cpu_time:.4f}s cpu time compared to {results['cpu_times']}")
        good_policy = True
    else:
        print(f"FIXED ACTION {best_action} IS BETTER THAN RL POLICY - {min_cpu_time:.4f}s cpu time compared to {results['cpu_times']}")
        good_policy = False
    
    return results, good_policy

def generate_comparison_plots(comparison_results: Dict[str, List],
                            work_dir: str, iteration: int, logger: Logger, action_list: List[str]) -> str:
    """
    Generate and save comparison plots between policy and fixed actions.
    """
    plots_dir = work_dir 

    # Prepare data for plotting
    all_actions = [get_action_name(action_list, comparison_results['actions'][i]) for i in range(len(comparison_results['actions']))]
    
    # Plot 1: Action distributions and metrics
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20), dpi=300)
    
    # Action distributions
    for i, dist in enumerate(comparison_results['action_distributions']):
        if i == len(comparison_results['action_distributions']) - 1:  # Last one is RL policy
            ax1.plot(dist, label='RL Policy', marker='o', markersize=5, color='black')
        else:
            ax1.plot(dist, label=get_action_name(action_list, comparison_results['actions'][i]))
            
    ax1.legend(loc='upper right', frameon=True, framealpha=0.5, fontsize=6)
    ax1.set_title('Action Distribution')
    
    # CPU Times
    all_times = comparison_results['cpu_times']
    ax2.bar(range(len(all_actions)), all_times)
    ax2.set_xticks(range(len(all_actions)))
    ax2.set_xticklabels(all_actions, rotation=45)
    for i, v in enumerate(all_times):
        ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    ax2.set_title('Average CPU Time')
    
    # Errors
    all_errors = comparison_results['errors']
    ax3.bar(range(len(all_actions)), all_errors)
    ax3.set_xticks(range(len(all_actions)))
    ax3.set_xticklabels(all_actions, rotation=45)
    for i, v in enumerate(all_errors):
        ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax3.set_title('Average Error')
    
    plot_path = os.path.join(plots_dir, f'comparison_metrics_{iteration}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    if logger:
        logger.log_comparison_plot(plot_path, iteration)
    plt.close()

    # Generate additional plots (rewards comparison and time series)
    generate_rewards_plot(comparison_results, all_actions, plots_dir, iteration, logger)
    generate_timeseries_plot(comparison_results, all_actions, plots_dir, iteration, logger)
    
    return plots_dir

def generate_rewards_plot(comparison_results: Dict[str, List], all_actions: List[str], 
                        plots_dir: str, iteration: int, logger: Logger):
    """Generate rewards comparison plot."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20), dpi=300)
    
    # Total rewards
    all_rewards = [np.sum(r) for r in comparison_results['rewards']]
    ax1.bar(range(len(all_actions)), all_rewards)
    ax1.set_xticks(range(len(all_actions)))
    ax1.set_xticklabels(all_actions, rotation=45)
    for i, v in enumerate(all_rewards):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax1.set_title('Total Reward')
    
    # Error rewards
    all_error_rewards = [np.sum(r) for r in comparison_results['error_rewards']]
    ax2.bar(range(len(all_actions)), all_error_rewards)
    ax2.set_xticks(range(len(all_actions)))
    ax2.set_xticklabels(all_actions, rotation=45)
    for i, v in enumerate(all_error_rewards):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax2.set_title('Error Reward')
    
    # Time rewards
    all_time_rewards = [np.sum(r) for r in comparison_results['time_rewards']]
    ax3.bar(range(len(all_actions)), all_time_rewards)
    ax3.set_xticks(range(len(all_actions)))
    ax3.set_xticklabels(all_actions, rotation=45)
    for i, v in enumerate(all_time_rewards):
        ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax3.set_title('Time Reward')
    
    rewards_plot_path = os.path.join(plots_dir, f'reward_comparison_{iteration}.png')
    plt.tight_layout()
    plt.savefig(rewards_plot_path)
    if logger:
        logger.log_comparison_plot(rewards_plot_path, iteration)
    plt.close()

def generate_timeseries_plot(comparison_results: Dict[str, List], all_actions: List[str], 
                           plots_dir: str, iteration: int, logger: Logger):
    """Generate time series comparison plot."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20), dpi=300)

    # Plot rewards over time
    for i, rewards in enumerate(comparison_results['rewards']):
        ax1.plot(rewards, label=all_actions[i])
    ax1.set_title('Reward Over Time')
    ax1.legend(fontsize=8)
    
    # Error rewards over time
    for i, errors in enumerate(comparison_results['error_rewards']):
        ax2.plot(errors, label=all_actions[i])
    ax2.set_title('Error Reward Over Time')
    ax2.legend(fontsize=8)
    
    # Time rewards over time
    for i, times in enumerate(comparison_results['time_rewards']):
        ax3.plot(times, label=all_actions[i])
    ax3.set_title('Time Reward Over Time')
    ax3.legend(fontsize=8)
    
    timeseries_path = os.path.join(plots_dir, f'timeseries_comparison_{iteration}.png')
    plt.tight_layout()
    plt.savefig(timeseries_path)
    if logger:
        logger.log_comparison_plot(timeseries_path, iteration)
    plt.close()

def get_action_name(action_list: List[str], action: Optional[int] = None) -> str:
    """Get the name of an action for plotting."""
    if action is None:
        return 'RL Policy'
    else:
        return str(action_list[action]) if isinstance(action_list, list) else str(action)