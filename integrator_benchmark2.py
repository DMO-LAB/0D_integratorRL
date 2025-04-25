import os
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import argparse
import random
from pathlib import Path

from environment.env_wrapper import EnvManager
from environment.combustion_problem import CombustionProblem, setup_problem, CombustionStage
from environment.modified_integrator import ChemicalIntegrator, IntegratorConfig
from config.default_config import Args

class IntegratorBenchmark:
    """Benchmark different integrators on combustion problems with various conditions."""
    
    def __init__(self, args, temperature=None, pressure=None, phi=None, 
                 fixed_dt=1e-5, end_time=1e-3, timeout=5.0, reference_method="CVODE_BDF",
                 metric="reward", output_dir="benchmark_results"):
        """
        Initialize benchmark tool.
        
        Args:
            args: Configuration arguments
            temperature: Temperature for the problem (if None, will be randomly sampled)
            pressure: Pressure for the problem (if None, will be randomly sampled)
            phi: Equivalence ratio for the problem (if None, will be randomly sampled)
            fixed_dt: Fixed timestep for the problem
            end_time: End time for simulation
            timeout: Timeout for each integration step
            reference_method: Method to use as reference for states (default: CVODE_BDF)
            metric: Metric to use for determining best integrator ("cpu_time", "error", or "reward")
            output_dir: Directory to save results
        """
        self.args = args
        self.temperature = temperature
        self.pressure = pressure
        self.phi = phi
        self.fixed_dt = fixed_dt
        self.end_time = end_time
        self.timeout = timeout
        self.reference_method = reference_method
        self.metric = metric
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize environments
        self.env_manager = EnvManager(args, single_env=False, create_envs=False)
        
        # Get list of all actions
        self.integrator_config = IntegratorConfig(
            integrator_list=self.args.integrator_list,
            tolerance_list=self.args.tolerance_list
        )
        self.action_list = self.integrator_config.get_action_list()
        print(f"Available actions: {self.action_list}")
        
        # Dictionary to store environments for each action
        self.envs = {}
        
        # Ensure reference method is in the action list
        reference_found = False
        for i, (method, rtol, atol) in enumerate(self.action_list):
            if method == self.reference_method:
                self.reference_action_idx = i
                reference_found = True
                break
                
        if not reference_found:
            print(f"WARNING: Reference method {self.reference_method} not found in action list. Using first method as reference.")
            self.reference_action_idx = 0
            
        print(f"Using action {self.action_list[self.reference_action_idx]} as reference.")
        
    def setup_environments(self):
        """Create environments for each action."""
        print("Setting up environments...")
        
        # Use provided values or randomly sample if None
        if self.temperature is None:
            self.temperature = np.random.choice(self.args.temperature_range)
        if self.pressure is None:
            self.pressure = np.random.choice(self.args.pressure_range)
        if self.phi is None:
            self.phi = np.random.choice(self.args.phi_range)
            
        print(f"Problem conditions: T={self.temperature} K, P={self.pressure} atm, phi={self.phi}")
        
        # First, create the reference environment
        ref_method, ref_rtol, ref_atol = self.action_list[self.reference_action_idx]
        
        # Create a modified args with only the reference method
        ref_args = Args()
        ref_args.integrator_list = [ref_method]
        ref_args.tolerance_list = [(ref_rtol, ref_atol)]
        
        # Create reference environment
        self.reference_env = self.env_manager.create_single_env(
            end_time=self.end_time,
            fixed_temperature=self.temperature,
            fixed_pressure=self.pressure,
            fixed_phi=self.phi,
            fixed_dt=self.fixed_dt,
            randomize=False
        )
        
        # Now create an environment for each action
        for i, (method, rtol, atol) in enumerate(self.action_list):
            if i == self.reference_action_idx:
                self.envs[i] = self.reference_env
            else:
                # Create a modified args with only this method
                action_args = Args()
                action_args.integrator_list = [method]
                action_args.tolerance_list = [(rtol, atol)]
                
                # Create environment with same problem as reference
                env = self.env_manager.create_single_env(
                    end_time=self.end_time,
                    fixed_temperature=self.temperature,
                    fixed_pressure=self.pressure,
                    fixed_phi=self.phi,
                    fixed_dt=self.fixed_dt,
                    randomize=False
                )
                
                self.envs[i] = env
                
        print(f"Created {len(self.envs)} environments.")
        
    def run_benchmark(self, max_steps=100):
        """Run benchmark on all actions."""
        # Create condition-specific output directory
        condition_dir = os.path.join(self.output_dir, f"T{self.temperature:.1f}_P{self.pressure:.1f}_phi{self.phi:.2f}")
        os.makedirs(condition_dir, exist_ok=True)
        
        print(f"Running benchmark with metric: {self.metric}")
        print(f"Results will be saved to: {condition_dir}")
        
        # Initialize result storage
        results = {
            'step': [],
            'temperature': [],
            'stage': [],
            'best_action': []
        }

        # Create a dictionary to store the done status of each action
        actions_dones = {
            i: False for i, (method, rtol, atol) in enumerate(self.action_list)
        }
        
        # Add columns for each action
        for i, (method, rtol, atol) in enumerate(self.action_list):
            action_name = f"{method}_{rtol}_{atol}"
            results[f"{action_name}_cpu_time"] = []
            results[f"{action_name}_error"] = []
            results[f"{action_name}_reward"] = []
            results[f"{action_name}_success"] = []
        
        # Add columns for observation elements from the reference environment
        reference_env = self.envs[self.reference_action_idx]
        observation_dim = reference_env.observation_space.shape[0]
        for i in range(observation_dim):
            results[f'obs_{i}'] = []
        
        # Reset all environments to ensure same initial state
        for i, env in self.envs.items():
            observation, _ = env.reset()
            
            # Store initial observation from reference environment
            if i == self.reference_action_idx:
                for j in range(len(observation)):
                    results[f'obs_{j}'].append(observation[j])
        
        # Run steps
        for step in tqdm(range(max_steps), desc="Running benchmark"):
            
            # Get state from reference environment
            reference_temp = self.envs[self.reference_action_idx].integrator.y[0]
            reference_stage = self.envs[self.reference_action_idx].integrator.current_stage
            
            # Store step data
            results['step'].append(step)
            results['temperature'].append(reference_temp)
            results['stage'].append(reference_stage.name)
            
            # Dictionary to store metrics for determining best action
            step_metrics = {}
            
            
            
            # Take a step with each action
            for i, (method, rtol, atol) in enumerate(self.action_list):
                action_name = f"{method}_{rtol}_{atol}"
                
                
                # Take a step
                try:
                    if actions_dones[i]:
                        print(f"Skipping environment {i} ({action_name}) because it terminated at step {step}")
                        raise 
                    env = self.envs[i]
                    obs, reward, terminated, truncated, info = env.step(i, timeout=self.timeout)  # Action 0 corresponds to the only method for this env
                    
                    # Store metrics
                    cpu_time = info.get('cpu_time', float('inf'))
                    error = info.get('error', float('inf'))
                    success = info.get('success', False)

                    #print(f"Action {i} ({action_name}) - cpu_time: {cpu_time:.6f} - error: {error:.6e} - reward: {reward:.4f} - success: {success}")
                    
                    results[f"{action_name}_cpu_time"].append(cpu_time)
                    results[f"{action_name}_error"].append(error)
                    results[f"{action_name}_reward"].append(reward)
                    results[f"{action_name}_success"].append(success)
                    
                    # Store observation data from reference environment only
                    if i == self.reference_action_idx:
                        for j in range(len(obs)):
                            # Make sure we add to results only if we're past the first step
                            # (which was already added during reset)
                            if step > 0 or j >= len(results[f'obs_{j}']):
                                results[f'obs_{j}'].append(obs[j])
                    
                    # Store metric for determining best action
                    if self.metric == "cpu_time":
                        step_metrics[i] = cpu_time if success else float('inf')
                    elif self.metric == "error":
                        step_metrics[i] = error if success else float('inf')
                    else:  # "reward"
                        step_metrics[i] = -reward if success else float('-inf')  # Negative because we want to maximize reward
                    
                    if terminated or truncated:
                        print(f"Environment {i} ({action_name}) terminated at step {step}")
                        # For terminated environments, set future metrics to infinity
                        actions_dones[i] = True
                except Exception as e:
                    # import traceback
                    # traceback.print_exc()
                    #print(f"Exception raise for environment {i} ({action_name}) at step {step} - {e}")
                    results[f"{action_name}_cpu_time"].append(float('inf'))
                    results[f"{action_name}_error"].append(float('inf'))
                    results[f"{action_name}_reward"].append(float('-inf'))
                    results[f"{action_name}_success"].append(False)
                    step_metrics[i] = float('inf')
                    actions_dones[i] = True
                    
            # Determine best action
            if step_metrics:
                best_action = min(step_metrics, key=step_metrics.get)
                results['best_action'].append(best_action)
                #print(f"Best action at step {step}: {best_action} ({self.action_list[best_action][0]})")
            else:
                results['best_action'].append(-1)  # No valid actions
                
            # Check if all environments are done
            all_done = True
            for i, env in self.envs.items():
                if not actions_dones[i] and not env.integrator.end_simulation:
                    all_done = False
                    break
                    
            if all_done:
                print(f"All environments completed at step {step}")
                break
        
        # Ensure all lists in the results dictionary have the same length
        max_length = max(len(val) for val in results.values())
        for key in results:
            if len(results[key]) < max_length:
                # Pad shorter lists with NaN values
                results[key].extend([float('nan')] * (max_length - len(results[key])))
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(condition_dir, f"benchmark_results_{self.metric}.csv")
        df.to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")
        
        # Generate plots
        self.generate_plots(df, condition_dir)
        
        # Generate summary statistics
        self.generate_summary_stats(df, condition_dir)
        
        return df, condition_dir
                    
    
    def generate_plots(self, df, output_dir):
        """Generate benchmark plots."""
        # Create a directory for plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot temperature evolution
        plt.figure(figsize=(12, 6))
        plt.plot(df['step'], df['temperature'])
        plt.xlabel('Step')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature Evolution (T={self.temperature:.1f}K, phi={self.phi:.2f})')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "temperature_evolution.png"))
        plt.close()
        
        # Plot best action distribution
        action_counts = df['best_action'].value_counts().sort_index()
        
        labels = [f"{self.action_list[int(i)]}" 
                 for i in action_counts.index if i >= 0]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, action_counts.values)
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.title(f'Best Action Distribution (T={self.temperature:.1f}K, phi={self.phi:.2f}, Metric: {self.metric})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"best_action_distribution_{self.metric}.png"))
        plt.close()
        
        # Plot metric over time for each action
        plt.figure(figsize=(15, 8))
        
        for i, (method, rtol, atol) in enumerate(self.action_list):
            action_name = f"{method}_{rtol}_{atol}"
            
            if self.metric == "cpu_time":
                y = df[f"{action_name}_cpu_time"]
                title = f"CPU Time Over Steps (T={self.temperature:.1f}K, phi={self.phi:.2f})"
                ylabel = "CPU Time (s)"
            elif self.metric == "error":
                y = df[f"{action_name}_error"]
                title = f"Error Over Steps (T={self.temperature:.1f}K, phi={self.phi:.2f})"
                ylabel = "Error"
            else:  # "reward"
                y = df[f"{action_name}_reward"]
                title = f"Reward Over Steps (T={self.temperature:.1f}K, phi={self.phi:.2f})"
                ylabel = "Reward"
                
            # Replace inf with NaN for plotting
            y = y.replace([float('inf'), float('-inf')], np.nan)
            
            plt.plot(df['step'], y, label=action_name)
            
        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{self.metric}_over_steps.png"))
        plt.close()
        
        # Plot success rate over all steps
        success_rates = {}
        for i, (method, rtol, atol) in enumerate(self.action_list):
            action_name = f"{method}_{rtol}_{atol}"
            success_rates[action_name] = df[f"{action_name}_success"].mean() * 100
            
        sorted_actions = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
        actions = [a[0] for a in sorted_actions]
        rates = [a[1] for a in sorted_actions]
        
        plt.figure(figsize=(12, 6))
        plt.bar(actions, rates)
        plt.xlabel('Action')
        plt.ylabel('Success Rate (%)')
        plt.title(f'Success Rate by Action (T={self.temperature:.1f}K, phi={self.phi:.2f})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "success_rate.png"))
        plt.close()
        
        # Plot best action by combustion stage
        stage_best_actions = df.groupby('stage')['best_action'].value_counts().unstack().fillna(0)
        
        if not stage_best_actions.empty:
            plt.figure(figsize=(12, 8))
            stage_best_actions.plot(kind='bar', stacked=True)
            plt.xlabel('Combustion Stage')
            plt.ylabel('Count')
            plt.title(f'Best Action by Combustion Stage (T={self.temperature:.1f}K, phi={self.phi:.2f}, Metric: {self.metric})')
            plt.legend(title='Action Index')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"best_action_by_stage_{self.metric}.png"))
            plt.close()
        
    def generate_summary_stats(self, df, output_dir):
        """Generate summary statistics."""
        summary_path = os.path.join(output_dir, "summary_statistics.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Integrator Benchmark Summary (Metric: {self.metric})\n")
            f.write("="*80 + "\n\n")
            
            f.write("Problem Parameters:\n")
            f.write(f"  Temperature: {self.temperature} K\n")
            f.write(f"  Pressure: {self.pressure} atm\n")
            f.write(f"  Phi: {self.phi}\n")
            f.write(f"  Timestep: {self.fixed_dt}\n")
            f.write(f"  End Time: {self.end_time}\n\n")
            
            f.write("Action Statistics:\n")
            
            # Calculate statistics for each action
            for i, (method, rtol, atol) in enumerate(self.action_list):
                action_name = f"{method}_{rtol}_{atol}"
                f.write(f"  Action {i}: {action_name}\n")
                
                # CPU time stats
                cpu_times = df[f"{action_name}_cpu_time"].replace([float('inf')], np.nan)
                if not cpu_times.isnull().all():
                    f.write(f"    Avg CPU Time: {cpu_times.mean():.6f} s\n")
                    f.write(f"    Min CPU Time: {cpu_times.min():.6f} s\n")
                    f.write(f"    Max CPU Time: {cpu_times.max():.6f} s\n")
                else:
                    f.write("    CPU Time: All steps failed\n")
                
                # Error stats
                errors = df[f"{action_name}_error"].replace([float('inf')], np.nan)
                if not errors.isnull().all():
                    f.write(f"    Avg Error: {errors.mean():.6e}\n")
                    f.write(f"    Min Error: {errors.min():.6e}\n")
                    f.write(f"    Max Error: {errors.max():.6e}\n")
                else:
                    f.write("    Error: All steps failed\n")
                
                # Reward stats
                rewards = df[f"{action_name}_reward"].replace([float('-inf')], np.nan)
                if not rewards.isnull().all():
                    f.write(f"    Avg Reward: {rewards.mean():.4f}\n")
                    f.write(f"    Min Reward: {rewards.min():.4f}\n")
                    f.write(f"    Max Reward: {rewards.max():.4f}\n")
                else:
                    f.write("    Reward: All steps failed\n")
                
                # Success rate
                success_rate = df[f"{action_name}_success"].mean() * 100
                f.write(f"    Success Rate: {success_rate:.2f}%\n")
                
                # Best action count
                best_count = (df['best_action'] == i).sum()
                best_pct = best_count / len(df) * 100 if len(df) > 0 else 0
                f.write(f"    Best Action Count: {best_count} ({best_pct:.2f}%)\n\n")
            
            # Stage-based statistics
            f.write("Stage-Based Statistics:\n")
            stages = df['stage'].unique()
            
            for stage in stages:
                stage_df = df[df['stage'] == stage]
                f.write(f"  Stage: {stage}\n")
                
                # Best action distribution in this stage
                stage_best = stage_df['best_action'].value_counts().sort_index()
                for action_idx, count in stage_best.items():
                    if action_idx >= 0:  # Skip invalid actions
                        action_name = f"{self.action_list[int(action_idx)][0]}_{self.action_list[int(action_idx)][1]}_{self.action_list[int(action_idx)][2]}"
                        pct = count / len(stage_df) * 100
                        f.write(f"    Best Action: {action_name} - {count} times ({pct:.2f}%)\n")
                
                f.write("\n")
            
            # Overall best action
            if len(df) > 0 and df['best_action'].value_counts().size > 0:
                best_overall = df['best_action'].value_counts().sort_values(ascending=False).index[0]
                if best_overall >= 0:
                    best_name = f"{self.action_list[int(best_overall)][0]}_{self.action_list[int(best_overall)][1]}_{self.action_list[int(best_overall)][2]}"
                    best_count = (df['best_action'] == best_overall).sum()
                    best_pct = best_count / len(df) * 100
                    f.write(f"Overall Best Action: {best_name} - {best_count} times ({best_pct:.2f}%)\n")
                else:
                    f.write("No valid best action found.\n")
            else:
                f.write("No data available for best action analysis.\n")
        
        print(f"Generated summary statistics at {summary_path}")


def run_multi_condition_benchmark(args, temperature_range, phi_range, n_samples=5, 
                                 pressure=1.0, dt=1e-5, end_time=1e-3, timeout=5.0, 
                                 reference_method="CVODE_BDF", metric="reward", 
                                 output_dir="benchmark_results", max_steps=100):
    """
    Run benchmarks for multiple randomly sampled conditions.
    
    Args:
        args: Configuration arguments
        temperature_range: Array of temperatures to sample from
        phi_range: Array of phi values to sample from
        n_samples: Number of random condition samples to run
        pressure: Pressure value (constant)
        dt: Timestep
        end_time: End time for simulation
        timeout: Timeout for each step
        reference_method: Reference method to use
        metric: Metric for determining best integrator
        output_dir: Base output directory
        max_steps: Maximum steps per benchmark
    """
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(output_dir, f"benchmark_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save parameters
    with open(os.path.join(base_output_dir, "parameters.txt"), "w") as f:
        f.write(f"Temperature Range: {temperature_range.min()}-{temperature_range.max()} K\n")
        f.write(f"Phi Range: {phi_range.min()}-{phi_range.max()}\n")
        f.write(f"Number of Samples: {n_samples}\n")
        f.write(f"Pressure: {pressure} atm\n")
        f.write(f"Timestep: {dt} s\n")
        f.write(f"End Time: {end_time} s\n")
        f.write(f"Timeout: {timeout} s\n")
        f.write(f"Reference Method: {reference_method}\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Max Steps: {max_steps}\n")
    
    # Sample conditions
    temperatures = np.random.choice(temperature_range, size=n_samples)
    phis = np.random.choice(phi_range, size=n_samples)
    
    # Create DataFrame to store summary results
    summary_results = {
        "temperature": [],
        "phi": [],
        "condition_dir": [],
        "best_overall_action": [],
        "best_preignition_action": [],
        "best_ignition_action": [],
        "best_postignition_action": []
    }
    
    # Add columns for each action's success rate
    action_list = IntegratorConfig(
        integrator_list=args.integrator_list,
        tolerance_list=args.tolerance_list
    ).get_action_list()
    
    for method, rtol, atol in action_list:
        action_name = f"{method}_{rtol}_{atol}"
        summary_results[f"{action_name}_success_rate"] = []
        summary_results[f"{action_name}_avg_cpu_time"] = []
        summary_results[f"{action_name}_avg_error"] = []
        summary_results[f"{action_name}_avg_reward"] = []
    
    # Run benchmarks for each sampled condition
    for i, (temp, phi) in enumerate(zip(temperatures, phis)):
        print(f"\n\n{'='*80}")
        print(f"Running benchmark {i+1}/{n_samples}: T={temp:.1f}K, phi={phi:.2f}")
        print(f"{'='*80}\n")
        
        # Create and run benchmark
        benchmark = IntegratorBenchmark(
            args=args,
            temperature=temp,
            pressure=pressure,
            phi=phi,
            fixed_dt=dt,
            end_time=end_time,
            timeout=timeout,
            reference_method=reference_method,
            metric=metric,
            output_dir=base_output_dir
        )
        
        # Setup environments
        benchmark.setup_environments()
        
        # Run benchmark
        df, condition_dir = benchmark.run_benchmark(max_steps=max_steps)
        
        # Extract and store summary data
        summary_results["temperature"].append(temp)
        summary_results["phi"].append(phi)
        summary_results["condition_dir"].append(os.path.basename(condition_dir))
        
        # Best overall action
        if len(df) > 0 and df['best_action'].value_counts().size > 0:
            best_overall = df['best_action'].value_counts().sort_values(ascending=False).index[0]
            if best_overall >= 0:
                best_name = f"{action_list[int(best_overall)][0]}_{action_list[int(best_overall)][1]}_{action_list[int(best_overall)][2]}"
                summary_results["best_overall_action"].append(best_name)
            else:
                summary_results["best_overall_action"].append("none")
        else:
            summary_results["best_overall_action"].append("none")
        
        # Best actions by stage
        for stage, stage_name in [(CombustionStage.PREIGNITION.name, "best_preignition_action"), 
                                 (CombustionStage.IGNITION.name, "best_ignition_action"),
                                 (CombustionStage.POSTIGNITION.name, "best_postignition_action")]:
            stage_df = df[df['stage'] == stage]
            if len(stage_df) > 0 and stage_df['best_action'].value_counts().size > 0:
                best_stage = stage_df['best_action'].value_counts().sort_values(ascending=False).index[0]
                if best_stage >= 0:
                    best_stage_name = f"{action_list[int(best_stage)][0]}_{action_list[int(best_stage)][1]}_{action_list[int(best_stage)][2]}"
                    summary_results[stage_name].append(best_stage_name)
                else:
                    summary_results[stage_name].append("none")
            else:
                summary_results[stage_name].append("none")
        
        # Per-action statistics
        for i, (method, rtol, atol) in enumerate(action_list):
            action_name = f"{method}_{rtol}_{atol}"
            
            # Success rate
            success_rate = df[f"{action_name}_success"].mean() * 100 if f"{action_name}_success" in df.columns else 0
            summary_results[f"{action_name}_success_rate"].append(success_rate)
            
            # Average CPU time 
            cpu_times = df[f"{action_name}_cpu_time"].replace([float('inf')], np.nan) if f"{action_name}_cpu_time" in df.columns else pd.Series([np.nan])
            summary_results[f"{action_name}_avg_cpu_time"].append(cpu_times.mean())
            
            # Average error
            errors = df[f"{action_name}_error"].replace([float('inf')], np.nan) if f"{action_name}_error" in df.columns else pd.Series([np.nan])
            summary_results[f"{action_name}_avg_error"].append(errors.mean())
            
            # Average reward
            rewards = df[f"{action_name}_reward"].replace([float('-inf')], np.nan) if f"{action_name}_reward" in df.columns else pd.Series([np.nan])
            summary_results[f"{action_name}_avg_reward"].append(rewards.mean())
    
    # Convert summary results to DataFrame and save
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(base_output_dir, "summary_across_conditions.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary results to {summary_path}")
    
    # Generate summary plots
    generate_summary_plots(summary_df, base_output_dir, action_list)
    
    return summary_df, base_output_dir


def generate_summary_plots(summary_df, output_dir, action_list):
    """Generate summary plots across all conditions."""
    plots_dir = os.path.join(output_dir, "summary_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Success rate by temperature and phi
    plt.figure(figsize=(15, 10))
    
    # Create color map for actions
    action_colors = {}
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(action_list)))
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        action_colors[action_name] = color_cycle[i]
    
    # Plot success rates as a function of temperature
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            summary_df['temperature'], 
            summary_df[f"{action_name}_success_rate"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Success Rate (%)')
    plt.title('Integration Success Rate by Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "success_rate_by_temperature.png"))
    plt.close()
    
    # 2. Success rate as a function of phi
    plt.figure(figsize=(15, 10))
    
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            summary_df['phi'], 
            summary_df[f"{action_name}_success_rate"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Phi')
    plt.ylabel('Success Rate (%)')
    plt.title('Integration Success Rate by Phi')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "success_rate_by_phi.png"))
    plt.close()
    
    # 3. CPU time by temperature
    plt.figure(figsize=(15, 10))
    
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            summary_df['temperature'], 
            summary_df[f"{action_name}_avg_cpu_time"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average CPU Time (s)')
    plt.title('Average CPU Time by Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cpu_time_by_temperature.png"))
    plt.close()
    
    # 4. CPU time by phi
    plt.figure(figsize=(15, 10))
    
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            summary_df['phi'], 
            summary_df[f"{action_name}_avg_cpu_time"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Phi')
    plt.ylabel('Average CPU Time (s)')
    plt.title('Average CPU Time by Phi')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cpu_time_by_phi.png"))
    plt.close()
    
    # 5. Error by temperature
    plt.figure(figsize=(15, 10))
    
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            summary_df['temperature'], 
            summary_df[f"{action_name}_avg_error"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Error')
    plt.yscale('log')  # Use log scale for error
    plt.title('Average Error by Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_by_temperature.png"))
    plt.close()
    
    # 6. Error by phi
    plt.figure(figsize=(15, 10))
    
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            summary_df['phi'], 
            summary_df[f"{action_name}_avg_error"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Phi')
    plt.ylabel('Average Error')
    plt.yscale('log')  # Use log scale for error
    plt.title('Average Error by Phi')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_by_phi.png"))
    plt.close()
    
    # 7. Best action distribution - overall
    action_counts = summary_df['best_overall_action'].value_counts()
    
    plt.figure(figsize=(12, 6))
    colors = [action_colors.get(action, 'gray') for action in action_counts.index]
    plt.bar(action_counts.index, action_counts.values, color=colors)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Best Overall Action Distribution Across All Conditions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "best_overall_action_distribution.png"))
    plt.close()
    
    # 8. Best action by stage
    for stage, column in [
        ('Pre-Ignition', 'best_preignition_action'),
        ('Ignition', 'best_ignition_action'),
        ('Post-Ignition', 'best_postignition_action')
    ]:
        action_counts = summary_df[column].value_counts()
        if len(action_counts) > 0:
            plt.figure(figsize=(12, 6))
            colors = [action_colors.get(action, 'gray') for action in action_counts.index]
            plt.bar(action_counts.index, action_counts.values, color=colors)
            plt.xlabel('Action')
            plt.ylabel('Count')
            plt.title(f'Best Action Distribution for {stage} Stage')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"best_action_{column}.png"))
            plt.close()
    
    # 9. Temperature vs. Phi with best action as color
    plt.figure(figsize=(15, 10))
    
    # Map action names to integers for colormap
    action_to_int = {action: i for i, action in enumerate(set(summary_df['best_overall_action']))}
    colors = [action_to_int[action] for action in summary_df['best_overall_action']]
    
    scatter = plt.scatter(
        summary_df['temperature'],
        summary_df['phi'],
        c=colors,
        s=150,
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add action labels to each point
    for i, txt in enumerate(summary_df['best_overall_action']):
        plt.annotate(txt, 
                    (summary_df['temperature'].iloc[i], summary_df['phi'].iloc[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.colorbar(scatter, ticks=list(action_to_int.values()), 
                label='Best Action')
    plt.clim(-0.5, len(action_to_int)-0.5)
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Phi')
    plt.title('Best Integration Method by Temperature and Phi')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "best_method_by_temp_phi.png"))
    plt.close()
    
    # 10. Reward comparison across all conditions
    plt.figure(figsize=(15, 10))
    
    for i, (method, rtol, atol) in enumerate(action_list):
        action_name = f"{method}_{rtol}_{atol}"
        plt.scatter(
            range(len(summary_df)), 
            summary_df[f"{action_name}_avg_reward"],
            label=action_name,
            color=action_colors[action_name],
            s=80,
            alpha=0.7
        )
    
    plt.xlabel('Condition Index')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Across All Conditions')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_comparison.png"))
    plt.close()


def main():
    """Main function for running the multi-condition benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark different integrators across multiple conditions.')
    
    # Temperature range parameters
    parser.add_argument('--temp_min', type=float, default=300.0, help='Minimum temperature (K)')
    parser.add_argument('--temp_max', type=float, default=1500.0, help='Maximum temperature (K)')
    parser.add_argument('--num_steps_temp', type=int, default=101, help='Number of temperature steps')
    
    # Phi range parameters
    parser.add_argument('--phi_min', type=float, default=0, help='Minimum equivalence ratio')
    parser.add_argument('--phi_max', type=float, default=10, help='Maximum equivalence ratio')
    parser.add_argument('--num_steps_phi', type=int, default=1001, help='Number of phi steps')
    
    # Other parameters
    parser.add_argument('--pressure', type=float, default=1.0, help='Pressure (atm)')
    parser.add_argument('--dt', type=float, default=1e-5, help='Timestep size (s)')
    parser.add_argument('--end_time', type=float, default=1e-1, help='End time for simulation (s)')
    parser.add_argument('--timeout', type=float, default=5.0, help='Timeout for each step (s)')
    parser.add_argument('--reference', type=str, default="CVODE_BDF", help='Reference method for state progression')
    parser.add_argument('--metric', type=str, choices=['cpu_time', 'error', 'reward'], default='reward',
                        help='Metric to use for determining best integrator')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Directory to save results')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum number of steps per benchmark')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of random samples to run')
    
    args_cli = parser.parse_args()
    
    # Create temperature and phi ranges
    temperature_range = np.linspace(args_cli.temp_min, args_cli.temp_max, args_cli.num_steps_temp)
    phi_range = np.linspace(args_cli.phi_min, args_cli.phi_max, args_cli.num_steps_phi)
    
    print(f"Temperature range: {temperature_range.min()}-{temperature_range.max()} K, {len(temperature_range)} values")
    print(f"Phi range: {phi_range.min()}-{phi_range.max()}, {len(phi_range)} values")
    
    # Create a default Args instance for the environment settings
    env_args = Args()
    
    # Run multi-condition benchmark
    summary_df, output_dir = run_multi_condition_benchmark(
        args=env_args,
        temperature_range=temperature_range,
        phi_range=phi_range,
        n_samples=args_cli.num_samples,
        pressure=args_cli.pressure,
        dt=args_cli.dt,
        end_time=args_cli.end_time,
        timeout=args_cli.timeout,
        reference_method=args_cli.reference,
        metric=args_cli.metric,
        output_dir=args_cli.output_dir,
        max_steps=args_cli.max_steps
    )
    
    print(f"\nBenchmark completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()