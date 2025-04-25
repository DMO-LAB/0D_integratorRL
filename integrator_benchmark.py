import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import argparse

from environment.env_wrapper import EnvManager
from environment.combustion_problem import CombustionProblem, setup_problem, CombustionStage
from environment.modified_integrator import ChemicalIntegrator, IntegratorConfig
from config.default_config import Args

class IntegratorBenchmark:
    """Benchmark different integrators on the same combustion problem."""
    
    def __init__(self, args, fixed_temperature=1100.0, fixed_pressure=1.0, fixed_phi=1.0, 
                 fixed_dt=1e-5, end_time=1e-3, timeout=5.0, reference_method="CVODE_BDF",
                 metric="reward", output_dir="benchmark_results"):
        """
        Initialize benchmark tool.
        
        Args:
            args: Configuration arguments
            fixed_temperature: Fixed temperature for the problem
            fixed_pressure: Fixed pressure for the problem
            fixed_phi: Fixed equivalence ratio for the problem
            fixed_dt: Fixed timestep for the problem
            end_time: End time for simulation
            timeout: Timeout for each integration step
            reference_method: Method to use as reference for states (default: CVODE_BDF)
            metric: Metric to use for determining best integrator ("cpu_time", "error", or "reward")
            output_dir: Directory to save results
        """
        self.args = args
        self.fixed_temperature = fixed_temperature
        self.fixed_pressure = fixed_pressure
        self.fixed_phi = fixed_phi
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
        
        # First, create the reference environment
        ref_method, ref_rtol, ref_atol = self.action_list[self.reference_action_idx]
        
        # Create a modified args with only the reference method
        ref_args = Args()
        ref_args.integrator_list = [ref_method]
        ref_args.tolerance_list = [(ref_rtol, ref_atol)]
        
        # Create reference environment
        self.reference_env = self.env_manager.create_single_env(
            end_time=self.end_time,
            fixed_temperature=self.fixed_temperature,
            fixed_pressure=self.fixed_pressure,
            fixed_phi=self.fixed_phi,
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
                    fixed_temperature=self.fixed_temperature,
                    fixed_pressure=self.fixed_pressure,
                    fixed_phi=self.fixed_phi,
                    fixed_dt=self.fixed_dt,
                    randomize=False
                )
                
                self.envs[i] = env
                
        print(f"Created {len(self.envs)} environments.")
        
    def run_benchmark(self, max_steps=100):
        """Run benchmark on all actions."""
        print(f"Running benchmark with metric: {self.metric}")
        
        # Initialize result storage
        results = {
            'step': [],
            'temperature': [],
            'stage': [],
            'best_action': []
        }
        
        # Add columns for each action
        for i, (method, rtol, atol) in enumerate(self.action_list):
            action_name = f"{method}_{rtol}_{atol}"
            results[f"{action_name}_cpu_time"] = []
            results[f"{action_name}_error"] = []
            results[f"{action_name}_reward"] = []
            results[f"{action_name}_success"] = []
        
        # Reset all environments to ensure same initial state
        for i, env in self.envs.items():
            observation, _ = env.reset()
            
        # Run steps
        for step in range(max_steps):
            print(f"Step {step}/{max_steps}...")
            
            # Get state from reference environment
            reference_temp = self.envs[self.reference_action_idx].integrator.y[0]
            reference_stage = self.envs[self.reference_action_idx].integrator.current_stage
            
            # Store step data
            results['step'].append(step)
            results['temperature'].append(reference_temp)
            results['stage'].append(reference_stage.name)
            
            # Dictionary to store metrics for determining best action
            step_metrics = {}
            
            # create a dictionary to store the done status of each action
            actions_dones = {
                i: False for i, (method, rtol, atol) in enumerate(self.action_list)
            }
            
            # Take a step with each action
            for i, (method, rtol, atol) in enumerate(self.action_list):
                action_name = f"{method}_{rtol}_{atol}"
                if actions_dones[i]:
                    print(f"skipping environment {i} ({action_name}) because it terminated at step {step}")
                    raise
                # Take a step
                try:
                    env = self.envs[i]
                    _, reward, terminated, truncated, info = env.step(i, timeout=self.timeout)  # Action 0 corresponds to the only method for this env
                    
                    # Store metrics
                    cpu_time = info.get('cpu_time', float('inf'))
                    error = info.get('error', float('inf'))
                    success = info.get('success', False)

                    print(f"Action {i} ({action_name}) - cpu_time: {cpu_time} - error: {error} - reward: {reward} - success: {success}")
                    
                    results[f"{action_name}_cpu_time"].append(cpu_time)
                    results[f"{action_name}_error"].append(error)
                    results[f"{action_name}_reward"].append(reward)
                    results[f"{action_name}_success"].append(success)
                    
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
                        cpu_time = float('inf')
                        error = float('inf')
                        reward = float('-inf')
                        actions_dones[i] = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error in environment {i} ({action_name}): {e}")
                    # Log failures
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
                print(f"Best action at step {step}: {best_action}")
            else:
                results['best_action'].append(-1)  # No valid actions
                
            # Check if all environments are done
            all_done = True
            for i, env in self.envs.items():
                if not env.integrator.end_simulation:
                    all_done = False
                    break
                    
            if all_done:
                print(f"All environments completed at step {step}")
                break
            
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(self.output_dir, f"benchmark_results_{self.metric}.csv")
        df.to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")
        
        # Generate plots
        self.generate_plots(df)
        
        return df
    
    def generate_plots(self, df):
        """Generate benchmark plots."""
        # Create a directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot temperature evolution
        plt.figure(figsize=(12, 6))
        plt.plot(df['step'], df['temperature'])
        plt.xlabel('Step')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "temperature_evolution.png"))
        plt.close()
        
        # Plot best action distribution
        action_counts = df['best_action'].value_counts().sort_index()
        labels = [f"{self.action_list[i][0]}_{self.action_list[i][1]}_{self.action_list[i][2]}" for i in action_counts.index if i >= 0]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, action_counts.values)
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.title(f'Best Action Distribution (Metric: {self.metric})')
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
                title = "CPU Time Over Steps"
                ylabel = "CPU Time (s)"
            elif self.metric == "error":
                y = df[f"{action_name}_error"]
                title = "Error Over Steps"
                ylabel = "Error"
            else:  # "reward"
                y = df[f"{action_name}_reward"]
                title = "Reward Over Steps"
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
        plt.title('Success Rate by Action')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "success_rate.png"))
        plt.close()
        
        # Plot best action by combustion stage
        stage_best_actions = df.groupby('stage')['best_action'].value_counts().unstack().fillna(0)
        
        plt.figure(figsize=(12, 8))
        stage_best_actions.plot(kind='bar', stacked=True)
        plt.xlabel('Combustion Stage')
        plt.ylabel('Count')
        plt.title(f'Best Action by Combustion Stage (Metric: {self.metric})')
        plt.legend(title='Action Index')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"best_action_by_stage_{self.metric}.png"))
        plt.close()
        
        # Generate summary statistics
        self.generate_summary_stats(df)
        
    def generate_summary_stats(self, df):
        """Generate summary statistics."""
        summary_path = os.path.join(self.output_dir, "summary_statistics.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Integrator Benchmark Summary (Metric: {self.metric})\n")
            f.write("="*80 + "\n\n")
            
            f.write("Problem Parameters:\n")
            f.write(f"  Temperature: {self.fixed_temperature} K\n")
            f.write(f"  Pressure: {self.fixed_pressure} atm\n")
            f.write(f"  Phi: {self.fixed_phi}\n")
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
                best_pct = best_count / len(df) * 100
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
                        action_name = f"{self.action_list[action_idx][0]}_{self.action_list[action_idx][1]}_{self.action_list[action_idx][2]}"
                        pct = count / len(stage_df) * 100
                        f.write(f"    Best Action: {action_name} - {count} times ({pct:.2f}%)\n")
                
                f.write("\n")
            
            # Overall best action
            best_overall = df['best_action'].value_counts().sort_values(ascending=False).index[0]
            if best_overall >= 0:
                best_name = f"{self.action_list[best_overall][0]}_{self.action_list[best_overall][1]}_{self.action_list[best_overall][2]}"
                best_count = (df['best_action'] == best_overall).sum()
                best_pct = best_count / len(df) * 100
                f.write(f"Overall Best Action: {best_name} - {best_count} times ({best_pct:.2f}%)\n")
            else:
                f.write("No valid best action found.\n")
        
        print(f"Generated summary statistics at {summary_path}")

def main():
    """Main function for running the benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark different integrators for chemical kinetics.')
    parser.add_argument('--temperature', type=float, default=1100.0, help='Initial temperature (K)')
    parser.add_argument('--pressure', type=float, default=1.0, help='Initial pressure (atm)')
    parser.add_argument('--phi', type=float, default=1.0, help='Equivalence ratio')
    parser.add_argument('--dt', type=float, default=1e-5, help='Timestep size (s)')
    parser.add_argument('--end_time', type=float, default=1e-3, help='End time for simulation (s)')
    parser.add_argument('--timeout', type=float, default=5.0, help='Timeout for each step (s)')
    parser.add_argument('--reference', type=str, default="CVODE_BDF", help='Reference method for state progression')
    parser.add_argument('--metric', type=str, choices=['cpu_time', 'error', 'reward'], default='reward',
                        help='Metric to use for determining best integrator')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Directory to save results')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of steps to run')
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = IntegratorBenchmark(
        args=Args(),
        fixed_temperature=args.temperature,
        fixed_pressure=args.pressure,
        fixed_phi=args.phi,
        fixed_dt=args.dt,
        end_time=args.end_time,
        timeout=args.timeout,
        reference_method=args.reference,
        metric=args.metric,
        output_dir=args.output_dir
    )
    
    # Setup environments
    benchmark.setup_environments()
    
    # Run benchmark
    benchmark.run_benchmark(max_steps=args.max_steps)
    
if __name__ == "__main__":
    main()