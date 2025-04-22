#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Integrator Comparison Script for Combustion Problems

This script compares the performance of various ODE integrators:
- CVODE (BDF, ADAMS)
- ARKODE Explicit (ERK methods of various orders)
- ARKODE Implicit (DIRK methods of various orders)
- SciPy BDF
- Custom RK23

It measures CPU time, accuracy, and analyzes performance during different
combustion stages (pre-ignition, ignition, and post-ignition).

Results are stored in a timestamped directory for later analysis.
"""

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Any, Optional

# Import our modules
from combustion_problem import setup_problem, CombustionProblem, CombustionStage
from modified_integrator import IntegratorConfig, ChemicalIntegrator, IntegratorFactory

# Check if sundials_py is available
try:
    import sundials_py
    SUNDIALS_AVAILABLE = True
except ImportError:
    SUNDIALS_AVAILABLE = False
    print("Warning: sundials_py not available. CVODE and ARKODE integrators will not be available.")


def create_work_directory() -> str:
    """Create a timestamped work directory to store all results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = f"integrator_comparison/{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def create_best_solver_visualization(df, solver_column, value_column, title, save_path):
    """
    Create a visualization showing which solver performs best at each timestep.
    
    Args:
        df: DataFrame with timestep, simulation_time, and solver information
        solver_column: Column name containing the best solver for each timestep
        value_column: Column containing the value (CPU time or error)
        title: Plot title
        save_path: Path to save the visualization
    """
    # Get unique solvers and assign colors
    unique_solvers = df[solver_column].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_solvers)))
    solver_colors = dict(zip(unique_solvers, colors))
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)
    
    # Prepare for the scatter plot
    x_times = df['simulation_time']
    y_steps = df['timestep']
    
    # Count how many times each solver is the best
    solver_counts = df[solver_column].value_counts()
    
    # Create scatter plot with different colors for different solvers
    for solver, color in solver_colors.items():
        mask = df[solver_column] == solver
        scatter = ax1.scatter(x_times[mask], y_steps[mask], 
                             c=[color], s=100, label=solver, alpha=0.7)
    
    # Customize the top plot
    ax1.set_ylabel('Timestep')
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with count information
    handles, labels = ax1.get_legend_handles_labels()
    # Get the counts for each solver in the same order as the legend
    counts = [solver_counts[label] for label in labels]
    # Create new labels with counts
    new_labels = [f"{label} ({count} timesteps)" for label, count in zip(labels, counts)]
    ax1.legend(handles, new_labels, title="Solver (Frequency)", loc='best')
    
    # Add percentage summary as text
    total_steps = len(df)
    percentage_text = "Percentage of best timesteps:\n"
    for solver, count in solver_counts.items():
        percentage = (count / total_steps) * 100
        percentage_text += f"{solver}: {percentage:.1f}%\n"
    
    # Add text box with percentages
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.02, 0.98, percentage_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Line plot for the values in the bottom subplot
    for solver, color in solver_colors.items():
        mask = df[solver_column] == solver
        if any(mask):
            ax2.plot(x_times[mask], df.loc[mask, value_column], 'o-', 
                    color=color, label=solver, alpha=0.7, markersize=4)
    
    # Customize the bottom plot
    ax2.set_xlabel('Simulation Time (s)')
    ax2.set_ylabel('Value')
    if 'cpu' in value_column.lower():
        ax2.set_ylabel('CPU Time (s)')
    elif 'error' in value_column.lower():
        ax2.set_ylabel('Error')
        ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def generate_timestep_comparisons(timestep_data: Dict[str, Dict], work_dir: str):
    """
    Generate per-timestep comparison plots and tables.
    
    Args:
        timestep_data: Dictionary of integrator results with per-timestep information
        work_dir: Directory to save results
    """
    # Set up plot for CPU time per timestep comparison
    fig_cpu, ax_cpu = plt.subplots(figsize=(12, 8))
    
    # Set up plot for error per timestep comparison
    fig_error, ax_error = plt.subplots(figsize=(12, 8))
    
    # Create a combined dataframe for timestep performance
    combined_data = []
    
    # Process each integrator's data
    for integrator_name, data in timestep_data.items():
        # Plot CPU time per timestep
        times = data['times'][1:]  # Skip initial point (t=0)
        if 'cpu_times' in data and len(data['cpu_times']) > 0:
            cpu_times = data['cpu_times']
            ax_cpu.plot(times, cpu_times, label=integrator_name, marker='o', markersize=4)
            
            # Plot error per timestep if available
            if 'errors' in data and len(data['errors']) > 0:
                errors = data['errors']
                if len(errors) == len(times):
                    ax_error.plot(times, errors, label=integrator_name, marker='o', markersize=4)
            
            # Add data to combined dataframe
            for i, (t, cpu, err) in enumerate(zip(times, cpu_times, 
                                                 data['errors'] if 'errors' in data and len(data['errors']) > 0 
                                                 else [np.nan] * len(times))):
                combined_data.append({
                    'timestep': i + 1,
                    'simulation_time': t,
                    'integrator': integrator_name,
                    'cpu_time': cpu,
                    'error': err if not np.isnan(err) else None
                })
    
    # Finalize CPU time plot
    ax_cpu.set_xlabel('Simulation Time (s)')
    ax_cpu.set_ylabel('CPU Time per Timestep (s)')
    ax_cpu.set_title('CPU Time per Timestep Comparison')
    ax_cpu.legend()
    ax_cpu.grid(True)
    # Use log scale if range is large
    if ax_cpu.get_ylim()[1] / max(1e-10, ax_cpu.get_ylim()[0]) > 100:
        ax_cpu.set_yscale('log')
    fig_cpu.tight_layout()
    fig_cpu.savefig(os.path.join(work_dir, "cpu_time_per_timestep.png"))
    plt.close(fig_cpu)
    
    # Finalize error plot
    ax_error.set_xlabel('Simulation Time (s)')
    ax_error.set_ylabel('Error per Timestep')
    ax_error.set_title('Error per Timestep Comparison')
    ax_error.legend()
    ax_error.grid(True)
    # Use log scale for errors
    ax_error.set_yscale('log')
    fig_error.tight_layout()
    fig_error.savefig(os.path.join(work_dir, "error_per_timestep.png"))
    plt.close(fig_error)
    
    # Create best performers analysis
    if combined_data:
        # Convert to DataFrame
        df = pd.DataFrame(combined_data)
        
        # Find best CPU time and best error for each timestep
        best_cpu_times = []
        best_errors = []
        
        # Group by timestep
        timestep_groups = df.groupby('timestep')
        
        for timestep, group in timestep_groups:
            # Get simulation time for this timestep
            sim_time = group['simulation_time'].iloc[0] if not group.empty else None
            
            # Find best CPU time performer
            if 'cpu_time' in group.columns:
                best_cpu_idx = group['cpu_time'].idxmin()
                if best_cpu_idx is not None:
                    best_cpu_times.append({
                        'timestep': timestep,
                        'simulation_time': sim_time,
                        'best_cpu_time_integrator': df.loc[best_cpu_idx, 'integrator'],
                        'best_cpu_time': df.loc[best_cpu_idx, 'cpu_time']
                    })
            
            # Find best error performer
            if 'error' in group.columns:
                valid_errors = group[group['error'].notna()]
                if not valid_errors.empty:
                    best_error_idx = valid_errors['error'].idxmin()
                    if best_error_idx is not None:
                        best_errors.append({
                            'timestep': timestep,
                            'simulation_time': sim_time,
                            'best_error_integrator': df.loc[best_error_idx, 'integrator'],
                            'best_error': df.loc[best_error_idx, 'error']
                        })
        
        # Save individual files for best CPU time and best error
        if best_cpu_times:
            best_cpu_df = pd.DataFrame(best_cpu_times)
            best_cpu_df.to_csv(os.path.join(work_dir, "best_cpu_time_by_timestep.csv"), index=False)
            
            # Create a visualization of best CPU time solver per timestep
            create_best_solver_visualization(
                best_cpu_df, 
                'best_cpu_time_integrator',
                'best_cpu_time',
                'Best CPU Time Solver by Timestep',
                os.path.join(work_dir, "best_cpu_time_visualization.png")
            )
        
        if best_errors:
            best_error_df = pd.DataFrame(best_errors)
            best_error_df.to_csv(os.path.join(work_dir, "best_error_by_timestep.csv"), index=False)
            
            # Create a visualization of best error solver per timestep
            create_best_solver_visualization(
                best_error_df, 
                'best_error_integrator',
                'best_error',
                'Best Error Solver by Timestep',
                os.path.join(work_dir, "best_error_visualization.png")
            )
        
        # Create a combined per-timestep comparison table
        # We'll do this without using pivot or merge to avoid multi-index issues
        if best_cpu_times and best_errors:
            # Create timestep comparison table
            comparison_rows = []
            
            # Get all unique timesteps
            all_timesteps = sorted(set(df['timestep'].unique()))
            
            # Get all unique integrators
            all_integrators = sorted(df['integrator'].unique())
            
            # Build a comparison row for each timestep
            for ts in all_timesteps:
                # Basic timestep info
                row = {'timestep': ts}
                
                # Get simulation time
                ts_data = df[df['timestep'] == ts]
                if not ts_data.empty:
                    row['simulation_time'] = ts_data['simulation_time'].iloc[0]
                
                # Best CPU time info
                cpu_data = best_cpu_df[best_cpu_df['timestep'] == ts]
                if not cpu_data.empty:
                    row['best_cpu_time_integrator'] = cpu_data['best_cpu_time_integrator'].iloc[0]
                    row['best_cpu_time'] = cpu_data['best_cpu_time'].iloc[0]
                
                # Best error info
                error_data = best_error_df[best_error_df['timestep'] == ts]
                if not error_data.empty:
                    row['best_error_integrator'] = error_data['best_error_integrator'].iloc[0]
                    row['best_error'] = error_data['best_error'].iloc[0]
                
                # Add CPU time and error for each integrator
                for integrator in all_integrators:
                    integ_data = ts_data[ts_data['integrator'] == integrator]
                    if not integ_data.empty:
                        if 'cpu_time' in integ_data.columns:
                            row[f'{integrator}_cpu_time'] = integ_data['cpu_time'].iloc[0]
                        if 'error' in integ_data.columns and not integ_data['error'].isna().all():
                            row[f'{integrator}_error'] = integ_data['error'].iloc[0]
                
                comparison_rows.append(row)
            
            # Create and save comparison table
            if comparison_rows:
                comparison_df = pd.DataFrame(comparison_rows)
                comparison_df.to_csv(os.path.join(work_dir, "timestep_comparison_table.csv"), index=False)


def run_comparison(problem: CombustionProblem, integrator_configs: List[Dict], 
                  work_dir: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Run comparison of multiple integrators on the same problem.
    """
    results = {}
    stage_metrics = defaultdict(lambda: defaultdict(list))
    
    # Set up figure for temperature profiles
    fig_temp, ax_temp = plt.subplots(figsize=(12, 8))
    
    # Get reference solution for comparison
    reference = problem.get_reference_solution()
    ax_temp.plot(reference['times'][:problem.completed_steps], 
                reference['temperatures'][:problem.completed_steps], 
                'k--', label='Reference Solution')
    
    # Prepare for timestep comparison data
    timestep_data = {}
    
    # Run each integrator
    for config in integrator_configs:
        print(f"\nRunning {config['name']} with rtol={config['rtol']}, atol={config['atol']}...")
        
        try:
            # Create integrator config
            integ_config = IntegratorConfig(
                integrator_list=[config['name']],
                tolerance_list=[(config['rtol'], config['atol'])]
            )
            
            # Create the integrator
            integrator = ChemicalIntegrator(problem, integ_config)
            
            # Run integration loop with proper error handling
            action_idx = 0  # We only have one method in the list
            success_count = 0
            timeout_count = 0
            
            while not integrator.end_simulation:
                try:
                    result = integrator.integrate_step(action_idx, timeout=timeout)
                    
                    if result['success']:
                        success_count += 1
                    
                    if result.get('timed_out', False):
                        timeout_count += 1
                        print(f"Step {integrator.step_count} timed out after {timeout} seconds")
                        
                    if timeout_count > 10:
                        print(f"Too many timeouts, stopping integration for {config['name']}")
                        break
                        
                    # Check for max_step limit
                    if integrator.step_count > 100:  # Prevent infinite loops
                        print(f"Reached maximum step count limit of 500, stopping integration for {config['name']}")
                        break
                        
                except Exception as e:
                    print(f"Error during integration step: {str(e)}")
                    break
            
            # Get statistics
            stats = integrator.get_statistics()
            
            # Store results
            config_name = f"{config['name']}_rtol{config['rtol']}_atol{config['atol']}"
            results[config_name] = {
                'history': integrator.history,
                'statistics': stats,
                'success_rate': success_count / max(1, integrator.step_count),
                'timeout_count': timeout_count
            }
            
            # Store per-timestep data for comparisons
            timestep_data[config_name] = {
                'times': integrator.history['times'],
                'cpu_times': integrator.history['cpu_times'],
                'errors': integrator.history['errors'] if 'errors' in integrator.history else [],
                'temperatures': integrator.history['temperatures']
            }
            
            # Calculate stage metrics
            for stage in CombustionStage:
                stage_name = stage.name
                if stage.value in stats['stage_steps'] and stats['stage_steps'][stage.value] > 0:
                    cpu_time = stats['stage_cpu_times'][stage.value]
                    num_steps = stats['stage_steps'][stage.value]
                    
                    if stage.value == CombustionStage.PREIGNITION.value:
                        steps = stats['stage_steps'][stage.value]
                    elif stage.value == CombustionStage.IGNITION.value:
                        steps = stats['stage_steps'][stage.value] - stats['stage_steps'][CombustionStage.PREIGNITION.value]
                    else:
                        steps = stats['num_steps'] - stats['stage_steps'][CombustionStage.IGNITION.value]
                    
                    stage_metrics[stage_name]['integrator'].append(config_name)
                    stage_metrics[stage_name]['cpu_time'].append(cpu_time)
                    stage_metrics[stage_name]['steps'].append(steps)
                    stage_metrics[stage_name]['avg_time_per_step'].append(cpu_time / max(1, steps))
            
            # Plot temperature profile
            times = integrator.history['times']
            temperatures = integrator.history['temperatures']
            if len(times) > 0 and len(temperatures) > 0:
                ax_temp.plot(times, temperatures, label=config_name)
            
            # Save individual integrator plots
            try:
                integrator.plot_history(save_path=os.path.join(work_dir, f"{config_name}_history.png"))
            except Exception as e:
                print(f"Error saving plot for {config_name}: {str(e)}")
            
            # Save history to CSV
            try:
                history_df = pd.DataFrame({
                    'time': integrator.history['times'],
                    'temperature': integrator.history['temperatures'],
                    'cpu_time': [0] + integrator.history['cpu_times'],  # Add 0 for initial step
                    'success': [True] + integrator.history['success_flags'],  # Add True for initial step
                    'stage': [CombustionStage.PREIGNITION.name] + [s.name for s in integrator.history['stages']]
                })
                
                # Add species profiles
                for spec, profile in integrator.history['species_profiles'].items():
                    history_df[f"{spec}_mass_fraction"] = profile
                
                history_df.to_csv(os.path.join(work_dir, f"{config_name}_history.csv"), index=False)
            except Exception as e:
                print(f"Error saving history data for {config_name}: {str(e)}")
                
        except Exception as e:
            print(f"Failed to run integrator {config['name']}: {str(e)}")
            # Add empty results for failed integrator
            config_name = f"{config['name']}_rtol{config['rtol']}_atol{config['atol']}"
            results[config_name] = {
                'history': {'times': [], 'temperatures': [], 'cpu_times': []},
                'statistics': {'total_cpu_time': 0, 'average_cpu_time': 0, 'average_error': float('inf')},
                'success_rate': 0.0,
                'timeout_count': 0,
                'error': str(e)
            }
            
            timestep_data[config_name] = {
                'times': [],
                'cpu_times': [],
                'errors': [],
                'temperatures': []
            }
        
    
    # Finalize and save temperature plot
    ax_temp.set_xlabel('Time (s)')
    ax_temp.set_ylabel('Temperature (K)')
    ax_temp.set_title('Temperature Evolution Comparison')
    ax_temp.legend()
    ax_temp.grid(True)
    fig_temp.savefig(os.path.join(work_dir, "temperature_comparison.png"))
    plt.close(fig_temp)
    
    # Generate per-timestep comparison plots and tables
    generate_timestep_comparisons(timestep_data, work_dir)
    
    # Create stage performance dataframes and plots
    for stage_name, metrics in stage_metrics.items():
        if metrics['integrator']:  # Check if we have data for this stage
            # Create DataFrame
            stage_df = pd.DataFrame({
                'integrator': metrics['integrator'],
                'cpu_time': metrics['cpu_time'],
                'steps': metrics['steps'],
                'avg_time_per_step': metrics['avg_time_per_step']
            })
            
            # Save to CSV
            stage_df.to_csv(os.path.join(work_dir, f"{stage_name}_metrics.csv"), index=False)
            
            # Create bar plots for CPU time
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(stage_df['integrator'], stage_df['cpu_time'])
            ax.set_xlabel('Integrator')
            ax.set_ylabel('CPU Time (s)')
            ax.set_title(f'CPU Time for {stage_name} Stage')
            plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(os.path.join(work_dir, f"{stage_name}_cpu_time.png"))
            plt.close(fig)
            
            # Create bar plots for average time per step
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(stage_df['integrator'], stage_df['avg_time_per_step'])
            ax.set_xlabel('Integrator')
            ax.set_ylabel('Average Time per Step (s)')
            ax.set_title(f'Average Step Time for {stage_name} Stage')
            plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.6f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(os.path.join(work_dir, f"{stage_name}_avg_step_time.png"))
            plt.close(fig)
    
    # Create overall performance comparison
    summary_data = {
        'integrator': [],
        'total_cpu_time': [],
        'avg_step_time': [],
        'success_rate': [],
        'timeout_count': [],
        'max_error': [],
        'avg_error': []
    }
    
    for config_name, result in results.items():
        stats = result['statistics']
        summary_data['integrator'].append(config_name)
        summary_data['total_cpu_time'].append(stats['total_cpu_time'])
        summary_data['avg_step_time'].append(stats['average_cpu_time'])
        summary_data['success_rate'].append(result['success_rate'])
        summary_data['timeout_count'].append(result['timeout_count'])
        summary_data['max_error'].append(stats['max_error'])
        summary_data['avg_error'].append(stats['average_error'])
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(work_dir, "performance_summary.csv"), index=False)
    
    # Create summary plots
    # Total CPU time
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(summary_df['integrator'], summary_df['total_cpu_time'])
    ax.set_xlabel('Integrator')
    ax.set_ylabel('Total CPU Time (s)')
    ax.set_title('Total CPU Time Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "total_cpu_time.png"))
    plt.close(fig)
    
    # Average error
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(summary_df['integrator'], summary_df['avg_error'])
    ax.set_xlabel('Integrator')
    ax.set_ylabel('Average Error')
    ax.set_title('Average Error Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "average_error.png"))
    plt.close(fig)
    
    # Success rate
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(summary_df['integrator'], summary_df['success_rate'])
    ax.set_xlabel('Integrator')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "success_rate.png"))
    plt.close(fig)
    
    # Save problem parameters
    problem_params = problem.get_problem_params()
    with open(os.path.join(work_dir, "problem_parameters.json"), 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v 
                for k, v in problem_params.items()}, f, indent=4)
    
    return {
        'summary': summary_df.to_dict(orient='records'),
        'stage_metrics': {stage: pd.DataFrame(metrics).to_dict(orient='records') 
                          for stage, metrics in stage_metrics.items() if metrics['integrator']},
        'work_dir': work_dir
    }


def create_categorized_plots(results, work_dir):
    """Create additional plots grouping integrators by family."""
    summary_df = pd.DataFrame(results['summary'])
    
    # Define categories
    categories = {}
    for integrator in summary_df['integrator']:
        if 'CVODE' in integrator:
            category = 'CVODE'
        elif 'ARKODE' in integrator:
            if any(keyword in integrator for keyword in 
                   ['SDIRK', 'BILLINGTON', 'TRBDF2', 'KVAERNO', 'CASH']):
                category = 'ARKODE_IMPLICIT'
            else:
                category = 'ARKODE_EXPLICIT'
        elif 'CPP_RK23' in integrator:
            category = 'CUSTOM_RK'
        else:
            category = 'SCIPY'
        
        if category not in categories:
            categories[category] = []
        categories[category].append(integrator)
    
    # Create plots by category
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # CPU time by category
    cat_times = []
    cat_names = []
    for cat_name, integrators in categories.items():
        # Calculate average time for the category
        times = summary_df.loc[summary_df['integrator'].isin(integrators), 'total_cpu_time']
        if len(times) > 0:
            cat_times.append(times.mean())
            cat_names.append(cat_name)
    
    # Sort by time (ascending)
    sorted_indices = np.argsort(cat_times)
    sorted_times = [cat_times[i] for i in sorted_indices]
    sorted_names = [cat_names[i] for i in sorted_indices]
    
    # Plot
    bars = axes[0].bar(sorted_names, sorted_times)
    axes[0].set_xlabel('Integrator Family')
    axes[0].set_ylabel('Average CPU Time (s)')
    axes[0].set_title('CPU Time by Integrator Family')
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Errors by category
    cat_errors = []
    for cat_name, integrators in categories.items():
        # Calculate average error for the category
        errors = summary_df.loc[summary_df['integrator'].isin(integrators), 'avg_error']
        if len(errors) > 0:
            cat_errors.append(errors.mean())
        else:
            cat_errors.append(0)
    
    # Sort by same order as times
    sorted_errors = [cat_errors[sorted_names.index(name)] for name in sorted_names]
    
    # Plot
    bars = axes[1].bar(sorted_names, sorted_errors)
    axes[1].set_xlabel('Integrator Family')
    axes[1].set_ylabel('Average Error')
    axes[1].set_title('Error by Integrator Family')
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.4e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "family_comparison.png"))
    plt.close(fig)


def main():
    """Main function to run the integrator comparison."""
    print("Starting integrator comparison for combustion problems...")
    
    # Create work directory
    work_dir = create_work_directory()
    print(f"Created work directory: {work_dir}")
    
    # Set up problem parameters
    temperature_range = np.linspace(300, 1300, 101)  # Initial temperature range
    pressure_range = np.array([1.0])  # Pressure in atm
    phi_range = np.linspace(0, 10, 50)  # Equivalence ratio
    
    # Create a combustion problem
    print("Setting up combustion problem...")
    problem = setup_problem(
        temperature_range=temperature_range,
        pressure_range=pressure_range,
        phi_range=phi_range,
        mech_file="/home/elo/ubunu_codes/SCI-ML/0D_integratorRL/large_mechanism/large_mechanism/n-dodecane.yaml",  # or the path to your mechanism file
        fuel="nc12h26",
        oxidizer="O2:1, N2:3.76",
        end_time=0.001,  # seconds
        reference_rtol=1e-10,
        reference_atol=1e-20,
        state_change_threshold=1,
        randomize=True,  # Use fixed values for reproducibility
        verbose=True
    )
    
    # Save the reference solution plot
    problem.plot_reference_solution(save_path=os.path.join(work_dir, "reference_solution.png"))
    
    # Define integrator configurations to compare
    integrator_configs = []
    
    # # Add SciPy BDF
    # integrator_configs.append({
    #     'name': 'BDF',
    #     'rtol': 1e-6,
    #     'atol': 1e-8
    # })
    
    # # Add the custom RK23 solver
    # integrator_configs.append({
    #     'name': 'CPP_RK23',
    #     'rtol': 1e-6,
    #     'atol': 1e-8
    # })
    
    # Add SUNDIALS integrators if available
    if SUNDIALS_AVAILABLE:
        # # Add CVODE BDF
        # integrator_configs.append({
        #     'name': 'CVODE_BDF',
        #     'rtol': 1e-6,
        #     'atol': 1e-8
        # })
        
        # Add ARKODE explicit methods (ERK)
        erk_methods = [
            'ARKODE_HEUN_EULER',      # 2nd order
            'ARKODE_BOGACKI_SHAMPINE', # 3rd order
            'ARKODE_ZONNEVELD',        # 4th order
            'ARKODE_ARK436L2SA_ERK',   # 4th order, optimized
            'ARKODE_VERNER',           # 6th order
            'ARKODE_FEHLBERG'          # 8th order
        ]
        
        for method in erk_methods:
            integrator_configs.append({
                'name': method,
                'rtol': 1e-6,
                'atol': 1e-8
            })
        
        # Add ARKODE implicit methods (DIRK)
        dirk_methods = [
            'ARKODE_SDIRK_2_1_2',       # 2nd order, 2 stages
            'ARKODE_BILLINGTON_3_3_2',  # 2nd order, 3 stages
            'ARKODE_TRBDF2_3_3_2',      # 2nd order, 3 stages  
            'ARKODE_KVAERNO_4_2_3',     # 3rd order, 4 stages
            'ARKODE_CASH_5_2_4',        # 4th order, 5 stages
            'ARKODE_SDIRK_5_3_4',       # 4th order, 5 stages
        ]
        
        for method in dirk_methods:
            integrator_configs.append({
                'name': method,
                'rtol': 1e-6,
                'atol': 1e-8
            })
    
    # Run the comparison
    print("\nRunning integrator comparison...")
    print(f"Testing {len(integrator_configs)} different integrators")
    results = run_comparison(problem, integrator_configs, work_dir)
    
    # Print summary
    print("\nComparison completed. Results saved to:", work_dir)
    print("\nSummary of results:")
    summary_df = pd.DataFrame(results['summary'])
    print(summary_df.to_string(index=False))
    
    # Print stage-specific analysis
    print("\nStage-specific performance:")
    for stage, metrics in results['stage_metrics'].items():
        print(f"\n{stage} stage:")
        stage_df = pd.DataFrame(metrics)
        if not stage_df.empty:
            print(stage_df.to_string(index=False))
    
    # Create categorized performance plots
    create_categorized_plots(results, work_dir)
    
    print("\nCheck the output directory for detailed plots and data.")


if __name__ == "__main__":
    main()