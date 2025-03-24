import wandb
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import time
import numpy as np
from typing import Dict, Any

class Logger:
    """Logger class for tracking training metrics and visualizations."""
    
    def __init__(self, args, run_name: str, work_dir: str):
        """Initialize logger with configuration."""
        self.args = args
        self.run_name = run_name
        self.work_dir = work_dir
        self.writer = None
        
        # Create necessary directories
        self.plots_dir = f"{work_dir}/plots"
        self.models_dir = f"{work_dir}/models"
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.setup_logging()
        
    def setup_logging(self):
        """Set up WandB and TensorBoard logging."""
        if self.args.track:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        
        self.writer = SummaryWriter(f"{self.work_dir}/runs/{self.run_name}")
        
        # Log hyperparameters
        hyperparams = {
            "Environment": {
                "mech_file": self.args.mech_file,
                "fuel": self.args.fuel,
                "species_to_track": self.args.species_to_track,
                "temperature_range": f"{self.args.temp_min}-{self.args.temp_max}",
                "pressure_range": f"{self.args.press_min}-{self.args.press_max}",
                "phi_range": f"{self.args.phi_min}-{self.args.phi_max}",
            },
            "Integration": {
                "integrator_list": self.args.integrator_list,
                "tolerance_list": self.args.tolerance_list,
                "reference_rtol": self.args.reference_rtol,
                "reference_atol": self.args.reference_atol,
            },
            "PPO": {
                "learning_rate": self.args.learning_rate,
                "total_timesteps": self.args.total_timesteps,
                "num_minibatches": self.args.num_minibatches,
            }
        }
        
        # Log hyperparameters in a structured format
        self.writer.add_text(
            "hyperparameters",
            "|Category|Parameter|Value|\n|-|-|-|\n" + "\n".join(
                [f"|{cat}|{param}|{value}|" for cat, params in hyperparams.items() 
                 for param, value in params.items()]
            )
        )
    
    def log_episode_info(self, global_step: int, episode_number: int, 
                        info: Dict[str, Any], prefix: str = "charts", 
                        end_of_episode: bool = False):
        """Log episode-specific information."""
        # Standard episode metrics
        if 'episode' in info:
            self.writer.add_scalar(f"{prefix}/episodic_return", info["episode"]["r"], global_step)
            self.writer.add_scalar(f"{prefix}/episodic_length", info["episode"]["l"], global_step)
        
        # Combustion-specific metrics
        metric_groups = {
            'instantaneous': {
                'cpu_time': 'Computation Time (s)',
                'error': 'Solution Error',
                'temperature_error': 'Temperature Error (K)',
                'time_reward': 'Time-based Reward',
                'error_reward': 'Error-based Reward',
                'stage_value': 'Combustion Stage Value'
            },
            'cumulative': {
                'cummulative_cpu_time': 'Total Computation Time (s)',
                'cummulative_temperature_error': 'Total Temperature Error (K)',
                'cummulative_reward': 'Total Episode Reward'
            }
        }
        
        # Log metrics by group
        for group, metrics in metric_groups.items():
            for key, label in metrics.items():
                if key in info and info[key] is not None:
                    self.writer.add_scalar(f"{prefix}/{key}", info[key], global_step)
                    if self.args.track:
                        wandb.log({f"{prefix}/{key}": info[key]}, step=global_step)
        
        # Log stage transitions
        if 'current_stage' in info:
            self.writer.add_text(f"{prefix}/combustion_stage", 
                               f"Step {global_step}: {info['current_stage']}", 
                               global_step)
        
        # End of episode summary
        if end_of_episode:
            try:
                print(f"Episode {episode_number} - Global step {global_step}")
                summary = (f"Episode {episode_number} - Global step {global_step}\n"
                      f"Reward: {info.get('cummulative_reward', 0):.4f}\n"
                      f"CPU Time: {info.get('cummulative_cpu_time', 0):.4f}s\n"
                      f"Temp Error: {info.get('cummulative_error', 0):.4f}K")
                self.writer.add_text(f"{prefix}/episode_summary", summary, global_step)
            except Exception as e:
                import traceback
                print(f"Error logging episode summary: {e}")
                print(traceback.format_exc())
            
    def log_eval_results(self, iteration: int, cummulative_reward: float, 
                         cummulative_cpu_time: float, cummulative_error: float,
                         episode_mean_reward: float, episode_mean_error: float, 
                         episode_mean_time: float):
        """Log evaluation results."""
        self.writer.add_scalar("charts/eval_reward", cummulative_reward, iteration)
        self.writer.add_scalar("charts/eval_cpu_time", cummulative_cpu_time, iteration)
        self.writer.add_scalar("charts/eval_error", cummulative_error, iteration)
        self.writer.add_scalar("charts/eval_mean_reward", episode_mean_reward, iteration)
        self.writer.add_scalar("charts/eval_mean_error", episode_mean_error, iteration)
        self.writer.add_scalar("charts/eval_mean_time", episode_mean_time, iteration)
        
        if self.args.track:
            wandb.log({
                "charts/eval_reward": cummulative_reward,
                "charts/eval_cpu_time": cummulative_cpu_time,
                "charts/eval_error": cummulative_error,
                "charts/eval_mean_reward": episode_mean_reward,
                "charts/eval_mean_error": episode_mean_error,
                "charts/eval_mean_time": episode_mean_time
            }, step=iteration)
    
    def log_training_progress(self, agent, clipfracs: np.ndarray, 
                            losses: Dict[str, float], global_step: int, 
                            start_time: float, statistics: Dict[str, np.ndarray]):
        """Log training progress metrics."""
        # Performance metrics
        sps = int(global_step / (time.time() - start_time))
        self.writer.add_scalar("charts/SPS", sps, global_step)
        
        # Loss components
        for loss_name, loss_value in losses.items():
            self.writer.add_scalar(f"losses/{loss_name}", loss_value, global_step)
        
        # PPO metrics
        self.writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)
        
        # Learning rate
        if hasattr(agent, 'optimizer'):
            current_lr = agent.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("charts/learning_rate", current_lr, global_step)
        
        # Value and return statistics
        for stat_name, values in statistics.items():
            if values is not None:
                self.writer.add_scalar(f"charts/{stat_name}", values, global_step)
    
    def log_comparison_plot(self, plot_path: str, episode: int):
        """Log comparison plot to WandB."""
        if self.args.track:
            wandb.log({
                f"evaluation/comparison_plot_{episode}": wandb.Image(plot_path),
                "episode": episode
            })
    
    def plot_training_results(self):
        """Generate and save training result plots."""
        event_acc = EventAccumulator(self.writer.log_dir)
        event_acc.Reload()
        
        metric_groups = {
            'Performance': [
                'charts/episodic_return',
                'charts/episodic_length',
                'charts/SPS'
            ],
            'Losses': [
                'losses/policy_loss',
                'losses/value_loss',
                'losses/entropy_loss'
            ],
            'Combustion': [
                'charts/average_error',
                'charts/average_cpu_time',
                'charts/temperature_error'
            ]
        }
        
        for group_name, metrics in metric_groups.items():
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
            fig.suptitle(group_name, fontsize=16)
            
            for i, metric in enumerate(metrics):
                if metric in event_acc.Tags()['scalars']:
                    events = event_acc.Scalars(metric)
                    steps = [event.step for event in events]
                    values = [event.value for event in events]
                    
                    ax = axes[i] if len(metrics) > 1 else axes
                    ax.plot(steps, values)
                    ax.set_title(metric.split('/')[-1].replace('_', ' ').title())
                    ax.set_xlabel('Steps')
                    ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/{group_name.lower()}_metrics.png")
            plt.close()
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint and log to WandB."""
        if self.args.track:
            wandb.save(checkpoint_path)
    
    def close(self):
        """Clean up logger resources."""
        if self.writer is not None:
            self.writer.close()
        if self.args.track:
            wandb.finish()