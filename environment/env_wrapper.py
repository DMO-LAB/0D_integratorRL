import gymnasium as gym
import cantera as ct
from environment.environment import CombustionEnv
from environment.integrator import ChemicalIntegrator, IntegratorConfig
from environment.combustion_problem import CombustionProblem, setup_problem
import numpy as np
from gymnasium.vector import SyncVectorEnv
from typing import Dict


class EnvManager:
    """Manages multiple instances of the CombustionEnv environment."""
    
    def __init__(self, args, single_env=False, create_envs=True):
        """Initialize environment manager with configuration."""
        self.args = args
        self.single_env = single_env
        if create_envs:
            if single_env:
                self.env = self.create_single_env()
            else:
                self.envs = self._create_vector_env()
        else:
            self.env = None
            self.envs = None
        
    def _create_vector_env(self) -> SyncVectorEnv:
        """Create vectorized environment."""
        def make_env(args, idx):
            def thunk():
                # Create problem instance
                problem = setup_problem(
                    temperature_range=args.temperature_range,
                    pressure_range=args.pressure_range,
                    phi_range=args.phi_range,
                    mech_file=args.mech_file,
                    fuel=args.fuel,
                    species_to_track=args.species_to_track,
                    reference_rtol=args.reference_rtol,
                    reference_atol=args.reference_atol,
                    end_time=args.end_time,
                    min_time_steps_range=args.min_time_steps_range,
                    max_time_steps_range=args.max_time_steps_range
                )
                
                # Create integrator
                integrator_config = IntegratorConfig(
                    integrator_list=args.integrator_list,
                    tolerance_list=args.tolerance_list
                )
                integrator = ChemicalIntegrator(problem=problem, config=integrator_config)
                
                # Create environment
                env = CombustionEnv(
                    problem=problem,
                    integrator=integrator,
                    features_config=args.features_config,
                    reward_weights=args.reward_weights
                )
                
                print(f"Environment {idx} created with T={problem.temperature}, P={problem.pressure}, phi={problem.phi} and timestep={problem.timestep}")
                # Add wrappers
                env = gym.wrappers.RecordEpisodeStatistics(env)
                if args.normalize_obs:
                    env = gym.wrappers.NormalizeObservation(env)
                if args.normalize_reward:
                    env = gym.wrappers.NormalizeReward(env)
                    
                return env
            return thunk
            
        return SyncVectorEnv([make_env(self.args, i) for i in range(self.args.num_envs)])
    
    
    def create_single_env(self, end_time: float = 1e-3, fixed_temperature: float = None, fixed_pressure: float = None, fixed_phi: float = None, fixed_dt: float = None, randomize: bool = True, initial_mixture: str = None):
        """Create a single environment."""
        if randomize:
            print(f"Creating a single environment with random parameters")
        else:
            print(f"Creating a single environment with fixed parameters - T={fixed_temperature}, P={fixed_pressure}, phi={fixed_phi}, dt={fixed_dt}")
        problem = setup_problem(
            temperature_range=self.args.temperature_range,
            pressure_range=self.args.pressure_range,
            phi_range=self.args.phi_range,
            mech_file=self.args.mech_file,
            fuel=self.args.fuel,
            species_to_track=self.args.species_to_track,
            end_time=end_time,
            fixed_temperature=fixed_temperature,
            fixed_pressure=fixed_pressure,
            fixed_phi=fixed_phi,
            fixed_dt=fixed_dt,
            randomize=randomize,
            initial_mixture=initial_mixture
        )
        print(f"Done setting up problem")
        
        integrator_config = IntegratorConfig(
            integrator_list=self.args.integrator_list,
            tolerance_list=self.args.tolerance_list
        )
        integrator = ChemicalIntegrator(problem=problem, config=integrator_config)
        env = CombustionEnv(
            problem=problem,
            integrator=integrator,
            features_config=self.args.features_config,
            reward_weights=self.args.reward_weights
        )
        return env
        
    def generate_environments(self, single_env=False):
        """Generate environments."""
        print("Generating new environments...")
        if single_env:
            self.env = self.create_single_env()
        else:
            self.envs = self._create_vector_env()
    
    def get_env_metrics(self) -> Dict[str, float]:
        """Get metrics from all environments."""
        metrics = {
            'errors': [],
            'cpu_times': [],
            'rewards': [],
            'episode_lengths': []
        }
        
        for env in self.envs.envs:
            if hasattr(env.unwrapped, 'integrator'):
                stats = env.unwrapped.integrator.get_statistics()
                metrics['errors'].append(stats.get('average_error', 0))
                metrics['cpu_times'].append(stats.get('total_cpu_time', 0))
                
            if hasattr(env, 'return_queue'):
                metrics['rewards'].extend(list(env.return_queue))
            if hasattr(env, 'length_queue'):
                metrics['episode_lengths'].extend(list(env.length_queue))
        
        return {
            'mean_error': np.mean(metrics['errors']) if metrics['errors'] else 0,
            'mean_cpu_time': np.mean(metrics['cpu_times']) if metrics['cpu_times'] else 0,
            'mean_reward': np.mean(metrics['rewards']) if metrics['rewards'] else 0,
            'mean_episode_length': np.mean(metrics['episode_lengths']) if metrics['episode_lengths'] else 0
        }
    
    def close(self):
        """Close all environments."""
        self.env.close()