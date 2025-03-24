# integrator.py
from dataclasses import dataclass
import rk_solver_cpp
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Optional, List, Tuple
from .combustion_problem import CombustionProblem
import cantera as ct
from enum import Enum
from multiprocessing import Process, Queue
    
@dataclass
class IntegratorConfig:
    """Configuration for the integrator."""
    integrator_list: List[str] = None
    tolerance_list: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.integrator_list is None:
            self.integrator_list = ['CPP_RK23', 'Radau', 'BDF']
        if self.tolerance_list is None:
            self.tolerance_list = [(1e-12, 1e-14), (1e-6, 1e-8)]
            
    def get_action_list(self):
        return [(integ, rtol, atol) 
                for integ in self.integrator_list 
                for rtol, atol in self.tolerance_list]

class CombustionStage(Enum):
    PREIGNITION = 0
    IGNITION = 1
    POSTIGNITION = 2

class ChemicalIntegrator:
    """Handles integration of chemical kinetics equations."""
    
    def __init__(self, 
                 problem: CombustionProblem,
                 config: Optional[IntegratorConfig] = None):
        """Initialize integrator."""
        self.problem = problem
        self.config = config or IntegratorConfig()
        
        self.gas = problem.gas
        self.timestep = problem.timestep
        self.P0 = problem.pressure
        self.state_change_threshold = problem.state_change_threshold
        
        self.reset_history()
        self.action_list = self.config.get_action_list()
        
        
    def reset_history(self):
        """Reset integration history."""
        self.history = {
            'times': [],
            'states': [],
            'temperatures': [],
            'species_profiles': {spec: [] for spec in self.problem.species_to_track},
            'cpu_times': [],
            'actions_taken': [],
            'success_flags': [],
            'errors': [],
            'stages': [],
            'stage_values': []
        }
        self.step_count = 0
        self.current_stage = CombustionStage.PREIGNITION
        self.t = 0.0
        # Reset gas state using equivalence ratio
        self.gas.set_equivalence_ratio(
            self.problem.phi, 
            self.problem.fuel, 
            self.problem.oxidizer
        )
        self.gas.TPX = self.problem.temperature, self.P0, self.problem.initial_mixture
        self.y = np.hstack([self.gas.T, self.gas.Y])
        
        self._store_state(self.y, 0.0, None, True, 0.0, 0.0, self.current_stage, 0.0)
        self.stage_changes = [False]
        self.stage_steps = {stage.value: 0 for stage in CombustionStage}
        self.stage_cpu_times = {stage.value: 0.0 for stage in CombustionStage}
        self.end_simulation = False
        
    def dydt(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute derivatives for the chemical system."""
        T = y[0]
        Y = y[1:]
        
        self.gas.TPY = T, self.P0, Y
        rho = self.gas.density_mass
        wdot = self.gas.net_production_rates
        cp = self.gas.cp_mass
        h = self.gas.partial_molar_enthalpies
        
        dTdt = -(np.dot(h, wdot) / (rho * cp))
        dYdt = wdot * self.gas.molecular_weights / rho
        
        return np.hstack([dTdt, dYdt])
    
    def _run_integration(self, method: str, rtol: float, atol: float, t_end: float, 
                        y_init: np.ndarray, result_queue: Queue):
        """Separate method for integration that can be pickled."""
        try:
            if method.lower().startswith('cpp_'):
                solver = rk_solver_cpp.RK23(
                    self.dydt, float(self.t), np.array(y_init), 
                    float(t_end), rtol=rtol, atol=atol
                )
                start_time = time.time()
                result = rk_solver_cpp.solve_ivp(solver, np.array(t_end))
                cpu_time = time.time() - start_time
                new_y = result['y'][-1]
                
        
            else:
                solver = ode(self.dydt)
                solver.set_integrator('vode', method=method, 
                                    with_jacobian=True,
                                    rtol=rtol, atol=atol)
                solver.set_initial_value(y_init, self.t)
                start_time = time.time()
                new_y = solver.integrate(t_end)
                cpu_time = time.time() - start_time
            
            result_queue.put({
                'success': True,
                'y': new_y,
                'cpu_time': cpu_time
            })
            
        except Exception as e:
            result_queue.put({
                'success': False,
                'error_message': str(e),
                'cpu_time': 0.0
            })

    def integrate_step(self, action_idx: int, time_limit: float = 3.0) -> Dict[str, Any]:
        """
        Perform one integration step with process-based timeout.
        
        Args:
            action_idx: Index of the integration method to use
            time_limit: Maximum time (in seconds) allowed for integration
        """
        method, rtol, atol = self.action_list[action_idx]
        t_end = self.t + self.timestep
        previous_state = self.y.copy()
        
        try:
            result_queue = Queue()
            
            # Create and start integration process
            process = Process(
                target=self._run_integration,
                args=(method, rtol, atol, t_end, previous_state, result_queue)
            )
            process.start()
            
            # Wait for result or timeout
            process.join(timeout=time_limit) # Wait for process to finish (time in seconds)
            
            # Check if process timed out
            if process.is_alive():
                process.terminate()
                process.join()  # Clean up
                
                print(f"Integration timed out after {time_limit}s using method {method} and tolerances {rtol}, {atol}")
                return {
                    'success': False,
                    'y': previous_state,
                    'cpu_time': time_limit,
                    'error': 0.0,
                    'message': 'Integration timed out',
                    'current_stage': self.current_stage,
                    'end_simulation': False,
                    'stage_cpu_times': self.stage_cpu_times,
                    'timed_out': True,
                    'previous_state': previous_state
                }
            
            # Get result if process completed
            if not result_queue.empty():
                integration_result = result_queue.get()
                
                if not integration_result['success']:
                    return {
                        'success': False,
                        'y': previous_state,
                        'cpu_time': 0.0,
                        'error': 0.0,
                        'message': integration_result.get('error_message', 'Integration failed'),
                        'current_stage': self.current_stage,
                        'end_simulation': False,
                        'stage_cpu_times': self.stage_cpu_times,
                        'timed_out': False,
                        'previous_state': previous_state
                    }
                
                new_y = integration_result['y']
                cpu_time = integration_result['cpu_time']
                
                # Process successful integration result
                ref_T = self.problem.reference_solution['temperatures'][self.step_count]
                T_current = new_y[0]
                error = abs(T_current/self.problem.temperature - ref_T/self.problem.temperature)
                
                stage_change, stage_value = self._state_changed_significantly(previous_state, new_y)
                self._store_state(new_y, t_end, action_idx, True, cpu_time, error, 
                                self.current_stage, stage_value)
                self.stage_changes.append(stage_change)
                
                # Handle stage transitions
                if self.stage_changes[-1] != self.stage_changes[-2]:
                    if self.current_stage == CombustionStage.PREIGNITION:
                        self.stage_steps[self.current_stage.value] = self.step_count
                        self.stage_cpu_times[self.current_stage.value] += np.sum(self.history['cpu_times'])
                        print(f"State changed to IGNITION at step {self.step_count}")
                        self.current_stage = CombustionStage.IGNITION
                    elif self.current_stage == CombustionStage.IGNITION:
                        self.stage_steps[self.current_stage.value] = self.step_count
                        self.stage_cpu_times[self.current_stage.value] += (
                            np.sum(self.history['cpu_times']) - 
                            self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                        )
                        print(f"State changed to POSTIGNITION at step {self.step_count}")
                        self.current_stage = CombustionStage.POSTIGNITION 
                
                # Handle end conditions        
                if self.current_stage == CombustionStage.POSTIGNITION:
                    if self.step_count > 2 * self.stage_steps[CombustionStage.IGNITION.value]:
                        print(f"Stopping simulation at step {self.step_count}")
                        self.end_simulation = True
                        self.stage_cpu_times[self.current_stage.value] += (
                            np.sum(self.history['cpu_times']) - 
                            self.stage_cpu_times[CombustionStage.IGNITION.value] - 
                            self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                        )
                
                self.step_count += 1
                
                if self.step_count >= self.problem.completed_steps:
                    self.end_simulation = True
                    print(f"Ending simulation at step {self.step_count}")
                    self.stage_cpu_times[self.current_stage.value] += (
                        np.sum(self.history['cpu_times']) - 
                        self.stage_cpu_times[CombustionStage.IGNITION.value] - 
                        self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                    )
                
                self.t = t_end
                self.y = new_y
                
                return {
                    'success': True,
                    'y': new_y,
                    'cpu_time': cpu_time,
                    'error': error,
                    'message': 'Success',
                    'current_stage': self.current_stage,
                    'end_simulation': self.end_simulation,
                    'stage_cpu_times': self.stage_cpu_times,
                    'timed_out': False,
                    'previous_state': previous_state
                }
        
        except Exception as e:
            cpu_time = 0.0
            print("An error occurred in the integration step - see traceback for details")
            import traceback
            print(traceback.format_exc())
            print(f"Error in integration step: {e}")
            
            return {
                'success': False,
                'y': previous_state,
                'cpu_time': cpu_time,
                'error': float('inf'),
                'message': str(e),
                'stage_cpu_times': self.stage_cpu_times,
                'timed_out': False,
                'previous_state': previous_state
            }
    
    def _state_changed_significantly(self, previous_state, current_state):
        """Check if state change is significant"""
        if isinstance(previous_state, list):
            previous_state = np.array(previous_state)
        if isinstance(current_state, list):
            current_state = np.array(current_state)
            
        state_change = np.linalg.norm(current_state - previous_state)
        return state_change > self.state_change_threshold, state_change
    
    def _store_state(self, y: np.ndarray, t: float, action_idx: Optional[int],
                     success: bool, cpu_time: float, error: float, stage: CombustionStage, stage_value: float):
        """Store the current state and integration results."""
        self.history['times'].append(t)
        self.history['states'].append(y.copy())
        self.history['temperatures'].append(y[0])
        
        for i, spec in enumerate(self.problem.species_to_track):
            idx = self.gas.species_index(spec)
            self.history['species_profiles'][spec].append(y[idx + 1])
        
        if action_idx is not None:
            self.history['actions_taken'].append(action_idx)
            self.history['success_flags'].append(success)
            self.history['cpu_times'].append(cpu_time)
            self.history['errors'].append(error)
            self.history['stages'].append(stage)
            self.history['stage_values'].append(stage_value)
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot integration history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        times = np.array(self.history['times'])
        ax1.plot(times, self.history['temperatures'], label='Computed')
        ax1.plot(times, self.problem.reference_solution['temperatures'][:len(times)], 
                '--', label='Reference')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Evolution')
        ax1.legend()
        
        ax2.plot(self.history['cpu_times'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('CPU Time (s)')
        ax2.set_title('Integration Time per Step')
        
        ax3.semilogy(self.history['errors'])
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Relative Error')
        ax3.set_title('Temperature Error Evolution')
        
        actions = np.array(self.history['actions_taken'])
        unique_actions, counts = np.unique(actions, return_counts=True)
        ax4.bar(unique_actions, counts)
        ax4.set_xlabel('Action Index')
        ax4.set_ylabel('Count')
        ax4.set_title('Integration Method Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
    def get_statistics(self) -> Dict[str, float]:
        """Get integration statistics."""
        return {
            'total_cpu_time': sum(self.history['cpu_times']),
            'average_cpu_time': np.mean(self.history['cpu_times']),
            'max_cpu_time': np.max(self.history['cpu_times']),
            'average_error': np.mean(self.history['errors']),
            'max_error': np.max(self.history['errors']),
            'success_rate': np.mean(self.history['success_flags']),
            'num_steps': len(self.history['times']) - 1
        }

