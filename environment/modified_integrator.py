# integrator.py
from collections import deque
from dataclasses import dataclass
# import rk_solver_cpp
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from enum import Enum
import importlib.util
import rk_solver_cpp
# Conditional import for sundials_py
try:
    import sundials_py
    SUNDIALS_AVAILABLE = True
except ImportError:
    SUNDIALS_AVAILABLE = False
    print("Warning: sundials_py not available. SUNDIALS integrators will not be available.")

# Conditional import for rk_solver_cpp
try:
    import rk_solver_cpp
    RK_SOLVER_CPP_AVAILABLE = True
except ImportError:
    RK_SOLVER_CPP_AVAILABLE = False
    print("Warning: rk_solver_cpp not available. Custom RK23 integrator will not be available.")

class IntegrationTimeoutError(Exception):
    pass

@dataclass
class IntegratorConfig:
    """Configuration for the integrator."""
    integrator_list: List[str] = None
    tolerance_list: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.integrator_list is None:
            # Only include SUNDIALS integrators if available
            if SUNDIALS_AVAILABLE:
                self.integrator_list = ['CPP_RK23', 'BDF', 'CVODE_BDF', 'ARKODE_ERK']
            else:
                self.integrator_list = ['CPP_RK23', 'BDF']
        
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


class IntegratorFactory:
    """Factory class to create integrators based on method name."""
    
    @staticmethod
    def create_integrator(method: str, 
                         dydt_func: Callable, 
                         t0: float, 
                         y0: np.ndarray, 
                         t_end: float, 
                         rtol: float, 
                         atol: Union[float, np.ndarray],
                         system_size: Optional[int] = None,
                         jacobian_func: Optional[Callable] = None):
        """
        Create an integrator based on the method name.
        
        Args:
            method: Name of the integration method
            dydt_func: ODE right-hand side function
            t0: Initial time
            y0: Initial state
            t_end: End time
            rtol: Relative tolerance
            atol: Absolute tolerance (scalar or array)
            system_size: Size of the ODE system (needed for SUNDIALS)
            jacobian_func: Optional Jacobian function for implicit methods
            
        Returns:
            Integrator object
        """
        #print(f"********************************* Method name: {method}******************************************")
        # Make sure y0 is a numpy array
        y0 = np.asarray(y0, dtype=float)
        
        # Use system size if provided, otherwise get from y0
        if system_size is None:
            system_size = len(y0)
        
        # Convert scalar atol to array if needed
        if np.isscalar(atol):
            atol_array = np.ones_like(y0) * atol
        else:
            atol_array = np.asarray(atol, dtype=float)
        
        # Create custom RK23 solver from C++
        if method.lower().startswith('cpp_'):
            if not RK_SOLVER_CPP_AVAILABLE:
                raise ImportError("rk_solver_cpp not available")
                
            return rk_solver_cpp.RK23(
                dydt_func, float(t0), y0, 
                float(t_end), rtol=rtol, atol=atol
            )
            
        # Create CVODE solver
        elif method.lower().startswith('cvode_'):
            if not SUNDIALS_AVAILABLE:
                raise ImportError("sundials_py not available")
            
            # Extract the specific method after 'CVODE_'
            cvode_method = method.split('_')[1]
            
            # Determine iteration type
            if cvode_method.upper() == 'BDF':
                iter_type = sundials_py.cvode.IterationType.NEWTON
            elif cvode_method.upper() == 'ADAMS':
                iter_type = sundials_py.cvode.IterationType.FUNCTIONAL
            else:
                iter_type = sundials_py.cvode.IterationType.NEWTON
            
            # Create CVODE solver
            solver = sundials_py.cvode.CVodeSolver(
                system_size=system_size,
                rhs_fn=dydt_func,
                iter_type=iter_type
            )
            
            # Initialize solver
            solver.initialize(y0, t0, rtol, atol_array)
            
            # Set Jacobian if provided
            if jacobian_func is not None and iter_type == sundials_py.cvode.IterationType.NEWTON:
                solver.set_jacobian(jacobian_func)
                
            # # # Set maximum number of steps to a reasonable value
            # if cvode_method.upper() != 'BDF':
            #     solver.set_max_num_steps(100000)
            
            return solver
        
        # Create ARKODE solver
        elif method.lower().startswith('arkode_'):
            if not SUNDIALS_AVAILABLE:
                raise ImportError("sundials_py not available")
            
            # Extract the specific method after 'ARKODE_'
            method_parts = method.split('_')
            if len(method_parts) < 2:
                raise ValueError(f"Invalid ARKODE method name: {method}")
                
            method_name = "_".join(method_parts[1:])  # Join all parts after ARKODE_
            
            # Determine if method is explicit, implicit, or IMEX
            is_implicit = any(keyword in method_name.upper() for keyword in 
                             ['DIRK', 'SDIRK', 'BILLINGTON', 'TRBDF2', 'KVAERNO', 'CASH'])
            is_imex = 'IMEX' in method_name.upper()
            
            # Set explicit/implicit functions
            if is_implicit and not is_imex:
                # Pure implicit method - put everything in implicit part
                def explicit_fn(t, y):
                    return np.zeros_like(y)
                    
                implicit_fn = dydt_func
            elif is_imex:
                # IMEX method requires splitting the function - user should provide
                raise ValueError("IMEX methods require manually split explicit/implicit functions")
            else:
                # Pure explicit method
                explicit_fn = dydt_func
                implicit_fn = None
            
            # Map method name to Butcher table
            butcher_table = IntegratorFactory._get_butcher_table(method_name)
            
            # Create ARKODE solver
            solver = sundials_py.arkode.ARKodeSolver(
                system_size=system_size,
                explicit_fn=explicit_fn if not is_imex else dydt_func,
                implicit_fn=implicit_fn,
                butcher_table=butcher_table,
                linsol_type=sundials_py.cvode.LinearSolverType.DENSE
            )
            
            # Initialize solver
            solver.initialize(y0, t0, rtol, atol_array)
            
            # Set Jacobian if provided and if method requires it
            if jacobian_func is not None and (is_implicit or is_imex):
                solver.set_jacobian(jacobian_func)
                
            # # Set maximum number of steps
            # solver.set_max_num_steps(100000)
            
            return solver
        
        # Use SciPy's ODE solver as fallback
        else:
            #print(f"********************************* OTHRTNMethod name: {method}******************************************")
            try:
                from scipy.integrate import ode
                
                solver = ode(dydt_func)
                solver.set_integrator('vode', method=method, 
                                    with_jacobian=jacobian_func is not None,
                                    rtol=rtol, atol=atol)
                solver.set_initial_value(y0, t0)
                
                # Set Jacobian if provided
                if jacobian_func is not None:
                    solver.set_jac_params(jacobian_func)
                    
                return solver
                
            except ImportError:
                raise ImportError("scipy.integrate not available and no other suitable integrator found")
    
    @staticmethod
    def _get_butcher_table(method_name):
        #print(f"********************************* Method name: {method_name}******************************************")
        """Map method name to appropriate Butcher table enum."""
        if not SUNDIALS_AVAILABLE:
            return None
            
        # Explicit methods
        if 'HEUN' in method_name.upper():
            return sundials_py.arkode.ButcherTable.HEUN_EULER_2_1_2
        elif 'BOGACKI' in method_name.upper():
            return sundials_py.arkode.ButcherTable.BOGACKI_SHAMPINE_4_2_3
        elif 'ARK324L2SA_ERK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK324L2SA_ERK_4_2_3
        elif 'ZONNEVELD' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ZONNEVELD_5_3_4
        elif 'ARK436L2SA_ERK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK436L2SA_ERK_6_3_4
        elif 'ARK437L2SA_ERK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK437L2SA_ERK_7_3_4
        elif 'ARK548L2SA_ERK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5
        elif 'VERNER' in method_name.upper():
            return sundials_py.arkode.ButcherTable.VERNER_8_5_6
        elif 'FEHLBERG' in method_name.upper():
            return sundials_py.arkode.ButcherTable.FEHLBERG_13_7_8
            
        # Implicit methods
        elif 'SDIRK_5_3_4' in method_name.upper():
            return sundials_py.arkode.ButcherTable.SDIRK_5_3_4
        elif 'SDIRK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.SDIRK_2_1_2
        elif 'BILLINGTON' in method_name.upper():
            return sundials_py.arkode.ButcherTable.BILLINGTON_3_3_2
        elif 'TRBDF2' in method_name.upper():
            return sundials_py.arkode.ButcherTable.TRBDF2_3_3_2
        elif 'ARK324L2SA_DIRK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK324L2SA_DIRK_4_2_3
        elif 'KVAERNO_7_4_5' in method_name.upper():
            return sundials_py.arkode.ButcherTable.KVAERNO_7_4_5
        elif 'KVAERNO' in method_name.upper():
            return sundials_py.arkode.ButcherTable.KVAERNO_4_2_3
        elif 'CASH_5_3_4' in method_name.upper():
            return sundials_py.arkode.ButcherTable.CASH_5_3_4
        elif 'CASH' in method_name.upper():
            return sundials_py.arkode.ButcherTable.CASH_5_2_4
        elif 'ARK436L2SA_DIRK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK436L2SA_DIRK_6_3_4
        elif 'ARK437L2SA_DIRK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK437L2SA_DIRK_7_3_4
        elif 'ARK548L2SA_DIRK' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK548L2SA_DIRK_8_4_5
            
        # IMEX pairs
        elif 'IMEX324' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK324L2SA_ERK_4_2_3_DIRK_4_2_3
        elif 'IMEX436' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK436L2SA_ERK_6_3_4_DIRK_6_3_4
        elif 'IMEX437' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK437L2SA_ERK_7_3_4_DIRK_7_3_4
        elif 'IMEX548' in method_name.upper():
            return sundials_py.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5_DIRK_8_4_5
            
        # Default
        else:
            print(f"Warning: Unknown ARKODE method '{method_name}', defaulting to ARK436L2SA_ERK")
            return sundials_py.arkode.ButcherTable.ARK436L2SA_ERK_6_3_4


    @staticmethod
    def solve(integrator, t_end: float):
        """
        Solve the ODE system with the given integrator.
        
        Args:
            integrator: Integrator object
            t_end: End time
            
        Returns:
            Dictionary with results
        """
        method_type = type(integrator).__module__
        
        try:
            if 'rk_solver_cpp' in method_type:
                # CPP_RK* solvers
                result = rk_solver_cpp.solve_ivp(integrator, np.array(t_end))
                if result['success']:
                    return {
                        'success': True,
                        'y': result['y'][-1],
                        'message': 'Success'
                    }
                else:
                    return {
                        'success': False,
                        'message': result.get('message', 'Unknown error in RK solver')
                    }
            
            if SUNDIALS_AVAILABLE:
                if 'sundials_py.cvode' in method_type:
                    # CVODE solver
                    try:
                        y_final = integrator.solve_to(t_end)
                        return {
                            'success': True,
                            'y': y_final,
                            'message': 'Success'
                        }
                    except Exception as e:
                        return {
                            'success': False,
                            'message': str(e)
                        }
                
                elif 'sundials_py.arkode' in method_type:
                    # ARKODE solver
                    try:
                        y_final = integrator.solve_to(t_end)
                        return {
                            'success': True,
                            'y': y_final,
                            'message': 'Success'
                        }
                    except Exception as e:
                        return {
                            'success': False,
                            'message': str(e)
                        }
            
            # SciPy ODE solver
            new_y = integrator.integrate(t_end)
            return {
                'success': integrator.successful(),
                'y': new_y if integrator.successful() else None,
                'message': 'Success' if integrator.successful() else 'Integration failed'
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f"Integration error: {str(e)}"
            }


class ChemicalIntegrator:
    """Handles integration of chemical kinetics equations."""
    
    def __init__(self, 
                 problem,  # CombustionProblem
                 config: Optional[IntegratorConfig] = None):
        """Initialize integrator."""
        self.problem = problem
        self.config = config or IntegratorConfig()
        
        self.gas = problem.gas
        self.timestep = problem.timestep
        self.P0 = problem.P0 if hasattr(problem, 'P0') else 101325  # Default to one atm
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
        self.temperature_queue = deque(maxlen=10)
    
        self.gas.TP = self.problem.temperature, self.P0
        self.gas.set_equivalence_ratio(self.problem.phi, self.problem.fuel, self.problem.oxidizer)
        self.y = np.hstack([self.gas.T, self.gas.Y])
        
        self._store_state(self.y, 0.0, None, True, 0.0, 0.0, self.current_stage, 0.0)
        self.stage_changes = [False]
        self.stage_steps = {stage.value: 0 for stage in CombustionStage}
        self.stage_cpu_times = {stage.value: 0.0 for stage in CombustionStage}
        self.end_simulation = False
        self.previous_temperature = self.problem.temperature
        
    def dydt(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute derivatives for the chemical system."""
        try:
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
        except Exception as e:
            print(f"[ERROR] : dydt failed with error {e}")
            return np.zeros_like(y)
        
    def check_steady_state(self, temperature_queue, initial_temperature, tolerance=0.1, increase_factor=1.2):
        """Check if steady state is reached based on temperature standard deviation and change from initial."""
        if len(temperature_queue) < 10:
            return False
            
        mean_temperature = np.mean(temperature_queue)
        std_temperature = np.std(temperature_queue)
        
        if mean_temperature > increase_factor*initial_temperature and std_temperature < tolerance:
            print(f"Steady state reached at step {self.step_count} with mean temperature {mean_temperature} and std {std_temperature}")
            return True
        else:
            return False
    
    def integrate_step(self, action_idx: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform one integration step with timeout.
        
        Args:
            action_idx (int): Index of the integration action to perform
            timeout (float): Maximum time in seconds allowed for integration
        """
        method, rtol, atol = self.action_list[action_idx]
        # print(f"Integrating step {self.step_count} with method {method}")
        t_end = self.t + self.timestep
        previous_state = self.y.copy()
        self.temperature_queue.append(self.y[0])
        self.step_count += 1
        
        if timeout is None:
            timeout = 0.1
        
        # Create event flag and result queue for thread communication
        integration_done = threading.Event()
        result_queue = queue.Queue()
        
        def integration_worker():
            try:
                start_time = time.time()
                
                # Create integrator using factory
                try:
                    integrator = IntegratorFactory.create_integrator(
                        method=method,
                        dydt_func=self.dydt,
                        t0=self.t,
                        y0=self.y,
                        t_end=t_end,
                        rtol=rtol,
                        atol=atol,
                        system_size=len(self.y)
                    )
                except Exception as e:
                    result_queue.put({
                        'success': False,
                        'cpu_time': time.time() - start_time,
                        'error': float('inf'),
                        'message': f"Failed to create integrator: {str(e)}"
                    })
                    return
                
                # Solve ODE system
                result = IntegratorFactory.solve(integrator, t_end)
                cpu_time = time.time() - start_time
                
                if not integration_done.is_set():  # Only process results if not timed out
                    # Add CPU time to the result
                    result['cpu_time'] = cpu_time
                    
                    # Calculate error if we have reference data
                    if result['success'] and 'y' in result:
                        new_y = result['y']
                        if hasattr(self.problem, 'reference_solution') and self.step_count < len(self.problem.reference_solution['temperatures']):
                            ref_T = self.problem.reference_solution['temperatures'][self.step_count]
                            T_current = new_y[0]
                            # print(f"Current Temperature: {T_current}")
                            # print(f"Reference Temperature: {ref_T}")
                            error = abs(T_current/self.problem.temperature - ref_T/self.problem.temperature)
                        else:
                            error = 0.0
                        result['error'] = error
                    else:
                        # For failed integrations
                        result['error'] = float('inf')
                    
                    result_queue.put(result)
            except Exception as e:
                if not integration_done.is_set():
                    result_queue.put({
                        'success': False,
                        'cpu_time': time.time() - start_time,
                        'error': float('inf'),
                        'message': f"Integration thread error: {str(e)}"
                    })
        
        # Start integration in separate thread
        integration_thread = threading.Thread(target=integration_worker)
        integration_thread.daemon = True
        start_time = time.time()
        integration_thread.start()
        
        # Wait for either completion or timeout
        integration_thread.join(timeout=timeout)
        
        if integration_thread.is_alive():
            # Integration took too long
            integration_done.set()  # Signal the thread to stop
            return {
                'success': False,
                'cpu_time': time.time() - start_time,
                'error': float('inf'),
                'message': f'Integration timed out after {timeout} seconds',
                'current_stage': self.current_stage,
                'end_simulation': False,
                'stage_cpu_times': self.stage_cpu_times,
                'timed_out': True
            }
        
        # Get the result from the queue
        try:
            result = result_queue.get_nowait()
            
            # Process stage changes and updates (only if integration was successful)
            if result['success'] and 'y' in result:
                new_y = result['y']
                stage_change, stage_value = self._state_changed_significantly(previous_state, new_y)
                self._store_state(new_y, t_end, action_idx, result['success'], 
                                result.get('cpu_time', 0.0), result.get('error', float('inf')), 
                                self.current_stage, stage_value)
                self.stage_changes.append(stage_change)

                if len(self.stage_changes) >= 2 and self.stage_changes[-1] != self.stage_changes[-2] and self.history['temperatures'][-1] > self.previous_temperature:
                    if self.current_stage == CombustionStage.PREIGNITION:
                        self.stage_steps[self.current_stage.value] = self.step_count
                        self.stage_cpu_times[self.current_stage.value] += np.sum(self.history['cpu_times'])
                        print(f"State changed to IGNITION at step {self.step_count}")
                        self.current_stage = CombustionStage.IGNITION
                    elif self.current_stage == CombustionStage.IGNITION:
                        self.stage_steps[self.current_stage.value] = self.step_count
                        self.stage_cpu_times[self.current_stage.value] += np.sum(self.history['cpu_times']) - self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                        print(f"State changed to POSTIGNITION at step {self.step_count}")
                        self.current_stage = CombustionStage.POSTIGNITION 

                if self.current_stage == CombustionStage.POSTIGNITION:
                    if self.step_count > 2 * self.stage_steps[CombustionStage.IGNITION.value]:
                        print(f"Stopping simulation at step {self.step_count}")
                        self.end_simulation = True
                        self.stage_cpu_times[self.current_stage.value] += np.sum(self.history['cpu_times']) - self.stage_cpu_times[CombustionStage.IGNITION.value] - self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                        
                self.t = t_end
                self.y = new_y
                steady_state = self.check_steady_state(self.temperature_queue, self.problem.temperature)
                if hasattr(self.problem, 'completed_steps') and self.step_count == self.problem.completed_steps or steady_state:
                    self.end_simulation = True
                    self.stage_cpu_times[self.current_stage.value] += np.sum(self.history['cpu_times']) - self.stage_cpu_times[CombustionStage.IGNITION.value] - self.stage_cpu_times[CombustionStage.PREIGNITION.value]
            else:
                # For failed integration, store the current state again but mark as failed
                self._store_state(self.y, self.t, action_idx, False, 
                                result.get('cpu_time', 0.0), float('inf'), 
                                self.current_stage, 0.0)
                print(f"Integration failed: {result.get('message', 'Unknown error')}")
            
            result.update({
                'current_stage': self.current_stage,
                'end_simulation': self.end_simulation,
                'stage_cpu_times': self.stage_cpu_times,
                'timed_out': False
            })
            
            # Important: Update previous temperature even if integration failed
            if len(self.history['temperatures']) > 0:
                self.previous_temperature = self.history['temperatures'][-1]
                
            return result
            
        except queue.Empty:
            # This should rarely happen if the thread completed, but just in case
            return {
                'success': False,
                'cpu_time': time.time() - start_time,
                'error': float('inf'),
                'message': 'Integration failed to return results',
                'current_stage': self.current_stage,
                'end_simulation': False,
                'stage_cpu_times': self.stage_cpu_times,
                'timed_out': False
            }
    
    def _state_changed_significantly(self, previous_state, current_state):
        """Check if state change is significant"""
        if isinstance(previous_state, list):
            previous_state = np.array(previous_state)
        if isinstance(current_state, list):
            current_state = np.array(current_state)
            
        state_change = np.linalg.norm(current_state - previous_state)
        return state_change > self.state_change_threshold, state_change
    
    # def _store_state(self, y: np.ndarray, t: float, action_idx: Optional[int],
    #                  success: bool, cpu_time: float, error: float, stage: CombustionStage, stage_value: float):
    #     """Store the current state and integration results."""
    #     self.history['times'].append(t)
    #     self.history['states'].append(y.copy())
    #     self.history['temperatures'].append(y[0])
        
    #     for i, spec in enumerate(self.problem.species_to_track):
    #         idx = self.gas.species_index(spec)
    #         self.history['species_profiles'][spec].append(y[idx + 1])
        
    #     if action_idx is not None:
    #         self.history['actions_taken'].append(action_idx)
    #         self.history['success_flags'].append(success)
    #         self.history['cpu_times'].append(cpu_time)
    #         self.history['errors'].append(error)
    #         self.history['stages'].append(stage)
    #         self.history['stage_values'].append(stage_value)

    # Add this method to track stages in the history buffer
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
            
            # Add to the history buffer in the environment as well
            if hasattr(self, 'env') and hasattr(self.env, 'history_buffer'):
                self.env.history_buffer['stages'].append(stage)
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot integration history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        times = np.array(self.history['times'])
        ax1.plot(times, self.history['temperatures'], label='Computed')
        
        # Plot reference solution if available
        if hasattr(self.problem, 'reference_solution'):
            ref_times = times[:len(self.problem.reference_solution['temperatures'])]
            ax1.plot(ref_times, self.problem.reference_solution['temperatures'][:len(times)], 
                    '--', label='Reference')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Evolution')
        ax1.legend()
        
        ax2.plot(self.history['cpu_times'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('CPU Time (s)')
        ax2.set_title('Integration Time per Step')
        
        # plot the action history
        if self.history['actions_taken']:
            ax3.plot(self.history['actions_taken'])
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Action Index')
            ax3.set_title('Action History')
            
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
        
    def plot_methods_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of different integration methods."""
        if not self.history['actions_taken']:
            print("No integration methods to compare")
            return
            
        # Get unique action indices
        unique_actions = np.unique(self.history['actions_taken'])
        n_methods = len(unique_actions)
        
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # For each method, calculate average CPU time and error
        method_names = []
        avg_cpu_times = []
        avg_errors = []
        success_rates = []
        
        for action_idx in unique_actions:
            method_name, rtol, atol = self.action_list[action_idx]
            method_names.append(f"{method_name}\nrtol={rtol}, atol={atol}")
            
            # Get mask for this method
            mask = np.array(self.history['actions_taken']) == action_idx
            
            # Calculate metrics
            if np.any(mask):
                avg_cpu_times.append(np.mean(np.array(self.history['cpu_times'])[mask]))
                avg_errors.append(np.mean(np.array(self.history['errors'])[mask]))
                success_rates.append(np.mean(np.array(self.history['success_flags'])[mask]))
            else:
                avg_cpu_times.append(0)
                avg_errors.append(0)
                success_rates.append(0)
        
        # Plot average CPU time
        axs[0].bar(range(n_methods), avg_cpu_times)
        axs[0].set_xticks(range(n_methods))
        axs[0].set_xticklabels(method_names, rotation=45, ha='right')
        axs[0].set_ylabel('Average CPU Time (s)')
        axs[0].set_title('Average CPU Time by Method')
        
        # Plot average error
        axs[1].bar(range(n_methods), avg_errors)
        axs[1].set_xticks(range(n_methods))
        axs[1].set_xticklabels(method_names, rotation=45, ha='right')
        axs[1].set_ylabel('Average Error')
        axs[1].set_title('Average Error by Method')
        
        # Plot success rate
        axs[2].bar(range(n_methods), success_rates)
        axs[2].set_xticks(range(n_methods))
        axs[2].set_xticklabels(method_names, rotation=45, ha='right')
        axs[2].set_ylabel('Success Rate')
        axs[2].set_title('Success Rate by Method')
        axs[2].set_ylim(0, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
    def get_statistics(self) -> Dict[str, float]:
        """Get integration statistics."""
        return {
            'total_cpu_time': sum(self.history['cpu_times']) if self.history['cpu_times'] else 0,
            'average_cpu_time': np.mean(self.history['cpu_times']) if self.history['cpu_times'] else 0,
            'max_cpu_time': np.max(self.history['cpu_times']) if self.history['cpu_times'] else 0,
            'average_error': np.mean(self.history['errors']) if self.history['errors'] else 0,
            'max_error': np.max(self.history['errors']) if self.history['errors'] else 0,
            'success_rate': np.mean(self.history['success_flags']) if self.history['success_flags'] else 0,
            'num_steps': len(self.history['times']) - 1,
            'stage_steps': self.stage_steps,
            'stage_cpu_times': self.stage_cpu_times
        }