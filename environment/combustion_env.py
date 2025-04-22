import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
from collections import deque
import rk_solver_cpp
from scipy.integrate import ode


def integrate_chemistry(dydt_wrapper, y0, t_span, npoints, method='rk23', 
                       rtol=1e-6, atol=1e-8, with_jacobian=True):
    """
    Unified integration function for chemical kinetics simulation.
    See detailed docstring in original implementation.
    """
    try:
        if method.lower() == 'rk23' or method.lower() == 'cpp_rk23':
            
            solver = rk_solver_cpp.RK23(dydt_wrapper, float(t_span[0]), np.array(y0), float(t_span[1]), 
                                      rtol=rtol, atol=atol)
            start_time = time.time()
            result = rk_solver_cpp.solve_ivp(solver, np.array(t_span[1]))
            cpu_time = time.time() - start_time
            if result['success']:
                return {
                    'success': True,
                    'y': result['y'][-1],
                    'cpu_time': cpu_time,
                    'message': 'Success'
                }
            else:
                return {
                    'success': False,
                    'message': result['message']
                }
                
        elif method.lower() in ['adams', 'bdf']:
            r = ode(dydt_wrapper)
            r.set_integrator('vode', 
                           method=method,
                           with_jacobian=with_jacobian,
                           rtol=rtol, 
                           atol=atol)
            r.set_initial_value(y0, t_span[0])
            
            start_time = time.time()
            r.integrate(t_span[1])
            cpu_time = time.time() - start_time
            
            return {
                'success': True,
                'y': r.y,
                'cpu_time': cpu_time,
                'message': 'Success'
            }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }

def run_cantera_solver(
    mech_file,
    initial_state,
    end_time,
    tcfd,
    species_to_track,
    rtol=1e-10,
    atol=1e-20
):
    """Run Cantera solver to generate reference solution."""
    # Set up the reactor
    gas = ct.Solution(mech_file)
    gas.TPX = initial_state['T'], initial_state['P'], initial_state['X']
    
    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    sim.rtol = rtol
    sim.atol = atol

    # Pre-allocate arrays for results
    num_steps = int(end_time / tcfd) + 1
    times = np.zeros(num_steps)
    temperatures = np.zeros(num_steps)
    pressures = np.zeros(num_steps)
    species_profiles = {spec: np.zeros(num_steps) for spec in species_to_track}
    
    t = 0.0
    for i in range(num_steps):
        sim.advance(t)
        times[i] = t
        temperatures[i] = reactor.T
        pressures[i] = reactor.thermo.P
        for spec in species_to_track:
            species_profiles[spec][i] = reactor.thermo[spec].Y
        t += tcfd
    
    results = {
        'times': times,
        'temperatures': temperatures,
        'pressures': pressures,
        'species_profiles': species_profiles,
    }
    return results

class CombustionEnv(gym.Env):
    def __init__(self, 
                 mech_file, 
                 initial_state, 
                 end_time, 
                 tcfd, 
                 species_to_track,
                 Etol=1e-7,
                 features_config={
                     'temporal_features': False,
                     'species_features': False,
                     'stiffness_features': False,
                     'basic_features': True,
                     'include_dt_etol': False,
                     'add_net_production_rates': False
                 },
                 window_size=5,
                 feature_params={
                     'epsilon': 1e-10,
                     'clip_value': 1e-20
                 },
                 reward_weights={
                     'alpha': 0.8,  # time weight
                     'beta': 0.2,   # error weight
                     'gamma': 0.1   # stability weight
                 },
                 integrator_list=['CPP_RK23', 'Radau', 'BDF'],
                 tolerance_list=[(1e-12, 1e-14), (1e-6, 1e-8)],
                 state_change_threshold=1,
                 time_threshold=0.01,
                 integration_method='integrate'
                 ):
        super(CombustionEnv, self).__init__()
        
        # Basic environment setup
        self.mech_file = mech_file
        self.initial_state = initial_state
        self.end_time = end_time
        self.tcfd = tcfd
        self.species_to_track = species_to_track
        self.Etol = Etol
        self.time_threshold = time_threshold
        
        # Feature configuration
        self.features_config = features_config
        self.window_size = window_size
        self.feature_params = feature_params
        self.reward_weights = reward_weights
        self.integrator_list = integrator_list
        self.tolerance_list = tolerance_list
        
        self.integration_method = integration_method
        
        # Extract reward weights
        self.alpha = reward_weights.get('alpha', 0.8)
        self.beta = reward_weights.get('beta', 0.2)
        self.gamma = reward_weights.get('gamma', 0.1)
        
        # Setup Cantera
        self.gas = ct.Solution(self.mech_file)
        self.gas.TPX = self.initial_state['T'], self.initial_state['P'], self.initial_state['X']
        self.P0 = self.initial_state['P']
        self.num_species = self.gas.n_species
        self.species_names = self.gas.species_names
        self.state_change_threshold = state_change_threshold
        
        # Feature flags
        self.add_net_production_rates = features_config.get('add_net_production_rates', False)
        self.include_dt_etol = features_config.get('include_dt_etol', False)
        
        # Initialize components
        self._initialize_feature_components()
        self._setup_observation_space()
        self._define_action_space()
        
        # Compute reference solution
        self.reference_solution = self._compute_reference_solution()
        
        self.reset()
    
    def _initialize_feature_components(self):
        """Initialize components needed for selected features"""
        self.history_buffer = {}
        
        if self.features_config.get('temporal_features', False):
            self.history_buffer['temperature'] = deque(maxlen=self.window_size)
            self.history_buffer['gradients'] = deque(maxlen=self.window_size)
            
        if self.features_config.get('species_features', False):
            self.history_buffer['species'] = {
                spec: deque(maxlen=self.window_size) 
                for spec in self.species_to_track
            }
    
    def _setup_observation_space(self):
        """Dynamically set up observation space based on selected features"""
        obs_size = 0
        
        if self.features_config.get('basic_features', True):
            obs_size += len(self.species_to_track) + 1
            if self.add_net_production_rates:
                obs_size += len(self.species_to_track)
            
        if self.features_config.get('temporal_features', False):
            obs_size += 4
            
        if self.features_config.get('species_features', False):
            obs_size += len(self.species_to_track) * 3
            
        if self.features_config.get('stiffness_features', False):
            obs_size += 3
            
        if self.include_dt_etol:
            obs_size += 2
        print(f"Observation space size: {obs_size}")
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def _define_action_space(self):
        """Define the action space for integrator and tolerance selection"""
        self.action_list = []
        for integrator in self.integrator_list:
            for rtol, atol in self.tolerance_list:
                self.action_list.append((integrator, rtol, atol))
        
        self.action_space = spaces.Discrete(len(self.action_list))
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.t = 0.0
        self.gas.TPX = self.initial_state['T'], self.P0, self.initial_state['X']
        self.y = np.hstack([self.gas.T, self.gas.Y])
        self.cummulative_cpu_time = 0
        self.cummulative_reward = 0
        self.num_steps = int(self.end_time / self.tcfd)
        
        # Reset storage arrays
        self.temperatures = []
        self.species_profiles = {spec: [] for spec in self.species_to_track}
        self.cpu_times = []
        self.errors = []
        self.action_history = []
        self.previous_action = None
        self.action_distribution = np.zeros(len(self.action_list))
        self.rewards = []
        self.time_rewards = []
        self.error_rewards = []
        self.state_changes = []
        self.state_change_values = []
        self.cummulative_temperature_error = 0
        self.cummulative_species_error = 0
        self.combustion_stage = 'pre-ignition'
        self.stage_cpu_times = {'pre-ignition': 0, 'ignition': 0, 'post-ignition': 0}
        # Reset history buffers
        self._initialize_feature_components()
        
        return self._get_observation(), {}

    def step(self, action):
        """Execute one environment step with support for both integration methods"""
        integrator, rtol, atol = self.action_list[action]
        
        def dydt(t, y):
            """ODE function for the combustion system"""
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

        try:
            start_time = time.time()
            
            if self.integration_method == 'solve_ivp':
                # Existing solve_ivp integration
                if integrator.startswith('CPP_'):
                    solver_class = rk_solver_cpp.RK23 if integrator == 'CPP_RK23' else rk_solver_cpp.RK45
                    solver = solver_class(dydt, float(self.t), np.array(self.y), 
                                       float(self.t + self.tcfd), rtol=rtol, atol=atol)
                    result = rk_solver_cpp.solve_ivp(solver, np.array(float(self.t + self.tcfd)))
                    
                    if result['success']:
                        sol_y = result['y'][-1]
                        success = True
                    else:
                        raise RuntimeError(result['message'])
                else:
                    sol = solve_ivp(dydt, (self.t, self.t + self.tcfd), self.y,
                                  method=integrator, t_eval=[self.t + self.tcfd],
                                  rtol=rtol, atol=atol)
                    sol_y = sol.y[:, -1]
                    success = sol.success
                
            else:  # 'integrate' method
                result = integrate_chemistry(
                    dydt_wrapper=dydt,
                    y0=self.y,
                    t_span=(self.t, self.t + self.tcfd),
                    npoints=1,  # Single step
                    method=integrator,
                    rtol=rtol,
                    atol=atol
                )
                if result['success']:
                    sol_y = result['y']
                    success = True
                else:
                    raise RuntimeError(result['message'])
            
            cpu_time = time.time() - start_time
            
            if not success:
            
                raise RuntimeError("Integration failed")
            
            # Update state and compute metrics
            previous_state = self.y.copy()
            self.y = sol_y
            self.t += self.tcfd
            
            
            # Compute metrics
            state_changed, state_change_value = self._state_changed_significantly(
                previous_state, self.y)
            self.state_changes.append(state_changed)
            self.state_change_values.append(state_change_value)
            
            # Extract current state
            T_current = self.y[0]
            Y_current = self.y[1:]
            
            # Compute error
            ref_T = self.reference_solution['temperatures'][self.current_step]
            ref_T_normalized = ref_T / self.initial_state['T']
            T_current_normalized = T_current / self.initial_state['T']
            error = np.abs(T_current_normalized - ref_T_normalized)
            
            # Update storage
            self.errors.append(error)
            self.temperatures.append(T_current)
            for spec in self.species_to_track:
                idx = self.gas.species_index(spec)
                self.species_profiles[spec].append(Y_current[idx])
            
            # Compute reward
            reward, time_reward, error_reward = self._compute_reward(cpu_time, error)
            
            # Update tracking variables
            self.time_rewards.append(time_reward)
            self.error_rewards.append(error_reward)
            self.cummulative_cpu_time += cpu_time
            self.cummulative_reward += reward
            self.rewards.append(reward)
            self.cpu_times.append(cpu_time)
            self.action_history.append(action)
            self.action_distribution[action] += 1
            
            # Check termination
            self.current_step += 1
            done = self.current_step >= self.num_steps
            if done and self.cummulative_cpu_time/self.num_steps > 0.05:
                reward -= (self.cummulative_cpu_time/self.num_steps - 0.05) * 100
            
            # Get observation and info
            observation = self._get_observation()
            temperature_error, species_error = self.calculate_error()
            self.cummulative_temperature_error += temperature_error
            
            if self.current_step > 1:
                if self.state_changes[-1] != self.state_changes[-2]:
                    print(f"State changed at step {self.current_step-1}")
                    if self.combustion_stage == 'pre-ignition':
                        self.combustion_stage = 'ignition'
                        self.stage_cpu_times['pre-ignition'] = self.cummulative_cpu_time
                    else:
                        self.combustion_stage = 'post-ignition'
                        self.stage_cpu_times['ignition'] = self.cummulative_cpu_time - self.stage_cpu_times['pre-ignition']
                        
                else:
                    self.stage_cpu_times['pre-ignition'] += cpu_time
                        
            if self.current_step == self.num_steps and self.combustion_stage == 'post-ignition':
                self.stage_cpu_times['post-ignition'] = self.cummulative_cpu_time - self.stage_cpu_times['ignition'] - self.stage_cpu_times['pre-ignition']
            
            # self.cummulative_species_error += species_error
            info = {
                'cpu_time': cpu_time,
                'error': error,
                'state_changed': state_changed,
                'state_change_value': state_change_value,
                'cummulative_cpu_time': self.cummulative_cpu_time,
                'cummulative_reward': self.cummulative_reward,
                'temperature_error': temperature_error,
                'species_error': species_error,
                'action_history': self.action_history,
                'action_distribution': self.action_distribution,
                'time_reward': time_reward,
                'error_reward': error_reward,
                'cummulative_temperature_error': self.cummulative_temperature_error,
                # 'cummulative_species_error': self.cummulative_species_error
            }
            
            return observation, reward, done, False, info
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in step: {e}")
            return (self._get_observation(), -300.0, True, False, 
                    {'error': -1000, 'cpu_time': time.time() - start_time,
                     'state_changed': False, 'state_change_value': 0.0,
                     'time_reward': 0.00001, 'error_reward': 0.00001,
                     'cummulative_temperature_error': None,
                     'cummulative_reward': None,
                     'cummulative_cpu_time': None
                    })

    def _state_changed_significantly(self, previous_state, current_state):
        """Check if state change is significant"""
        if isinstance(previous_state, list):
            previous_state = np.array(previous_state)
        if isinstance(current_state, list):
            current_state = np.array(current_state)
            
        state_change = np.linalg.norm(current_state - previous_state)
        return state_change > self.state_change_threshold, state_change
    
    def calculate_error(self):
        """Calculate error compared to reference solution"""
        ref_T = self.reference_solution['temperatures'][self.current_step-1]
        actual_T = self.temperatures[self.current_step-1]
        temperature_error = np.abs(actual_T - ref_T)

        species_error = {}
        ref_profiles = self.reference_solution['species_profiles']
        for spec in self.species_to_track:
            ref_Y = ref_profiles[spec][self.current_step-1]
            actual_Y = self.species_profiles[spec][self.current_step-1]
            species_error[spec] = np.abs(actual_Y - ref_Y)
    
        return temperature_error, species_error
    
    def _compute_reward(self, cpu_time, error):
        """Compute reward based on CPU time and error"""
        epsilon = self.feature_params['epsilon']
        
        # Base time reward
        time_reward = 1 / (1 + cpu_time / self.time_threshold)
        
        # Error reward with dynamic scaling
        error_ratio = error / (self.Etol + epsilon)
        if error <= self.Etol:
            error_reward = 1 - (error_ratio / 2)
        else:
            error_reward = max(0, 1 - error_ratio)
        
        # Get stability factor if temporal features are enabled
        if self.features_config.get('temporal_features', False) and len(self.history_buffer['gradients']) > 0:
            dT_dt = self.history_buffer['gradients'][-1]
            rate_magnitude = np.abs(dT_dt)
        else:
            rate_magnitude = 0.0
            
        # Dynamic weight adjustment
        w_time = self.alpha / (1 + np.log1p(rate_magnitude))
        w_error = self.beta * (1 + np.log1p(rate_magnitude))
        
        # Combined reward
        reward = w_time * time_reward + w_error * error_reward
        
        return reward, time_reward, error_reward

    def _compute_temporal_features(self):
        """Compute temporal features if enabled"""
        if not self.features_config.get('temporal_features', False):
            return np.array([])
            
        if len(self.history_buffer['temperature']) < 2:
            return np.zeros(4)
            
        temp_array = np.array(self.history_buffer['temperature'])
        dT_dt = np.gradient(temp_array) / self.tcfd
        d2T_dt2 = np.gradient(dT_dt) / self.tcfd
        
        max_rate = np.max(np.abs(dT_dt))
        mean_rate = np.mean(np.abs(dT_dt))
        rate_variability = np.std(dT_dt)
        acceleration = np.mean(np.abs(d2T_dt2))
        
        return np.array([
            np.log1p(max_rate),
            np.log1p(mean_rate),
            np.log1p(rate_variability),
            np.log1p(acceleration)
        ])
    
    def _compute_species_features(self):
        """Compute species features if enabled"""
        if not self.features_config.get('species_features', False):
            return np.array([])
            
        features = []
        for spec in self.species_to_track:
            if len(self.history_buffer['species'][spec]) < 2:
                features.extend([0.0, 0.0, 0.0])
                continue
                
            spec_array = np.array(self.history_buffer['species'][spec])
            dY_dt = np.gradient(spec_array) / self.tcfd
            
            features.extend([
                np.log1p(np.max(np.abs(dY_dt))),
                np.log1p(np.mean(np.abs(dY_dt))),
                np.log1p(np.std(dY_dt))
            ])
            
        return np.array(features)
    
    def _compute_stiffness_features(self):
        """Compute stiffness features if enabled"""
        if not self.features_config.get('stiffness_features', False):
            return np.array([])
            
        epsilon = self.feature_params['epsilon']
        
        # Compute rate ratios
        self.gas.TPY = self.y[0], self.gas.P, self.y[1:]
        net_rates = self.gas.net_production_rates
        significant_rates = np.abs(net_rates[np.abs(net_rates) > epsilon])
        
        if len(significant_rates) > 0:
            rate_ratio = np.max(significant_rates) / (np.min(significant_rates) + epsilon)
        else:
            rate_ratio = 1.0
            
        # Temperature gradient
        dT_dt = (self.history_buffer['gradients'][-1] 
                if self.features_config.get('temporal_features', False) and len(self.history_buffer['gradients']) > 0
                else 0.0)
        
        # Stiffness ratio from eigenvalues
        eigenvals = np.linalg.eigvals(self.gas.jacobian)
        stiffness_ratio = np.max(np.abs(eigenvals.real)) / (np.min(np.abs(eigenvals.real)) + epsilon)
        
        return np.array([
            np.log1p(rate_ratio),
            np.log1p(np.abs(dT_dt)),
            np.log1p(stiffness_ratio)
        ])

    def _update_history(self):
        """Update history buffers based on enabled features"""
        if self.features_config.get('temporal_features', False):
            self.history_buffer['temperature'].append(self.y[0])
            
            if len(self.history_buffer['temperature']) > 1:
                dT_dt = ((self.history_buffer['temperature'][-1] - 
                         self.history_buffer['temperature'][-2]) / self.tcfd)
            else:
                dT_dt = 0.0
            self.history_buffer['gradients'].append(dT_dt)
            
        if self.features_config.get('species_features', False):
            for spec in self.species_to_track:
                idx = self.gas.species_index(spec)
                self.history_buffer['species'][spec].append(self.y[idx + 1])
    
    def _get_observation(self):
        """Get observation based on enabled features"""
        self._update_history()
        
        observation_parts = []
        
        # Basic features
        if self.features_config.get('basic_features', True):
            T = self.y[0]
            Y = self.y[1:]
            
            # Get species mass fractions
            Y_tracked = np.array([Y[self.gas.species_index(spec)] 
                                for spec in self.species_to_track])
            
            if self.add_net_production_rates:
                net_rates = np.array([self.gas.net_production_rates[self.gas.species_index(spec)] 
                                    for spec in self.species_to_track])
                Y_tracked = np.hstack([Y_tracked, net_rates])
            
            # Process species data
            Y_tracked = np.clip(Y_tracked, self.feature_params['clip_value'], None)
            Y_log = np.log(Y_tracked)
            Y_normalized = (Y_log - np.mean(Y_log)) / np.std(Y_log)
            
            # Temperature normalization
            T_normalized = T / self.initial_state['T']
            
            observation_parts.extend([Y_normalized, [T_normalized]])
            
        # Additional features
        if self.features_config.get('temporal_features', False):
            observation_parts.append(self._compute_temporal_features())
            
        if self.features_config.get('species_features', False):
            observation_parts.append(self._compute_species_features())
            
        if self.features_config.get('stiffness_features', False):
            observation_parts.append(self._compute_stiffness_features())
            
        if self.include_dt_etol:
            dt_log = np.log(self.tcfd)
            Etol_log = np.log(self.Etol)
            observation_parts.extend([[dt_log, Etol_log]])
        
        return np.hstack(observation_parts).astype(np.float32)
    
    def _compute_reference_solution(self):
        """Compute reference solution using Cantera solver"""
        print("Computing reference solution...")
        return run_cantera_solver(
            self.mech_file,
            self.initial_state,
            self.end_time,
            self.tcfd,
            self.species_to_track,
            rtol=1e-10,
            atol=1e-20
        )
    
    def compare_state_with_reference(self, save_path=None):
        """Compare current state with reference solution and plot results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 30))

        # Temperature comparison
        ref_temps = self.reference_solution['temperatures'][:self.current_step]
        ax1.plot(ref_temps, label='Reference Temperature')
        ax1.plot(self.temperatures, label='Actual Temperature')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Temperature (K)')
        ax1.legend()
        ax1.set_title('Temperature Comparison')

        # # Species profiles
        # for spec in self.species_to_track:
        #     ax2.plot(self.reference_solution['species_profiles'][spec][:self.current_step],
        #             label=f'Reference {spec}')
        #     ax2.plot(self.species_profiles[spec], label=f'Actual {spec}')
        # ax2.set_xlabel('Time Step')
        # ax2.set_ylabel('Species Mass Fractions')
        # ax2.legend()
        # ax2.set_title('Species Profiles Comparison')

        # Action history
        ax2.plot(self.action_history, 'k-')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Action Index')
        ax2.set_title('Action History')
        # add the action list as text on the top of the plot
        ax2.text(0.5, 1.05, '\n'.join([f'{i}: {action}' for i, action in enumerate(self.action_list)]), 
                 transform=ax2.transAxes, ha='center', va='bottom')

        # State changes
        ax3.plot(self.state_changes, 'r-')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('State Changed')
        ax3.set_title('State Changes')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def __copy__(self):
        """Create a copy of the environment"""
        new_env = CombustionEnv(
            mech_file=self.mech_file,
            initial_state=self.initial_state,
            end_time=self.end_time,
            tcfd=self.tcfd,
            species_to_track=self.species_to_track,
            Etol=self.Etol,
            features_config=self.features_config.copy(),
            window_size=self.window_size,
            feature_params=self.feature_params.copy(),
            reward_weights=self.reward_weights.copy(),
            integrator_list=self.integrator_list.copy(),
            tolerance_list=self.tolerance_list.copy(),
            state_change_threshold=self.state_change_threshold,
            time_threshold=self.time_threshold
        )
        new_env.reference_solution = self.reference_solution
        return new_env

    def compare_with_fixed_action(self, save_path=None):
        """ Copy the current environment four times and run each with each action and also run the reference solution"""
        action_list = np.arange(len(self.action_list))
        env_list = [self.__copy__() for _ in range(len(action_list))]
        all_cpu_times = [[] for _ in range(len(action_list))]
        all_errors = [[] for _ in range(len(action_list))]
        all_rewards = [[] for _ in range(len(action_list))]
        time_rewards = [[] for _ in range(len(action_list))]
        error_rewards = [[] for _ in range(len(action_list))]
        # reset the environments
        try:
            for env in env_list:
                env.reset()
            while True:
                for (i, env) in enumerate(env_list):
                    action = action_list[i]
                    next_state, reward, terminated, truncated, info = env.step(action)
                    all_cpu_times[i].append(info['cpu_time'])
                    all_errors[i].append(info['error'])
                    all_rewards[i].append(reward)
                    time_rewards[i].append(info['time_reward'])
                    error_rewards[i].append(info['error_reward'])
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
                
            # create a 2 x 2 subplot - compare the cpu time, error and reward for each action
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 30))
            for i, action in enumerate(action_list):
                ax1.plot(time_rewards[i], label=f'Time Reward for Action {action}')
                ax2.plot(error_rewards[i], label=f'Error Reward for Action {action}')
                ax3.plot(all_rewards[i], label=f'Reward for Action {action}')
                
            ax4.plot(self.action_history, '-o', label=f'Action History for Reference')
                
            ax1.plot(self.time_rewards, '-o', label=f'RL Policy Time Reward')
            ax2.plot(self.error_rewards, '-o', label=f'RL Policy Error Reward')
            ax3.plot(self.rewards, '-o', label=f'RL Policy Reward')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Time Reward')
            ax1.legend()
            ax1.set_title(f'Time Reward Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax2.legend()
            ax2.set_title(f'Error Reward Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Error Reward')
            ax3.legend()
            ax3.set_title(f'Reward Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax4.legend()
            ax4.set_title(f'Action History Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Action')
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 30))
            # plot the time reward and error reward for each action
            for i, action in enumerate(action_list):
                ax1.plot(np.log10(all_cpu_times[i]), label=f'CPU Time for Action {action}')
                ax2.plot(np.log10(all_errors[i]), label=f'Error for Action {action}')
                ax3.plot(env_list[i].temperatures, label=f'Temperature for Action {action}')
            ax1.plot(np.log10(self.cpu_times), '-o', label=f'CPU Time for Reference')
            ax2.plot(np.log10(self.errors), '-o', label=f'Error for Reference')
            ax3.plot(self.temperatures, '-o', label=f'Temperature for Reference')
            ax1.legend()
            ax1.set_title(f'CPU Time Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('-log10(CPU Time)')
            ax2.legend()
            ax2.set_title(f'Error Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('log10(Error)')
            ax3.legend()
            ax3.set_title(f'Temperature Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Temperature (K)')
            ax3.legend()
            
            if save_path:
                save_path_ = save_path.replace('.png', '_comparison.png')
                plt.savefig(save_path_)
            else:
                plt.show()
            plt.close()
            
            # plot the total cpu time, total error for each action in a bar chart
        
            total_cpu_times = [env_list[i].cummulative_cpu_time for i in range(len(action_list))]
            total_errors = [np.sum(env_list[i].errors) for i in range(len(action_list))]
            
            total_cpu_times.append(self.cummulative_cpu_time)
            total_errors.append(np.sum(self.errors))
            action_lists = [str(action) for action in env.action_list]
            action_lists.append('RL Policy')
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 30), dpi=300)
            
            ax1.bar(action_lists, total_cpu_times, label='Total CPU Time')
            # write the total cpu time on top of each bar
            for i, cpu_time in enumerate(total_cpu_times):
                ax1.text(i, cpu_time, f'{cpu_time:.4f}', ha='center', va='bottom', fontsize=8)
            ax1.legend()
            ax1.set_xlabel('Action')
            ax1.set_ylabel('Total CPU Time')
            ax1.set_title(f'Total CPU Time Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax1.set_xticks(range(len(action_lists)), action_lists, rotation=45, ha='right')
            
            ax2.bar(action_lists, total_errors, label='Total Error')
            # write the total error on top of each bar
            for i, error in enumerate(total_errors):
                # write it in scientific notation
                ax2.text(i, error, f'{error:.4e}', ha='center', va='bottom', fontsize=8)
            ax2.legend()
            ax2.set_xlabel('Action')
            ax2.set_ylabel('Total Error')
            ax2.set_title(f'Total Error Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            ax2.set_xticks(range(len(action_lists)), action_lists, rotation=45, ha='right')
            
            # plot the reference action distribution in a bar chart (action distrbution is a numpy array of the action distribution)
            action_distribution = {str(action): int(self.action_distribution[i]) for i, action in enumerate(self.action_list)}
            ax3.bar(action_distribution.keys(), action_distribution.values())
            
            # write the action distribution on top of each bar
            for i, action in enumerate(action_distribution):
                ax3.text(i, action_distribution[action], f'{action_distribution[action]}', ha='center', va='bottom', fontsize=8)

            ax3.set_xlabel('Action')
            ax3.set_ylabel('Action Distribution')
            ax3.set_title(f'Action Distribution Comparison - Etol: {self.Etol}, tcfd: {self.tcfd}')
            # slant the x-axis labels
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            
            if save_path:
                save_path_  = save_path.replace('.png', '_comparison_bar.png')
                plt.savefig(save_path_)
            else:
                plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error in compare_with_fixed_action: {e}")
            return None
    
    
if __name__ == "__main__":
    # Example usage
    env = CombustionEnv(
        mech_file='mechanism_files/ch4_53species.yaml',
        initial_state={'T': 1400, 'P': ct.one_atm*40, 'X': 'CH4:1, O2:2, N2:7.52'},
        end_time=2e-4,
        tcfd=1e-6,
        species_to_track=['H2', 'O2', 'H', 'OH', 'O2', 'H2O', 'HO2', 'N2', 'H2O2'],
        Etol=1e-7,
        features_config={
            'temporal_features': True,
            'species_features': True,
            'stiffness_features': False,
            'basic_features': True,
            'include_dt_etol': False,
            'add_net_production_rates': False
        }
    )

    # Run an episode
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your policy
        obs, reward, done, truncated, info = env.step(action)
    
    env.compare_state_with_reference(save_path='comparison_plot.png')
    print(f"Cummulative reward: {info['cummulative_reward']}")
    print(f"Cummulative cpu time: {info['cummulative_cpu_time']}")
    print(f"Temperature error: {info['temperature_error']}")
    print(f"Species error: {info['species_error']}")