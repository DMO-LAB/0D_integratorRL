# problem.py
import cantera as ct
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import matplotlib.pyplot as plt
from enum import Enum
from tqdm import tqdm


class CombustionStage(Enum):
    PREIGNITION = 0
    IGNITION = 1
    POSTIGNITION = 2


def setup_problem(temperature_range: np.ndarray, pressure_range: np.ndarray, phi_range: np.ndarray, mech_file: str, fuel: str = 'CH4', oxidizer: str = 'O2:1, N2:3.76',
                  species_to_track: Optional[List[str]] = None,
                  end_time: float = 1e-3,
                  reference_rtol: float = 1e-10,
                  reference_atol: float = 1e-20,
                  state_change_threshold: float = 1,
                  min_time_steps_range: Tuple[int, int] = (1e-6, 1e-5),
                  max_time_steps_range: Tuple[int, int] = (1e-4, 1e-3),
                  randomize: bool = True,
                  verbose: bool = True,
                  fixed_temperature = None,
                  fixed_pressure = None,
                  fixed_phi = None,
                  fixed_dt = None,
                  initial_mixture = None
                  ):
    """ Randomly sample a problem from the given ranges."""
    if randomize:
        temperature = float(np.random.choice(temperature_range))
        pressure = float(np.random.choice(pressure_range))
        phi = float(np.random.choice(phi_range))
    else:
        if fixed_temperature is None:
            temperature = temperature_range[0]
        else:
            temperature = fixed_temperature
        if fixed_pressure is None:
            pressure = pressure_range[0]
        else:
            pressure = fixed_pressure
        if fixed_phi is None:
            phi = phi_range[0]
        else:
            phi = fixed_phi
            
    if temperature <= 1000:
        timestep = np.random.choice(max_time_steps_range) if randomize else fixed_dt
        end_time = 0.05 if randomize else end_time
    else:
        timestep = np.random.choice(min_time_steps_range) if randomize else fixed_dt
        end_time = end_time

    
    return CombustionProblem(mech_file=mech_file,
                            temperature=temperature,
                            pressure=pressure * ct.one_atm,
                            phi=phi,
                            fuel=fuel,
                            oxidizer=oxidizer,
                            end_time=end_time,
                            timestep=timestep,
                            species_to_track=species_to_track,
                            initial_mixture=initial_mixture,
                            reference_rtol=reference_rtol,
                            reference_atol=reference_atol,
                            state_change_threshold=state_change_threshold,
                            verbose=verbose)

class CombustionProblem:
    """Defines and manages a combustion problem including reference solution computation."""

    def __init__(self, 
                 mech_file: str,
                 temperature: float,
                 pressure: float,
                 phi: float,
                 fuel: str = 'CH4',
                 oxidizer: str = 'O2:1, N2:3.76',
                 end_time: float = 1e-3,
                 timestep: float = 1e-6,
                 species_to_track: Optional[List[str]] = None,
                 initial_mixture: str = None,
                 reference_rtol: float = 1e-10,
                 reference_atol: float = 1e-20,
                 state_change_threshold: float = 1,
                 normalizing_temperature: float = 1000,
                 verbose: bool = True
                 ):
        """Initialize combustion problem."""
        try:
            self.mech_file = mech_file
            self.temperature = temperature
            self.pressure = pressure
            self.phi = phi
            self.fuel = fuel
            self.oxidizer = oxidizer
            self.end_time = end_time
            self.timestep = timestep
            self.reference_rtol = reference_rtol
            self.reference_atol = reference_atol
            self.state_change_threshold = state_change_threshold
            self.normalizing_temperature = normalizing_temperature
            self.verbose = verbose
            # Initialize Cantera solution
            self.gas = ct.Solution(self.mech_file)
            
            # Set up mixture
            if initial_mixture is None:
                self.initial_mixture = f"{self.fuel}:1, {self.oxidizer}"
            else:
                self.initial_mixture = initial_mixture
            
            self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
            self.gas.TPX = self.temperature, self.pressure, self.initial_mixture
        
            # Set default species to track if none provided
            if species_to_track is None:
                self.species_to_track = ['H2', 'O2', 'H', 'OH', 'H2O', 'HO2', 'H2O2']
            else:
                self.species_to_track = species_to_track
            
            # Compute number of steps
            self.num_steps = int(self.end_time / self.timestep)
            self.ignition_delay = 0.0
            
            # Reference solution storage
            self.reference_solution = None
            self.current_state = CombustionStage.PREIGNITION
            if self.verbose:
                print(f"Combustion problem initialized with T={self.temperature}, P={self.pressure}, phi={self.phi} and timestep={self.timestep}")
            self._compute_reference_solution()
        except Exception as e:
            print(f"Error setting up problem: {e}")
            raise e
    
    def _state_changed_significantly(self, previous_state, current_state):
        """Check if state change is significant"""
        if isinstance(previous_state, list):
            previous_state = np.array(previous_state)
        if isinstance(current_state, list):
            current_state = np.array(current_state)
            
        state_change = np.linalg.norm(current_state - previous_state)
        return state_change > self.state_change_threshold, state_change
    
    def _compute_reference_solution(self) -> None:
        """Compute reference solution using Cantera."""
        # Set up the reactor
        reactor = ct.IdealGasConstPressureReactor(self.gas)
        sim = ct.ReactorNet([reactor])
        sim.rtol = self.reference_rtol
        sim.atol = self.reference_atol

        # Pre-allocate arrays
        num_steps = self.num_steps
        times = np.zeros(num_steps)
        temperatures = np.zeros(num_steps)
        pressures = np.zeros(num_steps)
        species_profiles = {spec: np.zeros(num_steps) for spec in self.species_to_track}
        
        start_time = time.time()
        t = 0.0
        stage_steps = {stage: 0 for stage in CombustionStage}
        stage_changes = [False]
        for i in tqdm(range(num_steps), desc="Computing reference solution", disable=not self.verbose):
            previous_state = reactor.thermo.state
            sim.advance(t)
            times[i] = t
            temperatures[i] = reactor.T
            pressures[i] = reactor.thermo.P
            for spec in self.species_to_track:
                species_profiles[spec][i] = reactor.thermo[spec].Y
                
            t += self.timestep
            state_changed, state_change = self._state_changed_significantly(previous_state, reactor.thermo.state)
            stage_changes.append(state_changed)
            #print(f"State changed: {state_changed} at step {i} with change {state_change:.6e} and temperature {reactor.T:.4e} K, stage {self.current_state}")
            if stage_changes[-1] != stage_changes[-2]:
                if self.current_state == CombustionStage.PREIGNITION:
                    stage_steps[self.current_state] = i
                    self.current_state = CombustionStage.IGNITION
                    self.ignition_delay = i * self.timestep
                elif self.current_state == CombustionStage.IGNITION:
                    stage_steps[self.current_state] = i
                    self.current_state = CombustionStage.POSTIGNITION 
            if self.current_state == CombustionStage.POSTIGNITION:
                # stop the simulation after 2 * ignition steps
                if i > 4 * stage_steps[CombustionStage.IGNITION]:
                    self.completed_steps = i
                    print(f"Postignition stage completed at step {i}")
                    break # stop the simulation after postignition
        
        self.completed_steps = i
        
        computation_time = time.time() - start_time
        if self.verbose:
            print(f"Reference solution computed in {computation_time:.2f} seconds")
        
        self.reference_solution = {
            'times': times,
            'temperatures': temperatures,
            'pressures': pressures,
            'species_profiles': species_profiles,
            'computation_time': computation_time
        }
    
    def get_reference_solution(self) -> Dict[str, Any]:
        """Get the pre-computed reference solution."""
        if self.reference_solution is None:
            self._compute_reference_solution()
        return self.reference_solution
    
    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial state dictionary."""
        return {
            'T': self.temperature,
            'P': self.pressure,
            'X': self.initial_mixture
        }
    
    def get_problem_params(self) -> Dict[str, Any]:
        """Get problem parameters dictionary."""
        return {
            'end_time': self.end_time,
            'timestep': self.timestep,
            'num_steps': self.num_steps,
            'species_to_track': self.species_to_track,
            'mech_file': self.mech_file,
            'phi': self.phi,
            'fuel': self.fuel,
            'oxidizer': self.oxidizer
        }
    
    def plot_reference_solution(self, save_path: Optional[str] = None) -> None:
        """Plot the reference solution."""
        if self.reference_solution is None:
            print("No reference solution available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        times = self.reference_solution['times'][:self.completed_steps]
        ax1.plot(times, self.reference_solution['temperatures'][:self.completed_steps])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Evolution')
        
        for species, profile in self.reference_solution['species_profiles'].items():
            ax2.plot(times, profile[:self.completed_steps], label=species)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mass Fraction')
        ax2.set_title('Species Evolution')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

