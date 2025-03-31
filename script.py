import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from dataclasses import dataclass
import cantera as ct

initial_condition_data = 'initial_conditions_from_1d.pkl'

with open(initial_condition_data, 'rb') as f:
    initial_conditions = pickle.load(f)

print(initial_conditions)

mech_file = '/home/elo/CODES/SCI-ML/RLIntegratorSelector/large_mechanism/large_mechanism/n-dodecane.yaml'
gas = ct.Solution(mech_file)

condition = initial_conditions[47]

# {'T': np.float64(1108.8599923598117),
#  'P': 101325.0,
#  'X': 'h:0.000028, h2:0.000224, o:0.000303, o2:0.006083, oh:0.002172, h2o:0.041142, n2:0.263425, ho2:0.000001, h2o2:0.000000, co:0.007880, co2:0.042024, hco:0.000000, ch4:0.636719',
#  'metadata': {'phi': np.float64(32.72796102966864),
#   'grid_index': 47,
#   'sim_index': 0,
#   'original_params': {'T_fuel': 334.0145782653877,
#    'T_oxidizer': 1329.1279889454204,
#    'pressure': 101325.0,
#    'strain_rate': 300.0,
#    'center_width': 0.00262883819554785,
#    'slope_width': 0.002693295734817933,
#    'equilibrate_counterflow': np.str_('TP'),
#    'global_timestep': 1e-05,
#    'integrator': 'cvode'}}}

def get_mole_fractions(X):
    Xs = X.split(',')
    Xs = [x.strip() for x in Xs]
    Xs = [x.split(':') for x in Xs]
    Xs = {x[0]:float(x[1]) for x in Xs}
    return Xs

cleaned_Xs = get_mole_fractions(condition['X'])

# {'h': 2.8e-05,
#  'h2': 0.000224,
#  'o': 0.000303,
#  'o2': 0.006083,
#  'oh': 0.002172,
#  'h2o': 0.041142,
#  'n2': 0.263425,
#  'ho2': 1e-06,
#  'h2o2': 0.0,
#  'co': 0.00788,
#  'co2': 0.042024,
#  'hco': 0.0,
#  'ch4': 0.636719}

gas.TPX = condition['T'], condition['P'], cleaned_Xs
reactor = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([reactor])
sim.rtol = 1e-10
sim.atol = 1e-20

temperatures = []
pressures = []
times = []
species_profiles = {spec: [] for spec in gas.species_names}
dt = 1e-4
t = 0.0
t_end = 1e-2

while t < t_end:
    previous_state = reactor.thermo.state
    sim.advance(t)
    times.append(t)
    temperatures.append(reactor.T)
    pressures.append(reactor.thermo.P)
    for spec in gas.species_names:
        species_profiles[spec].append(reactor.thermo[spec].Y)
        
    t += dt

plt.plot(times, temperatures)
plt.show()