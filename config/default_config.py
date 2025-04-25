from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
@dataclass
class Args:
    # Training settings
    exp_name: str = "combustion_ppo"
    seed: int = np.random.randint(0, 1000000)
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "combustion_control"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    normalize_obs: bool = False
    normalize_reward: bool = False
    
    learning_rate: float = 3e-4
    num_steps: int = 1024
    update_epochs: int = 5
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_frequency: int = 5
    min_episodes_before_update: int = 4  # Minimum episodes before allowing updates
    num_envs: int = 1
    minibatch_size: int = 128
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = 0.015
    adapt_temperature: bool = True
    target_entropy: float = -0.5
    
    # Training cycle parameters
    total_timesteps: int = 1000000
    env_reset_frequency: int = 1  # Reset environment every N iterations
    save_frequency: int = 50        # Save checkpoint every N iterations
    eval_frequency: int = 10       # Run evaluation every N iterations
    num_eval_episodes: int = 1    # Number of episodes to run during evaluation
    
    # Problem setup parameters
    mech_file: str = "/home/elo/ubunu_codes/SCI-ML/0D_integratorRL/large_mechanism/large_mechanism/n-dodecane.yaml"
    fuel: str = "nc12h26"
    species_to_track: List[str] = field(default_factory=lambda: ['H2', 'O2', 'H', 'OH', 'H2O', 'HO2', 'H2O2'])
    
    # Temperature, pressure, and phi ranges
    temp_min: float = 900.0
    temp_max: float = 1300.0
    temp_step: float = 50.0
    
    press_min: float = 1.0
    press_max: float = 1.0
    press_step: float = 1
    
    phi_min: float = 0.1
    phi_max: float = 5
    phi_step: float = 1.0
    
    timeout: float = 5
    
    # Time stepping parameters
    end_time: float = 0.2  # s
    min_time_steps_range: Tuple[float, float] = (1e-5, 1e-5)
    max_time_steps_range: Tuple[float, float] = (1e-5, 1e-4)
    
    # Integration parameters 
    reference_rtol: float = 1e-10
    reference_atol: float = 1e-20
#     top_integrators = ['CVODE_BDF',
# 'ARKODE_HEUN_EULER',
# 'ARKODE_BOGACKI',
# 'ARKODE_FEHLBERG_13_7_8',
# 'ARKODE_ARK548L2SA_ERK',  
# 'ARKODE_VERNER_8_5_6',
# 'ARKODE_ARK436L2SA_ERK',
# 'ARKODE_ARK437L2SA_ERK',
# 'ARKODE_ZONNEVELD',
# 'ARKODE_ARK324L2SA_ERK']
    # integrator_list: List[str] = field(default_factory=lambda: ['CPP_RK23', 'ARKODE_HEUN_EULER', 'ARKODE_ZONNEVELD', 
    #                                                            'ARKODE_BOGACKI', 'ARKODE_ARK324L2SA_ERK', 'ARKODE_ARK436L2SA_ERK',
    #                                                            'ARKODE_ARK437L2SA_ERK', 'ARKODE_ARK548L2SA_ERK',
    #                                                            'ARKODE_VERNER_8_5_6', 'ARKODE_FEHLBERG_13_7_8',
    #                                                            'CVODE_BDF', 
    #                                                            'ARKODE_SDIRK_2_1_2', 'ARKODE_KVAERNO_4_2_3','ARKODE_SDIRK_5_3_4',
    #                                                            'ARKODE_BILLINGTON_3_3_2', 'ARKODE_TRBDF2_3_3_2', 'ARKODE_ARK324L2SA_DIRK_4_2_3',
    #                                                            'ARKODE_CASH_5_2_4', 'ARKODE_CASH_5_3_4', 'ARKODE_ARK436L2SA_DIRK_6_3_4',
    #                                                            'ARKODE_ARK437L2SA_DIRK_7_3_4', 'ARKODE_ARK548L2SA_DIRK_8_4_5', 'ARKODE_KVAERNO_7_4_5'])
    
    integrator_list: List[str] = field(default_factory=lambda: ['CPP_RK23', 'ARKODE_HEUN_EULER', 
                                                               'ARKODE_BOGACKI', 'ARKODE_ARK324L2SA_ERK', 
                                                               'CVODE_BDF', 'BDF', 
                                                               'ARKODE_BILLINGTON_3_3_2',
                                                               'ARKODE_ARK437L2SA_DIRK_7_3_4', ])
    # integrator_list: List[str] = field(default_factory=lambda: ['ARKODE_HEUN_EULER', 'CVODE_BDF',
    #                                                             'ARKODE_KVAERNO_4_2_3'])
    tolerance_list: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1e-6, 1e-8)]
    )
    
    # Feature configuration
    features_config: Dict = field(default_factory=lambda: {
        'temporal_features': False,
        'species_features': False,
        'basic_features': True,
        'include_phi': True,
        'window_size': 5,
        'epsilon': 1e-10,
        'include_stage': False,
        'Etol': 1e-5,
        'time_threshold': 1e-2
    })
    
    # Reward weights
    reward_weights: Dict = field(default_factory=lambda: {
        'alpha': 1,    # time weight
        'beta': 1,   # error weight
    })
    
    # norm_adv: bool = True
    # clip_coef: float = 0.2
    # clip_vloss: bool = True
    # ent_coef: float = 0.01
    # vf_coef: float = 0.5
    # max_grad_norm: float = 0.5
    # target_kl: Optional[float] = None
    
    # Runtime variables (set during initialization)
    batch_size: int = 2000
    minibatch_size: int = 200
    num_iterations: int = 5000

    def __post_init__(self):
        import numpy as np
        
        # Create numpy arrays from range parameters
        self.temperature_range = np.linspace(self.temp_min, self.temp_max, 21)  # Initial temperature range
        self.pressure_range = np.array([self.press_min])  # Pressure in atm
        self.phi_range = np.linspace(self.phi_min, self.phi_max, 51)  # Equivalence ratio
        
        
        # # Calculate the number of steps based on end_time and smallest timestep
        # min_timestep = min(self.min_time_steps_range[0], self.max_time_steps_range[0])
        # self.num_steps = int(self.end_time / min_timestep)
        
        # # Update batch sizes
        # self.batch_size = int(self.num_envs * self.num_steps)
        # self.minibatch_size = int(self.batch_size // self.num_minibatches)
        # self.num_iterations = int(self.total_timesteps // self.batch_size)