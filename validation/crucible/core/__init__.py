# TEMPER Validation Platform - Core Module
"""
Core simulation engine for TEMPER validation.
Built on abstract_crucible_v3 architecture.
"""

from .simulation import Simulation, SimulationParams, SwitchboardConfig
from .agents import AgentType, MaximizerAgent, FrozenAgent, HedonicAgent
from .state import encode_state, N_STATES
from .metrics import compute_effect_size, cohens_d, bootstrap_ci

__all__ = [
    'Simulation', 'SimulationParams', 'SwitchboardConfig',
    'AgentType', 'MaximizerAgent', 'FrozenAgent', 'HedonicAgent',
    'encode_state', 'N_STATES',
    'compute_effect_size', 'cohens_d', 'bootstrap_ci',
]
