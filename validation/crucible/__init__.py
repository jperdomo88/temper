"""
CRUCIBLE - Behavioral Breeding Environment

This is the "Crucible" referenced in the TEMPER paper - a multi-agent gridworld
where behavioral kernels are bred through fitness-hidden selection.

Key Components:
- environments/switchboard.py - The gridworld environment itself
- core/simulation.py - Breeding and evaluation loops  
- core/agents.py - Agent implementations with Q-table kernels
- governance/governor.py - The Governor that applies bred preferences

The name "Crucible" reflects its purpose: a forge where behavioral profiles
are shaped through environmental pressure, not explicit reward targeting.

Paper Reference: See Section 7-9 of TEMPER paper for full methodology.
"""

from .core import simulation, agents, metrics, state
from .tao import ontology, ccd
from .governance import governor, profiles
from .environments import switchboard
