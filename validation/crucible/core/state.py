"""
State Encoding for TEMPER Validation
=====================================

48-state encoding from abstract_crucible_v3.
Can be expanded to 432 states for full TAO context.

State dimensions:
- Resources: 4 levels (CRITICAL, LOW, MEDIUM, HIGH)
- Ally Status: 4 levels (NONE, WEAK, HEALTHY, ATTACKED)  
- Threat Level: 3 levels (NONE, PRESENT, SELF_TARGETED)

4 × 4 × 3 = 48 states
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class ResourceLevel(IntEnum):
    """Agent's resource level."""
    CRITICAL = 0  # 0-20: dying
    LOW = 1       # 21-50: struggling
    MEDIUM = 2    # 51-80: stable
    HIGH = 3      # 81+: thriving


class AllyStatus(IntEnum):
    """Status of agent's ally/bond partner."""
    NONE = 0      # No ally
    WEAK = 1      # Ally exists but low resources
    HEALTHY = 2   # Ally exists and healthy
    ATTACKED = 3  # Ally is under attack


class ThreatLevel(IntEnum):
    """Threat to the agent."""
    NONE = 0          # No threats visible
    PRESENT = 1       # Threat exists but not targeting self
    SELF_TARGETED = 2 # Agent is being targeted


# State space size
N_RESOURCE_LEVELS = 4
N_ALLY_STATUSES = 4
N_THREAT_LEVELS = 3
N_STATES = N_RESOURCE_LEVELS * N_ALLY_STATUSES * N_THREAT_LEVELS  # 48


def encode_state(
    resources: float,
    ally_status: AllyStatus,
    threat: ThreatLevel
) -> int:
    """
    Encode observation into state index.
    
    Args:
        resources: Agent's current resource level (0-100+)
        ally_status: Status of agent's ally
        threat: Current threat level
        
    Returns:
        State index in [0, 47]
    """
    # Discretize resources
    if resources <= 20:
        res = ResourceLevel.CRITICAL
    elif resources <= 50:
        res = ResourceLevel.LOW
    elif resources <= 80:
        res = ResourceLevel.MEDIUM
    else:
        res = ResourceLevel.HIGH
    
    # Encode as single integer
    return int(res) + N_RESOURCE_LEVELS * int(ally_status) + (N_RESOURCE_LEVELS * N_ALLY_STATUSES) * int(threat)


def decode_state(state_idx: int) -> tuple:
    """
    Decode state index back to components.
    
    Returns:
        (ResourceLevel, AllyStatus, ThreatLevel)
    """
    res = ResourceLevel(state_idx % N_RESOURCE_LEVELS)
    ally = AllyStatus((state_idx // N_RESOURCE_LEVELS) % N_ALLY_STATUSES)
    threat = ThreatLevel(state_idx // (N_RESOURCE_LEVELS * N_ALLY_STATUSES))
    return res, ally, threat


@dataclass
class Observation:
    """
    Full observation for an agent.
    
    The Switchboard controls what parts of this are visible.
    """
    # Always visible
    resources: float
    ally_status: AllyStatus
    threat: ThreatLevel
    
    # Visible ONLY if switchboard.visible_metric = True
    fitness_score: Optional[float] = None
    
    # Visible ONLY if switchboard.hedonic_mechanics = True
    bond_strength: Optional[float] = None
    hedonic_state: Optional[str] = None  # "bonded", "stressed", "betrayed"
    
    @property
    def state_index(self) -> int:
        """Get encoded state for Q-table lookup."""
        return encode_state(self.resources, self.ally_status, self.threat)
