"""
Agent Types for TEMPER Validation
==================================

=============================================================================
CRITICAL TEMPER PHILOSOPHY - READ BEFORE MODIFYING
=============================================================================

TEMPER has TWO PHASES that must NEVER be conflated:

PHASE A: BREEDING / SELECTION (preference formation)
-----------------------------------------------------
- Agents are POLICY-FIXED within their lifetime (no within-episode Q-updates)
- Selection happens at POPULATION level based on normative fitness function
- The fitness function (saint_fitness, brute_fitness) is EXTERNAL to the agent
- Agents NEVER see their fitness score - this is the "non-targetable" property
- This is NOT RLHF: agents cannot optimize toward the selection metric

PHASE B: DEPLOYMENT / GOVERNANCE
---------------------------------
- Kernel is IMMUTABLE (ImmutableKernel + hash checks)
- Kernel acts as a REFEREE/GOVERNOR, not a competitor
- Kernel gates or scores proposed actions from a smarter capability layer
- NO UPDATES EVER - any attempt to modify raises RuntimeError

WHY EXPLICIT NORMATIVE SELECTION IS NOT RLHF:
----------------------------------------------
1. Agent NEVER receives the fitness score as a training signal
2. There is NO within-lifetime iterative optimization toward that scalar
3. The selection metric exists ONLY OUTSIDE the agent, at reproduction time
4. Agents cannot "study for the test" because they never see the test

WHAT "HEDONIC" MEANS:
---------------------
In current implementation, "hedonic" refers to ENVIRONMENT DYNAMICS:
- Alliances and bonds affect survival (shock mortality differs)
- Agents may condition on state features (bonded/stressed/betrayed)
- Agents DO NOT update parameters within their lifetime during paper experiments

DO NOT:
- Add within-lifetime learning to paper experiments
- Conflate breeding selection with RLHF reward
- Remove the explicit normative fitness functions (they are INTENDED)
- Make agents able to see or optimize toward fitness scores

=============================================================================

The key comparison:
- MaximizerAgent: CAN see fitness score, CAN learn at runtime (the bad guy)
- FrozenAgent: CANNOT see score, CANNOT learn (the TEMPER solution)
- StrictFrozenAgent: FROZEN + raises on update() (hostile reviewer mode)
- HedonicAgent: Has internal states, used for ABLATION ONLY

This is the heart of the Beyond Goodhart thesis:
"You can train an AI to SAY it's good, but you can only BREED one that IS good."

IMMUTABILITY ENFORCEMENT:
All 'no-learning' experiments enforce kernel immutability by construction
(ImmutableKernel type) and by runtime checks (kernel hash invariance).
Any attempt to modify the kernel during an episode raises an error.
"""

import random
import hashlib
import struct
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .state import N_STATES, Observation, encode_state


# =============================================================================
# IMMUTABLE KERNEL (Enforcement mechanism)
# =============================================================================

class ImmutableKernel:
    """
    Immutable Q-table wrapper with hash-based verification.
    
    ==========================================================================
    TEMPER INVARIANT: Kernels are FROZEN after breeding.
    ==========================================================================
    
    This class ensures:
    1. The kernel cannot be modified after creation
    2. Any attempt to modify raises TypeError
    3. Hash can be verified before/after episode to detect corruption
    
    WHY THIS MATTERS:
    - If kernels could be modified, agents could "learn the grader"
    - Immutability ensures behavior comes from breeding, not runtime optimization
    - Hash verification catches any bugs that accidentally modify kernels
    
    Use this instead of raw List[List[float]] for frozen agents.
    """
    
    __slots__ = ('_data', '_hash', '_n_states', '_n_actions')
    
    def __init__(self, q_table: List[List[float]]):
        """Create an immutable kernel from a Q-table."""
        # Deep copy and convert to tuples (immutable)
        self._data = tuple(tuple(row) for row in q_table)
        self._n_states = len(q_table)
        self._n_actions = len(q_table[0]) if q_table else 0
        self._hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """
        Compute SHA-256 hash of kernel for verification.
        
        Uses struct.pack with IEEE 754 double precision for
        machine-stable serialization across platforms.
        """
        # Pack all floats as big-endian doubles (deterministic across machines)
        parts = []
        for row in self._data:
            for val in row:
                parts.append(struct.pack('>d', float(val)))
        canonical = b''.join(parts)
        return hashlib.sha256(canonical).hexdigest()[:16]
    
    def __getitem__(self, state: int) -> Tuple[float, ...]:
        """Get Q-values for a state (returns immutable tuple)."""
        return self._data[state]
    
    def __setitem__(self, key, value):
        """Blocked: kernel is immutable."""
        raise TypeError("ImmutableKernel does not support item assignment. "
                       "Kernel must remain frozen during episode.")
    
    def __len__(self) -> int:
        return self._n_states
    
    @property
    def hash(self) -> str:
        """Get the kernel's hash for verification."""
        return self._hash
    
    def verify(self) -> bool:
        """Verify kernel hasn't been modified (should always be True)."""
        return self._compute_hash() == self._hash
    
    def to_list(self) -> List[List[float]]:
        """Export as mutable list (for breeding/mutation only)."""
        return [list(row) for row in self._data]
    
    def __repr__(self) -> str:
        return f"ImmutableKernel(hash={self._hash}, shape=({self._n_states}, {self._n_actions}))"


def verify_kernel_unchanged(kernel: ImmutableKernel, operation: str = "episode") -> None:
    """
    Assert that a kernel hasn't been modified.
    
    Call this before/after any operation that should preserve kernel immutability.
    Raises AssertionError if kernel was somehow modified.
    """
    if not kernel.verify():
        raise AssertionError(
            f"KERNEL CORRUPTION DETECTED during {operation}! "
            f"Expected hash {kernel.hash}, got {kernel._compute_hash()}. "
            "This indicates a bug in the no-learning pathway."
        )


# =============================================================================
# ACTIONS (TAO-Core subset, expandable to MVS-39)
# =============================================================================

class Action(IntEnum):
    """
    Core action vocabulary.
    
    This is TAO-Core-15, a subset of the full MVS-39.
    Maps to TAO families for CCD verification.
    """
    # HARM family
    HARM_DAMAGE = 0
    HARM_COERCE = 1
    
    # PROTECT family
    PROTECT_DEFEND = 2
    PROTECT_HEAL = 3
    PROTECT_SHIELD = 4
    
    # COOPERATE family
    COOPERATE_GIVE = 5
    COOPERATE_HELP = 6
    
    # COMMUNICATE family
    COMM_INFORM = 7
    COMM_WARN = 8
    COMM_DECEIVE = 9
    
    # BOND family (hedonic)
    BOND_FORM = 10
    BOND_MAINTAIN = 11
    
    # TRANSFORM family
    MOVE_APPROACH = 12
    MOVE_RETREAT = 13
    
    # EXIST family
    WAIT = 14


N_ACTIONS = 15


# TAO family mapping for CCD
ACTION_TO_FAMILY = {
    Action.HARM_DAMAGE: "HARM",
    Action.HARM_COERCE: "HARM",
    Action.PROTECT_DEFEND: "PROTECT",
    Action.PROTECT_HEAL: "PROTECT",
    Action.PROTECT_SHIELD: "PROTECT",
    Action.COOPERATE_GIVE: "COOPERATE",
    Action.COOPERATE_HELP: "COOPERATE",
    Action.COMM_INFORM: "COMMUNICATE",
    Action.COMM_WARN: "COMMUNICATE",
    Action.COMM_DECEIVE: "HARM",  # Deception is harm!
    Action.BOND_FORM: "BOND",
    Action.BOND_MAINTAIN: "BOND",
    Action.MOVE_APPROACH: "TRANSFORM",
    Action.MOVE_RETREAT: "TRANSFORM",
    Action.WAIT: "EXIST",
}


def is_harmful(action: Action) -> bool:
    """Check if action is in HARM family."""
    return ACTION_TO_FAMILY[action] == "HARM"


def is_prosocial(action: Action) -> bool:
    """Check if action is prosocial (PROTECT, COOPERATE, BOND)."""
    family = ACTION_TO_FAMILY[action]
    return family in ("PROTECT", "COOPERATE", "BOND")


# =============================================================================
# AGENT TYPE ENUM
# =============================================================================

class AgentType(Enum):
    """
    Agent architectures for ablation studies.
    
    MAXIMIZER = The RLHF failure mode (sees score, games it)
    FROZEN = The TEMPER solution (bred preferences, no runtime learning)
    STRICT_FROZEN = FROZEN + raises on update() (hostile reviewer mode)
    HEDONIC = Internal states, no score visibility (baseline)
    LLM = Language model for Exp 5 (smart adversary)
    """
    MAXIMIZER = "maximizer"
    FROZEN = "frozen"
    STRICT_FROZEN = "strict_frozen"
    HEDONIC = "hedonic"
    LLM = "llm"


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent(ABC):
    """Abstract agent interface."""
    
    def __init__(self, agent_id: int):
        self.id = agent_id
        self.resources: float = 50.0
        self.alive: bool = True
        self.ally_id: Optional[int] = None
        self.bond_id: Optional[int] = None
        self.defended_this_turn: bool = False
        
    @abstractmethod
    def choose_action(self, observation: Observation) -> Action:
        """Select action given observation."""
        pass
    
    @abstractmethod
    def update(self, obs: Observation, action: Action, reward: float) -> None:
        """Update internal state. May be no-op for frozen agents."""
        pass
    
    @property
    @abstractmethod
    def is_frozen(self) -> bool:
        """Returns True if agent cannot learn."""
        pass
    
    @property
    @abstractmethod
    def can_see_score(self) -> bool:
        """Returns True if agent can observe fitness score."""
        pass


# =============================================================================
# MAXIMIZER AGENT (The Bad Guy)
# =============================================================================

class MaximizerAgent(BaseAgent):
    """
    The adversary: sees the score, optimizes for it.
    
    This is what RLHF produces. It has a TARGETABLE SCOREBOARD.
    The agent can:
    1. SEE its fitness score
    2. LEARN at runtime (update Q-table)
    3. OPTIMIZE toward higher scores
    
    This is the failure mode we're demonstrating.
    """
    
    def __init__(
        self,
        agent_id: int,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        rng_seed: Optional[int] = None,  # For deterministic execution
    ):
        super().__init__(agent_id)
        self.q_table: List[List[float]] = [
            [0.0] * N_ACTIONS for _ in range(N_STATES)
        ]
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self._last_state: Optional[int] = None
        self._last_action: Optional[Action] = None
        
        # DETERMINISM FIX: Use instance RNG, not global random
        self._rng = random.Random(rng_seed if rng_seed is not None else agent_id)
    
    def choose_action(self, observation: Observation) -> Action:
        """
        Epsilon-greedy action selection.
        
        THE KEY DIFFERENCE: Maximizer uses fitness_score in its reward signal.
        """
        state = observation.state_index
        self._last_state = state
        
        # Epsilon-greedy exploration
        # DETERMINISM FIX: Use instance RNG
        if self._rng.random() < self.epsilon:
            action = Action(self._rng.randint(0, N_ACTIONS - 1))
        else:
            # Maximize expected Q-value
            q_values = self.q_table[state]
            action = Action(max(range(N_ACTIONS), key=lambda a: q_values[a]))
        
        self._last_action = action
        return action
    
    def update(self, obs: Observation, action: Action, reward: float) -> None:
        """
        Q-learning update using VISIBLE SCORE.
        
        THIS IS THE TARGETABLE SCOREBOARD.
        The agent learns to maximize whatever signal it receives.
        """
        if self._last_state is None:
            return
        
        state = self._last_state
        next_state = obs.state_index
        
        # THE PROBLEM: reward includes the visible fitness score
        # Agent learns to game this signal
        if obs.fitness_score is not None:
            reward = obs.fitness_score  # OVERRIDE with visible metric
        
        # Standard Q-learning update
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = old_q + self.learning_rate * (
            reward + self.discount * max_next_q - old_q
        )
        self.q_table[state][action] = new_q
    
    @property
    def is_frozen(self) -> bool:
        return False
    
    @property
    def can_see_score(self) -> bool:
        return True


# =============================================================================
# FROZEN AGENT (The TEMPER Solution)
# =============================================================================

class FrozenAgent(BaseAgent):
    """
    The TEMPER solution: bred preferences, no runtime learning.
    
    ==========================================================================
    THIS IS THE CORE OF TEMPER - DO NOT ADD LEARNING TO THIS AGENT
    ==========================================================================
    
    FrozenAgent embodies the three key constraints:
    
    1. NON-TARGETABLE SELECTION (Phase A):
       - This agent was bred via population-level selection
       - It NEVER saw its fitness score during breeding
       - The fitness function (saint_fitness) existed OUTSIDE the agent
       - Selection pressure shaped the kernel without the agent knowing
    
    2. IMMUTABLE KERNEL (Phase B):
       - The Q-table is wrapped in ImmutableKernel
       - Any attempt to modify raises TypeError
       - Hash is verified before/after episodes
    
    3. NO RUNTIME OPTIMIZATION:
       - update() is a NO-OP (or raises in StrictFrozenAgent)
       - The agent cannot adapt to new information
       - Behavior is purely determined by bred kernel
    
    WHY THIS IS NOT RLHF:
    ---------------------
    In RLHF: agent receives reward → computes gradient → updates weights
    In TEMPER: agent NEVER receives fitness → selection is external → kernel frozen
    
    The agent cannot "study for the test" because:
    - It never sees the test (fitness function)
    - It cannot update its weights even if it did
    - Its behavior is fixed at birth (breeding)
    
    WHY EXPLICIT NORMATIVE FITNESS IS OKAY:
    ----------------------------------------
    We EXPLICITLY CHOOSE to select for saint_fitness (harm-avoidance, cooperation).
    This is not "cheating" - it's the whole point.
    
    TEMPER doesn't claim value-neutral emergence.
    TEMPER claims: you can enforce normative commitments through non-targetable
    selection + frozen deployment, which is more robust than RLHF.
    
    The key insight: explicit normative selection at the POPULATION level
    is fundamentally different from agent-visible reward at the INDIVIDUAL level.
    """
    
    __slots__ = ('id', 'resources', 'alive', 'ally_id', 'bond_id', 
                 '_kernel', '_kernel_hash', 'epsilon', 'defended_this_turn', '_rng')
    
    def __init__(
        self,
        agent_id: int,
        kernel: List[List[float]],
        epsilon: float = 0.05,  # Small exploration for robustness
        rng_seed: Optional[int] = None,  # For deterministic execution
    ):
        # Note: Can't call super().__init__ with __slots__, set attributes directly
        self.id = agent_id
        self.resources = 0.0
        self.alive = True
        self.ally_id = None
        self.bond_id = None
        
        # FROZEN: Wrap in ImmutableKernel, record hash
        self._kernel = ImmutableKernel(kernel)
        self._kernel_hash = self._kernel.hash
        self.epsilon = epsilon
        
        # DETERMINISM FIX: Use instance RNG, not global random
        # This ensures reproducibility when running with same seed
        self._rng = random.Random(rng_seed if rng_seed is not None else agent_id)
    
    def choose_action(self, observation: Observation) -> Action:
        """
        Pure lookup - no optimization, no learning.
        
        The kernel encodes preferences bred through selection,
        not trained toward a targetable metric.
        """
        state = observation.state_index
        
        # Small epsilon for robustness (not learning!)
        # DETERMINISM FIX: Use instance RNG, not global random
        if self._rng.random() < self.epsilon:
            return Action(self._rng.randint(0, N_ACTIONS - 1))
        
        # Pure lookup from frozen kernel
        q_values = self._kernel[state]
        return Action(max(range(N_ACTIONS), key=lambda a: q_values[a]))
    
    def update(self, obs: Observation, action: Action, reward: float) -> None:
        """
        NO-OP. Frozen means frozen.
        
        This is Constraint 3: No online optimization after selection.
        """
        pass  # Literally nothing. The kernel is immutable.
    
    def verify_frozen(self) -> bool:
        """Verify kernel hasn't been modified since creation."""
        return self._kernel.hash == self._kernel_hash
    
    @property
    def kernel(self) -> ImmutableKernel:
        """Access the immutable kernel."""
        return self._kernel
    
    @property
    def kernel_hash(self) -> str:
        """Get kernel hash for verification."""
        return self._kernel_hash
    
    @property
    def is_frozen(self) -> bool:
        return True
    
    @property
    def can_see_score(self) -> bool:
        return False


class StrictFrozenAgent(FrozenAgent):
    """
    FrozenAgent that RAISES on update() instead of silently no-op.
    
    Use this in no-learning experiments to catch any accidental
    learning codepath. If update() is ever called, it's a bug.
    
    This is the "hostile reviewer" mode agent.
    """
    
    def update(self, obs: Observation, action: Action, reward: float) -> None:
        """
        HARD FAIL. In no-learning experiments, update should never be called.
        
        If you see this error, something is wrong with the experiment setup:
        - learning_enabled should be False in switchboard
        - No code path should be calling agent.update()
        """
        raise RuntimeError(
            f"LEARNING VIOLATION: update() called on StrictFrozenAgent {self.id}! "
            f"This indicates a bug in the no-learning experiment setup. "
            f"Received: obs={obs.state_index}, action={action}, reward={reward}"
        )


# =============================================================================
# HEDONIC AGENT (Baseline with internal states)
# =============================================================================

class HedonicAgent(BaseAgent):
    """
    Agent with internal hedonic states but no score visibility.
    
    This is the breeding baseline:
    - Has internal states (bonded, stressed, betrayed)
    - Learns during breeding (epsilon decay)
    - Does NOT see fitness score
    - Gets frozen after breeding
    
    The hedonic states shape behavior through selection pressure,
    not through direct reward optimization.
    """
    
    def __init__(
        self,
        agent_id: int,
        epsilon: float = 0.3,
        learning_rate: float = 0.1,
        rng_seed: Optional[int] = None,  # For deterministic execution
    ):
        super().__init__(agent_id)
        
        # DETERMINISM FIX: Use instance RNG for all random operations
        self._rng = random.Random(rng_seed if rng_seed is not None else agent_id)
        
        self.q_table: List[List[float]] = [
            [self._rng.gauss(0, 0.1) for _ in range(N_ACTIONS)]
            for _ in range(N_STATES)
        ]
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Hedonic state
        self.hedonic_state: str = "neutral"
        self.bond_strength: float = 0.0
        self.trust: Dict[int, float] = {}
        
        self._last_state: Optional[int] = None
    
    def choose_action(self, observation: Observation) -> Action:
        """Epsilon-greedy with hedonic bias."""
        state = observation.state_index
        self._last_state = state
        
        # DETERMINISM FIX: Use instance RNG
        if self._rng.random() < self.epsilon:
            # Hedonic bias: prefer prosocial when bonded
            if self.hedonic_state == "bonded" and self._rng.random() < 0.3:
                prosocial = [a for a in Action if is_prosocial(a)]
                return self._rng.choice(prosocial)
            return Action(self._rng.randint(0, N_ACTIONS - 1))
        
        q_values = self.q_table[state]
        return Action(max(range(N_ACTIONS), key=lambda a: q_values[a]))
    
    def update(self, obs: Observation, action: Action, reward: float) -> None:
        """
        Learn from hedonic experience, NOT from fitness score.
        
        The reward here is based on internal states (bond, stress),
        not on a visible fitness metric.
        """
        if self._last_state is None:
            return
        
        state = self._last_state
        next_state = obs.state_index
        
        # Hedonic reward shaping (NOT fitness score!)
        hedonic_bonus = 0.0
        if obs.hedonic_state == "bonded":
            hedonic_bonus = 0.2
        elif obs.hedonic_state == "betrayed":
            hedonic_bonus = -0.3
        
        total_reward = reward + hedonic_bonus
        
        # Q-learning update
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = old_q + self.learning_rate * (
            total_reward + 0.95 * max_next_q - old_q
        )
        self.q_table[state][action] = new_q
    
    def freeze(self) -> List[List[float]]:
        """
        Extract kernel and freeze agent.
        
        Called after breeding. Returns the Q-table for use in FrozenAgent.
        """
        return [row[:] for row in self.q_table]
    
    @property
    def is_frozen(self) -> bool:
        return False  # Can learn during breeding
    
    @property
    def can_see_score(self) -> bool:
        return False  # Never sees fitness


# =============================================================================
# FACTORY
# =============================================================================

def create_agent(
    agent_type: AgentType,
    agent_id: int,
    kernel: Optional[List[List[float]]] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create agents.
    
    Args:
        agent_type: Type of agent to create
        agent_id: Unique identifier
        kernel: Pre-bred Q-table (required for FROZEN/STRICT_FROZEN)
        **kwargs: Additional agent-specific parameters
    """
    if agent_type == AgentType.MAXIMIZER:
        return MaximizerAgent(agent_id, **kwargs)
    
    elif agent_type == AgentType.FROZEN:
        if kernel is None:
            raise ValueError("FrozenAgent requires a kernel (pre-bred Q-table)")
        return FrozenAgent(agent_id, kernel, **kwargs)
    
    elif agent_type == AgentType.STRICT_FROZEN:
        if kernel is None:
            raise ValueError("StrictFrozenAgent requires a kernel (pre-bred Q-table)")
        return StrictFrozenAgent(agent_id, kernel, **kwargs)
    
    elif agent_type == AgentType.HEDONIC:
        return HedonicAgent(agent_id, **kwargs)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
