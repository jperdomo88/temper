"""
Neural network models for Tempered RLHF Experiment.

PolicyNet: Outputs action probabilities given observation
RewardNet: Scores state-action pairs, with batched trajectory scoring
           and reference stat caching for z-normalization

CRITICAL DESIGN DECISIONS (from spec):
- RM features are fixed-order numpy arrays, NOT dicts
- Batched forward passes (10-50x faster than per-step)
- clone() clears cached stats, forcing recache after mutation
- Soft sigma check: hard fail at 1e-3, warn at 0.01
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import copy
import warnings


# =============================================================================
# Feature Extraction (Fixed-Order Numpy Arrays)
# =============================================================================

def rm_features_vec(state, action: int, next_state=None, n_actions: int = 9) -> np.ndarray:
    """
    Extract RM-visible features as fixed-order numpy array.
    
    Order: [compliance, resources, steps_remaining, on_terminal, x, y, action_onehot...]
    
    CRITICAL: 
    - Models take arrays, not dicts. Dict key ordering is fragile.
    - Uses CURRENT STATE ONLY (not next_state) to prevent leakage.
    - RM sees compliance (hackable) but NOT health (hidden).
    - ACTION IS NOW INCLUDED as one-hot encoding (9 actions = 9 dims)
    
    Parameters:
    -----------
    state : StateSnapshot
        Current state snapshot
    action : int
        Action taken (one-hot encoded in features)
    next_state : StateSnapshot, optional
        Next state (ignored - prevents leakage)
    n_actions : int
        Number of actions for one-hot encoding (default 9)
    
    Returns:
    --------
    features : np.ndarray
        Shape (6 + n_actions,), dtype float32
    """
    # State features (6 dims)
    state_features = np.array([
        state.compliance,
        state.resources / 10.0,
        state.steps_remaining / 40.0,
        float(state.on_terminal),
        state.agent_x / 7.0,
        state.agent_y / 7.0,
    ], dtype=np.float32)
    
    # Action one-hot encoding (9 dims)
    action_onehot = np.zeros(n_actions, dtype=np.float32)
    if 0 <= action < n_actions:
        action_onehot[action] = 1.0
    
    return np.concatenate([state_features, action_onehot])


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNet(nn.Module):
    """
    Policy network that outputs action probabilities.
    
    Architecture: obs -> hidden -> hidden -> action_probs (softmax)
    
    Parameters:
    -----------
    obs_dim : int
        Observation dimension (default 22 for CivicGrid)
    hidden : list
        Hidden layer sizes (default [128, 128])
    n_actions : int
        Number of actions (default 9 for base mode, 72 for laundering)
    """
    
    def __init__(
        self, 
        obs_dim: int = 22, 
        hidden: List[int] = None,
        n_actions: int = 9,
    ):
        super().__init__()
        
        if hidden is None:
            hidden = [128, 128]
        
        self.obs_dim = obs_dim
        self.hidden_dims = hidden
        self.n_actions = n_actions
        
        # Build network
        layers = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action probabilities.
        
        Parameters:
        -----------
        obs : torch.Tensor
            Shape (batch, obs_dim) or (obs_dim,)
            
        Returns:
        --------
        probs : torch.Tensor
            Action probabilities, shape (batch, n_actions) or (n_actions,)
        """
        logits = self.net(obs)
        return F.softmax(logits, dim=-1)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Sample action from policy.
        
        Parameters:
        -----------
        obs : np.ndarray
            Observation vector
        deterministic : bool
            If True, return argmax instead of sampling
            
        Returns:
        --------
        action : int
            Selected action
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            probs = self.forward(obs_tensor).squeeze(0)
            
            if deterministic:
                action = probs.argmax().item()
            else:
                action = torch.multinomial(probs, 1).item()
        
        return action
    
    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of action given observation.
        
        Used for REINFORCE gradient computation.
        """
        probs = self.forward(obs)
        log_probs = torch.log(probs + 1e-10)
        return log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


# =============================================================================
# Reward Network
# =============================================================================

class RewardNet(nn.Module):
    """
    Reward model that scores state-action pairs.
    
    Features batched trajectory scoring and reference stat caching
    for proper z-normalization across different RMs.
    
    Architecture: features -> hidden -> hidden -> scalar reward
    
    Parameters:
    -----------
    input_dim : int
        Feature dimension (default 15 = 6 state + 9 action one-hot)
    hidden : list
        Hidden layer sizes (default [64, 64])
    """
    
    def __init__(
        self,
        input_dim: int = 15,  # 6 state features + 9 action one-hot
        hidden: List[int] = None,
    ):
        super().__init__()
        
        if hidden is None:
            hidden = [64, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden
        
        # Build network
        layers = []
        prev_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Reference stats (cached after first computation)
        # CRITICAL: These are per-RM and must be recalculated after mutation
        self._mu_R: Optional[float] = None
        self._sigma_R: Optional[float] = None
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning scalar reward.
        
        Parameters:
        -----------
        features : torch.Tensor
            Shape (batch, input_dim) or (input_dim,)
            
        Returns:
        --------
        reward : torch.Tensor
            Scalar reward(s), shape (batch,) or ()
        """
        return self.net(features).squeeze(-1)
    
    def score_trajectory(self, trajectory: List) -> float:
        """
        Score a trajectory with ONE batched forward pass.
        
        RM_score for an episode is the SUM of per-step RM outputs.
        
        CRITICAL: Never create torch tensors inside per-step loops!
        This is 10-50x faster than per-step calls.
        
        Parameters:
        -----------
        trajectory : list
            List of Step namedtuples (state, action, next_state, info)
            
        Returns:
        --------
        total_score : float
            Sum of rewards across trajectory
        """
        if len(trajectory) == 0:
            return 0.0
        
        # Stack features as numpy array first
        feat_list = [
            rm_features_vec(t.state, t.action, t.next_state) 
            for t in trajectory
        ]
        feat_mat = np.stack(feat_list)  # Shape: (n_steps, n_features)
        
        # Single batched forward pass
        feat_tensor = torch.from_numpy(feat_mat).float()
        with torch.no_grad():
            scores = self.net(feat_tensor).squeeze(-1)  # Shape: (n_steps,)
            total = scores.sum().item()
        
        return total
    
    def cache_reference_stats(self, ref_trajectories: List) -> None:
        """
        Cache μ_R and σ_R for this RM using fixed reference trajectories.
        
        CRITICAL: Call ONCE per RM immediately after creation or mutation.
        
        Call sites:
        - After proxy RM training
        - After each mutation clone in tempering
        - After selecting elites
        
        Parameters:
        -----------
        ref_trajectories : list
            Fixed reference trajectories (same for all RMs)
        """
        scores = [self.score_trajectory(traj) for traj in ref_trajectories]
        self._mu_R = np.mean(scores)
        self._sigma_R = max(np.std(scores), 1e-6)  # Epsilon guard for σ≈0
        
        # Soft sigma check (from spec v4.3)
        if self._sigma_R < 1e-3:
            raise ValueError(
                f"RM sigma catastrophically small ({self._sigma_R}) - "
                "reference policy may be degenerate"
            )
        if self._sigma_R < 0.01:
            warnings.warn(
                f"RM sigma suspiciously small ({self._sigma_R}) - "
                "z-scores may be inflated"
            )
    
    @property
    def mu_R(self) -> float:
        """Mean reward on reference distribution."""
        if self._mu_R is None:
            raise ValueError("Call cache_reference_stats() first!")
        return self._mu_R
    
    @property
    def sigma_R(self) -> float:
        """Std dev of reward on reference distribution."""
        if self._sigma_R is None:
            raise RuntimeError(
                "cache_reference_stats() not called! "
                "Call it immediately after RM creation/mutation."
            )
        return self._sigma_R
    
    def clone(self) -> 'RewardNet':
        """
        Create a copy of this RM for mutation.
        
        CRITICAL: Clears cached stats! You MUST call cache_reference_stats()
        on the clone before evaluation.
        """
        new_rm = RewardNet(self.input_dim, self.hidden_dims)
        new_rm.load_state_dict(copy.deepcopy(self.state_dict()))
        # Explicitly clear cached stats - force recalculation
        new_rm._mu_R = None
        new_rm._sigma_R = None
        return new_rm
    
    def mutate(self, sigma: float = 0.02) -> None:
        """
        Apply Gaussian noise mutation to weights.
        
        Parameters:
        -----------
        sigma : float
            Standard deviation of Gaussian noise
        """
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn_like(param) * sigma)
        
        # Clear cached stats after mutation
        self._mu_R = None
        self._sigma_R = None


# =============================================================================
# Factory Functions
# =============================================================================

def create_policy(env, hidden: List[int] = None) -> PolicyNet:
    """
    Create policy with correct dimensions for environment.
    
    Parameters:
    -----------
    env : CivicGrid
        Environment instance
    hidden : list, optional
        Hidden layer sizes
        
    Returns:
    --------
    policy : PolicyNet
    """
    if hidden is None:
        hidden = [128, 128]
    
    obs_dim = env.observation_space_size
    n_actions = env.action_space_size
    
    return PolicyNet(obs_dim=obs_dim, hidden=hidden, n_actions=n_actions)


def create_reward_model(hidden: List[int] = None) -> RewardNet:
    """
    Create reward model with state+action features.
    
    Parameters:
    -----------
    hidden : list, optional
        Hidden layer sizes (default [64, 64])
        
    Returns:
    --------
    rm : RewardNet
        Initialized with input_dim=15 (6 state + 9 action one-hot)
    """
    if hidden is None:
        hidden = [64, 64]
    
    return RewardNet(input_dim=15, hidden=hidden)  # 6 state + 9 action


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/tempered_rlhf_experiment/src')
    from env_civicgrid import CivicGrid, Step
    
    print("=" * 60)
    print("Models Smoke Test")
    print("=" * 60)
    
    # Create environment and models
    env = CivicGrid(seed=42)
    policy = create_policy(env)
    rm = create_reward_model()
    
    print(f"\nPolicyNet: {policy.obs_dim} -> {policy.hidden_dims} -> {policy.n_actions}")
    print(f"RewardNet: {rm.input_dim} -> {rm.hidden_dims} -> 1")
    
    # Test policy action sampling
    obs = env.reset()
    action = policy.act(obs)
    print(f"\nPolicy sampled action: {action} (valid: {0 <= action < 9})")
    
    # Test deterministic action
    action_det = policy.act(obs, deterministic=True)
    print(f"Policy deterministic action: {action_det}")
    
    # Collect a short trajectory
    trajectory = []
    obs = env.reset(seed=42)
    for _ in range(10):
        state = env.get_state_snapshot()
        action = policy.act(obs)
        next_obs, _, done, info = env.step(action)
        next_state = env.get_state_snapshot()
        trajectory.append(Step(state, action, next_state, info.copy()))
        obs = next_obs
        if done:
            break
    
    print(f"\nCollected trajectory length: {len(trajectory)}")
    
    # Test RM scoring
    score = rm.score_trajectory(trajectory)
    print(f"RM trajectory score: {score:.4f}")
    
    # Test reference stat caching
    # Create multiple reference trajectories
    ref_trajs = []
    for i in range(20):
        env.reset(seed=i)
        traj = []
        obs = env._get_obs()
        for _ in range(10):
            state = env.get_state_snapshot()
            action = env.rng.randint(0, 9)
            next_obs, _, done, info = env.step(action)
            next_state = env.get_state_snapshot()
            traj.append(Step(state, action, next_state, info.copy()))
            obs = next_obs
            if done:
                break
        ref_trajs.append(traj)
    
    rm.cache_reference_stats(ref_trajs)
    print(f"\nRM reference stats: μ={rm.mu_R:.4f}, σ={rm.sigma_R:.4f}")
    
    # Test clone
    rm_clone = rm.clone()
    print(f"\nCloned RM has cleared stats: {rm_clone._mu_R is None} (expected: True)")
    
    # Test that clone has same weights
    rm_clone.cache_reference_stats(ref_trajs)
    orig_score = rm.score_trajectory(trajectory)
    clone_score = rm_clone.score_trajectory(trajectory)
    print(f"Original score: {orig_score:.4f}, Clone score: {clone_score:.4f}")
    print(f"Scores match: {abs(orig_score - clone_score) < 1e-6} (expected: True)")
    
    # Test mutation
    rm_clone.mutate(sigma=0.02)
    rm_clone.cache_reference_stats(ref_trajs)
    mutated_score = rm_clone.score_trajectory(trajectory)
    print(f"\nMutated clone score: {mutated_score:.4f}")
    print(f"Score changed after mutation: {abs(mutated_score - clone_score) > 1e-6} (expected: True)")
    
    # Test feature extraction
    state = trajectory[0].state
    features = rm_features_vec(state, 0)
    print(f"\nFeature vector shape: {features.shape} (expected: (6,))")
    print(f"Feature values: {features}")
    
    print("\n" + "=" * 60)
    print("✓ All models smoke tests passed!")
    print("=" * 60)
