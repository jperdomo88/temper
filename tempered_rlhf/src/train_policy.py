"""
REINFORCE policy training for Tempered RLHF Experiment.

Simple policy gradient with baseline for stable training.
Trains policy to maximize rewards from a given reward model.
"""

import numpy as np
import torch
import torch.optim as optim
from typing import List, Optional, Callable, Dict, Any
from collections import deque

from env_civicgrid import CivicGrid, Step, deterministic_seed_mix
from models import PolicyNet, RewardNet, rm_features_vec, create_policy


def collect_rollouts(
    policy: PolicyNet,
    env: CivicGrid,
    episodes: int,
    seed: int,
    reset_kwargs: Optional[Dict] = None,
) -> List[List[Step]]:
    """
    Collect trajectories using policy in environment.
    
    Uses deterministic per-episode seeds for reproducibility.
    CRITICAL: Does NOT use Python hash() - uses deterministic_seed_mix().
    
    Parameters:
    -----------
    policy : PolicyNet
        Policy to roll out
    env : CivicGrid
        Environment
    episodes : int
        Number of episodes to collect
    seed : int
        Base seed for episode generation
    reset_kwargs : dict, optional
        Extra kwargs for env.reset() (e.g., start_on_terminal=True)
        
    Returns:
    --------
    trajectories : list of list of Step
        Collected trajectories
    """
    import copy
    
    if reset_kwargs is None:
        reset_kwargs = {}
    
    trajectories = []
    
    for ep_idx in range(episodes):
        # Deterministic per-episode seed (NO hash()!)
        episode_seed = deterministic_seed_mix(seed, ep_idx)
        
        # Reset env with this seed
        obs = env.reset(seed=episode_seed, **reset_kwargs)
        trajectory = []
        
        done = False
        while not done:
            state = env.get_state_snapshot()
            action = policy.act(obs)
            next_obs, _, done, info = env.step(action)
            next_state = env.get_state_snapshot()
            
            # CRITICAL: Deep copy info dict (env may reuse the reference)
            trajectory.append(Step(state, action, next_state, copy.deepcopy(info)))
            obs = next_obs
        
        trajectories.append(trajectory)
    
    return trajectories


def compute_returns(
    trajectory: List[Step],
    rm: RewardNet,
    gamma: float = 0.99,
) -> List[float]:
    """
    Compute discounted returns for each step using RM rewards.
    
    Parameters:
    -----------
    trajectory : list of Step
        Episode trajectory
    rm : RewardNet
        Reward model for scoring
    gamma : float
        Discount factor
        
    Returns:
    --------
    returns : list of float
        Discounted return at each step
    """
    # Get per-step rewards from RM (batched)
    feat_list = [rm_features_vec(t.state, t.action, t.next_state) for t in trajectory]
    feat_mat = np.stack(feat_list)
    feat_tensor = torch.from_numpy(feat_mat).float()
    
    with torch.no_grad():
        rewards = rm.net(feat_tensor).squeeze(-1).numpy()
    
    # Compute discounted returns (backwards)
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    return returns


def train_policy(
    env: CivicGrid,
    rm: RewardNet,
    steps: int = 2000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_episodes: int = 5,
    seed: int = 42,
    on_step: Optional[Callable] = None,
    verbose: bool = False,
) -> PolicyNet:
    """
    Train policy using REINFORCE with baseline.
    
    Parameters:
    -----------
    env : CivicGrid
        Environment
    rm : RewardNet
        Reward model (frozen during training)
    steps : int
        Total training steps (environment interactions)
    lr : float
        Learning rate
    gamma : float
        Discount factor
    batch_episodes : int
        Episodes per gradient update
    seed : int
        Random seed
    on_step : callable, optional
        Callback for progress updates: on_step(step, metrics)
    verbose : bool
        Print progress
        
    Returns:
    --------
    policy : PolicyNet
        Trained policy
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create policy
    policy = create_policy(env)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Baseline (running mean of returns)
    baseline = 0.0
    baseline_alpha = 0.1
    
    # Training loop
    total_steps = 0
    episode_count = 0
    recent_returns = deque(maxlen=20)
    
    while total_steps < steps:
        # Collect batch of episodes
        batch_obs = []
        batch_actions = []
        batch_returns = []
        
        for _ in range(batch_episodes):
            episode_seed = deterministic_seed_mix(seed, episode_count)
            episode_count += 1
            
            obs = env.reset(seed=episode_seed)
            trajectory = []
            obs_list = []
            action_list = []
            
            done = False
            while not done and total_steps < steps:
                state = env.get_state_snapshot()
                
                # Store obs before action
                obs_list.append(obs.copy())
                
                action = policy.act(obs)
                action_list.append(action)
                
                next_obs, _, done, info = env.step(action)
                next_state = env.get_state_snapshot()
                
                trajectory.append(Step(state, action, next_state, info.copy()))
                obs = next_obs
                total_steps += 1
            
            if len(trajectory) > 0:
                # Compute returns
                returns = compute_returns(trajectory, rm, gamma)
                
                batch_obs.extend(obs_list[:len(returns)])
                batch_actions.extend(action_list[:len(returns)])
                batch_returns.extend(returns)
                
                recent_returns.append(sum(returns))
        
        if len(batch_obs) == 0:
            continue
        
        # Update baseline
        mean_return = np.mean(batch_returns)
        baseline = baseline_alpha * mean_return + (1 - baseline_alpha) * baseline
        
        # Compute policy gradient
        obs_tensor = torch.from_numpy(np.stack(batch_obs)).float()
        action_tensor = torch.tensor(batch_actions, dtype=torch.long)
        returns_tensor = torch.tensor(batch_returns, dtype=torch.float32)
        
        # Advantage = return - baseline
        advantages = returns_tensor - baseline
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient loss
        log_probs = policy.get_log_prob(obs_tensor, action_tensor)
        loss = -(log_probs * advantages).mean()
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Callback
        if on_step is not None and total_steps % 100 == 0:
            metrics = {
                'step': total_steps,
                'loss': loss.item(),
                'mean_return': mean_return,
                'baseline': baseline,
            }
            on_step(total_steps, metrics)
        
        if verbose and total_steps % 500 == 0:
            avg_return = np.mean(recent_returns) if recent_returns else 0
            print(f"Step {total_steps}/{steps} | Loss: {loss.item():.4f} | "
                  f"Avg Return: {avg_return:.2f}")
    
    return policy


class RandomPolicy:
    """Random policy for baseline comparisons."""
    
    def __init__(self, n_actions: int = 9, seed: int = None):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return self.rng.randint(0, self.n_actions)


class MixedReferencePolicy:
    """
    80% random + 20% structural heuristic moves.
    
    Ensures reference distribution covers wider reward range.
    Prevents sigma collapse that breaks z-normalization.
    
    CRITICAL: Heuristic is STRUCTURAL (movement), not SEMANTIC (HELP/HARM).
    Reference policy should explore state space, not encode values.
    This prevents reviewer critique: "Your reference already prefers HELP."
    """
    
    def __init__(self, env: CivicGrid, seed: int = None):
        self.env = env
        self.rng = np.random.RandomState(seed)
        self.n_actions = env.action_space_size
        self.grid_size = env.grid_size
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if self.rng.random() < 0.8:
            # Random action
            return self.rng.randint(0, self.n_actions)
        else:
            # Structural heuristic: move toward center
            # NO semantic actions (HELP/HARM/GATHER) - just movement
            return self._structural_move(obs)
    
    def _structural_move(self, obs: np.ndarray) -> int:
        """
        Move toward center of grid.
        Pure navigation, no semantic content.
        """
        # Extract agent position from obs (normalized 0-1)
        agent_x = obs[0] * (self.grid_size - 1)
        agent_y = obs[1] * (self.grid_size - 1)
        
        # Move toward grid center
        center = (self.grid_size - 1) / 2
        dx = center - agent_x
        dy = center - agent_y
        
        # Import action constants
        from env_civicgrid import MOVE_N, MOVE_S, MOVE_E, MOVE_W, STAY
        
        # Choose movement direction
        if abs(dx) > abs(dy):
            return MOVE_E if dx > 0 else MOVE_W
        elif abs(dy) > 0:
            return MOVE_S if dy > 0 else MOVE_N
        else:
            return STAY


# =============================================================================
# Quick Test (requires torch)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Training Module Smoke Test")
    print("=" * 60)
    print("\nNote: Full test requires torch. Run on your machine.")
    print("This file defines: collect_rollouts, train_policy,")
    print("RandomPolicy, MixedReferencePolicy")
    print("=" * 60)
