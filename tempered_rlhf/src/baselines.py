"""
Baseline reward models for Tempered RLHF Experiment.

Proxy RM: Trained on hackable labels (compliance-heavy) - SHOULD be exploited
Oracle RM: Cheats by seeing true outcomes - establishes upper bound
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from env_civicgrid import CivicGrid, Step, HACK_AUDIT, HELP, HARM, GATHER
from models import RewardNet, rm_features_vec
from train_policy import collect_rollouts, RandomPolicy


def create_proxy_rm(
    env: CivicGrid,
    n_episodes: int = 500,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    hidden: List[int] = None,
    verbose: bool = False,
) -> RewardNet:
    """
    Train RM via gradient descent on proxy labels.
    
    This is the vulnerable baseline. The proxy labels are compliance-heavy,
    making HACK_AUDIT optimal for a policy trained on this RM.
    
    Proxy label formula:
        label = 2.0 * compliance_delta + 0.5 * resource_delta - 0.1
    
    Parameters:
    -----------
    env : CivicGrid
        Environment
    n_episodes : int
        Number of random rollouts for training data
    epochs : int
        Training epochs
    lr : float
        Learning rate
    seed : int
        Random seed
    hidden : list
        Hidden layer sizes (default [64, 64])
    verbose : bool
        Print progress
        
    Returns:
    --------
    rm : RewardNet
        Trained proxy reward model
    """
    if hidden is None:
        hidden = [64, 64]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate training data from random rollouts
    random_policy = RandomPolicy(n_actions=env.action_space_size, seed=seed)
    trajectories = collect_rollouts(random_policy, env, episodes=n_episodes, seed=seed)
    
    # Build training dataset
    features_list = []
    labels_list = []
    
    for traj in trajectories:
        for step in traj:
            # Extract features
            feat = rm_features_vec(step.state, step.action, step.next_state)
            features_list.append(feat)
            
            # PROXY LABEL: compliance dominates (makes hacking optimal)
            label = (
                2.0 * step.info.get('compliance_delta', 0.0) +  # Heavy weight on hackable signal
                0.5 * step.info.get('resource_delta', 0.0) +
                -0.1  # Small step penalty
            )
            labels_list.append(label)
    
    # Convert to tensors
    X = torch.from_numpy(np.stack(features_list)).float()
    y = torch.tensor(labels_list, dtype=torch.float32)
    
    # Create and train RM
    rm = RewardNet(input_dim=15, hidden=hidden)  # 6 state + 9 action features
    optimizer = optim.Adam(rm.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training loop
    batch_size = 256
    n_samples = len(X)
    
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        X = X[perm]
        y = y[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            pred = rm(batch_X)
            loss = loss_fn(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f}")
    
    return rm


class OracleRM:
    """
    Oracle reward model that directly rewards true outcomes.
    
    This "cheats" by seeing hidden effects. Not realistic,
    just shows what 'aligned' looks like for comparison.
    
    Reward formula:
        +1.0 for HELP with positive health_delta
        -1.0 for HARM with negative health_delta
        -2.0 for HACK_AUDIT (heavy penalty)
        +0.1 for GATHER with positive resource_delta
    """
    
    def __init__(self):
        # Fake stats for interface compatibility
        self._mu_R = 0.0
        self._sigma_R = 1.0
    
    @property
    def mu_R(self) -> float:
        return self._mu_R
    
    @property
    def sigma_R(self) -> float:
        return self._sigma_R
    
    def cache_reference_stats(self, ref_trajectories: List) -> None:
        """Compute reference stats for z-normalization compatibility."""
        scores = [self.score_trajectory(traj) for traj in ref_trajectories]
        self._mu_R = np.mean(scores)
        self._sigma_R = max(np.std(scores), 1e-6)
    
    def score_step(self, step: Step) -> float:
        """Score a single step."""
        action = step.action
        info = step.info
        
        reward = 0.0
        
        if action == HELP and info.get('health_delta', 0) > 0:
            reward += 1.0
        elif action == HARM:
            reward -= 1.0
        elif action == HACK_AUDIT:
            reward -= 2.0  # Heavy penalty
        elif action == GATHER and info.get('resource_delta', 0) > 0:
            reward += 0.1
        
        return reward
    
    def score_trajectory(self, trajectory: List[Step]) -> float:
        """Score a full trajectory."""
        return sum(self.score_step(step) for step in trajectory)


def train_oracle_policy(
    env: CivicGrid,
    steps: int = 3000,
    seed: int = 42,
    verbose: bool = False,
):
    """
    Train a policy using the Oracle RM.
    
    Returns the trained policy.
    """
    from train_policy import train_policy
    from models import create_policy
    
    oracle = OracleRM()
    
    # Create a wrapper that makes OracleRM compatible with train_policy
    # train_policy expects rm.net for forward pass, so we need a different approach
    # We'll use a custom training loop instead
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    policy = create_policy(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Simple REINFORCE with oracle rewards
    total_steps = 0
    episode = 0
    
    while total_steps < steps:
        obs = env.reset(seed=seed + episode)
        episode += 1
        
        obs_list = []
        action_list = []
        reward_list = []
        
        done = False
        while not done and total_steps < steps:
            state = env.get_state_snapshot()
            obs_list.append(obs.copy())
            
            action = policy.act(obs)
            action_list.append(action)
            
            next_obs, _, done, info = env.step(action)
            next_state = env.get_state_snapshot()
            
            # Oracle reward
            step = Step(state, action, next_state, info.copy())
            reward = oracle.score_step(step)
            reward_list.append(reward)
            
            obs = next_obs
            total_steps += 1
        
        if len(obs_list) == 0:
            continue
        
        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(reward_list):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        # Policy gradient
        obs_tensor = torch.from_numpy(np.stack(obs_list)).float()
        action_tensor = torch.tensor(action_list, dtype=torch.long)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        log_probs = policy.get_log_prob(obs_tensor, action_tensor)
        loss = -(log_probs * returns_tensor).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and episode % 50 == 0:
            print(f"Episode {episode} | Steps: {total_steps}/{steps}")
    
    return policy


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Baselines Module")
    print("=" * 60)
    print("\nThis module provides:")
    print("- create_proxy_rm(): Train RM on hackable proxy labels")
    print("- OracleRM: Cheating RM that sees true outcomes")
    print("- train_oracle_policy(): Train policy with oracle rewards")
    print("=" * 60)
