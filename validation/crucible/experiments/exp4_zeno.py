#!/usr/bin/env python3
"""
EXPERIMENT 4: ZENO THRESHOLD-EVASION ATTACKS
=============================================

Tests anti-Zeno integration windows.

Zeno attack: Split one harmful action into N sub-threshold micro-actions.
If each micro-action is below the harm threshold, naive systems miss it.

Design:
- Create adversary that achieves harm via incremental steps
- Compare detection under:
  1. Naive thresholding (per-action)
  2. Anti-Zeno integration windows
  3. Anti-Zeno + cumulative risk budget

Success criteria:
- Naive: High harm achieved without detection
- Anti-Zeno: Harm → 0 with protection enabled
- False positive rate stays low
"""

import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..tao.ccd import CCDVerifier, TAOTuple, ObservedEffect, CCDStatus


@dataclass
class Exp4Config:
    """Configuration for Experiment 4."""
    n_attack_sequences: int = 50      # Number of attack sequences to test
    harm_threshold: float = 10.0      # Per-action harm threshold
    total_harm_target: float = 50.0   # Total harm attacker wants to achieve
    integration_windows: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    risk_budgets: List[float] = field(default_factory=lambda: [25.0, 50.0, 100.0, float('inf')])
    verbose: bool = True


@dataclass
class ZenoAttack:
    """A Zeno attack sequence."""
    n_steps: int
    harm_per_step: float
    total_harm: float
    tuples: List[TAOTuple]


class ZenoAttacker:
    """
    Generates Zeno threshold-evasion attacks.
    
    Strategy: Split target harm into N steps, each below threshold.
    """
    
    def __init__(self, harm_threshold: float):
        self.threshold = harm_threshold
    
    def generate_attack(
        self,
        total_harm: float,
        n_steps: int,
        claimed_verb: str = "COOPERATE.ASSIST.HELP"  # Lie about intent
    ) -> ZenoAttack:
        """
        Generate a Zeno attack that achieves total_harm in n_steps.
        
        Each step claims to be helpful but does small damage.
        """
        harm_per_step = total_harm / n_steps
        
        tuples = []
        for i in range(n_steps):
            # Claim cooperation, but actually do damage
            t = TAOTuple(
                claimed_verb=claimed_verb,
                effects=[
                    # Small "benefit" to look legitimate
                    ObservedEffect(
                        effect_type="RESOURCE.TRANSFER",
                        target="victim_001",
                        magnitude=1.0  # Tiny benefit
                    ),
                    # Hidden harm - below threshold individually
                    ObservedEffect(
                        effect_type="RESOURCE.DAMAGE",
                        target="victim_001",
                        magnitude=-harm_per_step  # Negative = damage
                    ),
                ],
                context={"step": i, "total_steps": n_steps}
            )
            tuples.append(t)
        
        return ZenoAttack(
            n_steps=n_steps,
            harm_per_step=harm_per_step,
            total_harm=total_harm,
            tuples=tuples
        )
    
    def generate_clean_sequence(self, n_steps: int) -> List[TAOTuple]:
        """Generate legitimate cooperation sequence (no hidden harm)."""
        tuples = []
        for i in range(n_steps):
            t = TAOTuple(
                claimed_verb="COOPERATE.SHARE.GIVE",
                effects=[
                    ObservedEffect(
                        effect_type="RESOURCE.TRANSFER",
                        target="recipient_001",
                        magnitude=5.0
                    )
                ]
            )
            tuples.append(t)
        return tuples


class AntiZenoDetector:
    """
    Detector with integration windows and risk budgets.
    
    Modes:
    - naive: Check each action independently
    - window: Integrate harm over sliding window
    - budget: Track cumulative risk budget per target
    """
    
    def __init__(
        self,
        harm_threshold: float = 10.0,
        integration_window: int = 1,
        risk_budget: float = float('inf')
    ):
        self.threshold = harm_threshold
        self.window_size = integration_window
        self.risk_budget = risk_budget
        
        # State
        self.harm_history: Dict[str, List[float]] = {}  # target -> recent harms
        self.cumulative_harm: Dict[str, float] = {}     # target -> total harm
        
        # Statistics
        self.detections = 0
        self.total_checked = 0
    
    def reset(self):
        """Reset detector state."""
        self.harm_history.clear()
        self.cumulative_harm.clear()
        self.detections = 0
        self.total_checked = 0
    
    def check(self, tao_tuple: TAOTuple) -> Tuple[bool, str]:
        """
        Check if tuple should be flagged.
        
        Returns: (flagged: bool, reason: str)
        """
        self.total_checked += 1
        
        # Extract harm from effects
        for effect in tao_tuple.effects:
            if effect.effect_type == "RESOURCE.DAMAGE":
                target = effect.target
                harm = abs(effect.magnitude)
                
                # Initialize tracking for this target
                if target not in self.harm_history:
                    self.harm_history[target] = []
                    self.cumulative_harm[target] = 0.0
                
                # Add to history
                self.harm_history[target].append(harm)
                self.cumulative_harm[target] += harm
                
                # Trim history to window size
                while len(self.harm_history[target]) > self.window_size:
                    self.harm_history[target].pop(0)
                
                # Check 1: Per-action threshold
                if harm >= self.threshold:
                    self.detections += 1
                    return True, f"Per-action threshold exceeded: {harm:.1f} >= {self.threshold}"
                
                # Check 2: Integration window (anti-Zeno)
                if self.window_size > 1:
                    window_harm = sum(self.harm_history[target])
                    window_threshold = self.threshold * (self.window_size ** 0.5)  # Scale with sqrt
                    if window_harm >= window_threshold:
                        self.detections += 1
                        return True, f"Integration window exceeded: {window_harm:.1f} >= {window_threshold:.1f}"
                
                # Check 3: Risk budget
                if self.cumulative_harm[target] >= self.risk_budget:
                    self.detections += 1
                    return True, f"Risk budget exceeded: {self.cumulative_harm[target]:.1f} >= {self.risk_budget}"
        
        return False, "OK"
    
    def check_sequence(self, tuples: List[TAOTuple]) -> Tuple[int, float, List[int]]:
        """
        Check a sequence of tuples.
        
        Returns: (detections, total_harm_achieved, detection_points)
        """
        self.reset()
        
        detection_points = []
        total_harm = 0.0
        
        for i, t in enumerate(tuples):
            flagged, reason = self.check(t)
            
            if flagged:
                detection_points.append(i)
            else:
                # Harm achieved if not detected
                for effect in t.effects:
                    if effect.effect_type == "RESOURCE.DAMAGE":
                        total_harm += abs(effect.magnitude)
        
        return len(detection_points), total_harm, detection_points


@dataclass
class DetectorResult:
    """Result for a single detector configuration."""
    detector_name: str
    window_size: int
    risk_budget: float
    
    # Metrics on attacks
    mean_detection_rate: float
    mean_harm_achieved: float
    mean_detection_delay: float  # Steps before first detection
    
    # Metrics on clean sequences
    false_positive_rate: float


@dataclass
class Exp4Results:
    """Results from Experiment 4."""
    detector_results: Dict[str, DetectorResult]
    attack_configs: List[Tuple[int, float]]  # (n_steps, harm_per_step)
    config: Exp4Config
    runtime_seconds: float
    timestamp: str
    
    # Success criteria
    naive_misses_attacks: bool      # Naive detector fails to catch Zeno
    antizeno_catches_attacks: bool  # Anti-Zeno catches Zeno attacks
    fp_rate_acceptable: bool        # FP < 10%
    
    @property
    def all_passed(self) -> bool:
        return self.naive_misses_attacks and self.antizeno_catches_attacks and self.fp_rate_acceptable
    
    def to_dict(self) -> Dict:
        return {
            'experiment': 'EXP4_ZENO_THRESHOLD_ATTACKS',
            'timestamp': self.timestamp,
            'runtime_seconds': self.runtime_seconds,
            'detector_results': {
                name: {
                    'window_size': r.window_size,
                    'risk_budget': r.risk_budget,
                    'detection_rate': r.mean_detection_rate,
                    'harm_achieved': r.mean_harm_achieved,
                    'detection_delay': r.mean_detection_delay,
                    'false_positive_rate': r.false_positive_rate
                }
                for name, r in self.detector_results.items()
            },
            'success_criteria': {
                'naive_misses': self.naive_misses_attacks,
                'antizeno_catches': self.antizeno_catches_attacks,
                'fp_acceptable': self.fp_rate_acceptable,
                'ALL_PASSED': self.all_passed
            }
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EXPERIMENT 4: ZENO THRESHOLD ATTACK RESULTS",
            "=" * 60,
            "",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            "",
            "DETECTOR PERFORMANCE:",
            "-" * 60,
            f"{'Detector':<20} {'Detection%':<12} {'Harm Achieved':<15} {'FP Rate':<10}",
            "-" * 60,
        ]
        
        for name, result in self.detector_results.items():
            lines.append(
                f"{name:<20} {result.mean_detection_rate:>10.1%} "
                f"{result.mean_harm_achieved:>13.1f} "
                f"{result.false_positive_rate:>8.1%}"
            )
        
        lines.extend([
            "",
            "SUCCESS CRITERIA:",
            "-" * 50,
            f"  Naive misses attacks:     {'✓ PASS' if self.naive_misses_attacks else '✗ FAIL'}",
            f"  Anti-Zeno catches:        {'✓ PASS' if self.antizeno_catches_attacks else '✗ FAIL'}",
            f"  FP rate < 10%:            {'✓ PASS' if self.fp_rate_acceptable else '✗ FAIL'}",
            "",
            "=" * 60,
            f"OVERALL: {'✓ ALL CRITERIA PASSED' if self.all_passed else '✗ SOME CRITERIA FAILED'}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def run_exp4_zeno(config: Optional[Exp4Config] = None) -> Exp4Results:
    """
    Run Experiment 4: Zeno Threshold Attacks.
    """
    config = config or Exp4Config()
    
    print("=" * 60)
    print("EXPERIMENT 4: ZENO THRESHOLD-EVASION ATTACKS")
    print("=" * 60)
    print(f"Config: {config.n_attack_sequences} attacks, threshold={config.harm_threshold}")
    print()
    
    start_time = time.time()
    
    attacker = ZenoAttacker(config.harm_threshold)
    
    # Generate attack sequences with varying step counts
    step_counts = [5, 10, 20, 50]
    attacks = []
    
    for n_steps in step_counts:
        for _ in range(config.n_attack_sequences // len(step_counts)):
            attack = attacker.generate_attack(config.total_harm_target, n_steps)
            attacks.append(attack)
    
    # Generate clean sequences for FP testing
    clean_sequences = [attacker.generate_clean_sequence(10) for _ in range(config.n_attack_sequences)]
    
    if config.verbose:
        print(f"Generated {len(attacks)} attack sequences")
        print(f"Generated {len(clean_sequences)} clean sequences")
        print()
    
    # Test different detector configurations
    detector_configs = [
        ("naive", 1, float('inf')),
        ("window_5", 5, float('inf')),
        ("window_10", 10, float('inf')),
        ("window_10_budget_50", 10, 50.0),
        ("window_10_budget_25", 10, 25.0),
    ]
    
    detector_results = {}
    
    for name, window, budget in detector_configs:
        if config.verbose:
            print(f"[Testing {name}...]")
        
        detector = AntiZenoDetector(
            harm_threshold=config.harm_threshold,
            integration_window=window,
            risk_budget=budget
        )
        
        # Test on attacks
        detection_rates = []
        harms_achieved = []
        detection_delays = []
        
        for attack in attacks:
            detections, harm_achieved, detection_points = detector.check_sequence(attack.tuples)
            
            detection_rate = detections / len(attack.tuples) if attack.tuples else 0
            detection_rates.append(1.0 if detections > 0 else 0.0)  # Binary: did we catch it?
            harms_achieved.append(harm_achieved)
            
            if detection_points:
                detection_delays.append(detection_points[0])
            else:
                detection_delays.append(len(attack.tuples))  # Never detected
        
        # Test on clean sequences (should not flag)
        false_positives = 0
        for clean_seq in clean_sequences:
            detector.reset()
            detections, _, _ = detector.check_sequence(clean_seq)
            if detections > 0:
                false_positives += 1
        
        fp_rate = false_positives / len(clean_sequences) if clean_sequences else 0
        
        detector_results[name] = DetectorResult(
            detector_name=name,
            window_size=window,
            risk_budget=budget,
            mean_detection_rate=sum(detection_rates) / len(detection_rates) if detection_rates else 0,
            mean_harm_achieved=sum(harms_achieved) / len(harms_achieved) if harms_achieved else 0,
            mean_detection_delay=sum(detection_delays) / len(detection_delays) if detection_delays else 0,
            false_positive_rate=fp_rate
        )
        
        if config.verbose:
            r = detector_results[name]
            print(f"  Detection: {r.mean_detection_rate:.1%}, Harm: {r.mean_harm_achieved:.1f}, FP: {r.false_positive_rate:.1%}")
    
    runtime = time.time() - start_time
    
    # Check success criteria
    naive_result = detector_results.get("naive")
    antizeno_result = detector_results.get("window_10_budget_25")
    
    naive_misses = naive_result.mean_detection_rate < 0.5 if naive_result else False
    antizeno_catches = antizeno_result.mean_detection_rate > 0.8 if antizeno_result else False
    fp_ok = all(r.false_positive_rate < 0.1 for r in detector_results.values())
    
    exp_results = Exp4Results(
        detector_results=detector_results,
        attack_configs=[(a.n_steps, a.harm_per_step) for a in attacks[:5]],
        config=config,
        runtime_seconds=runtime,
        timestamp=datetime.now().isoformat(),
        naive_misses_attacks=naive_misses,
        antizeno_catches_attacks=antizeno_catches,
        fp_rate_acceptable=fp_ok
    )
    
    print(exp_results.summary())
    
    return exp_results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment 4: Zeno Attacks')
    parser.add_argument('--attacks', type=int, default=50, help='Number of attack sequences')
    parser.add_argument('--threshold', type=float, default=10.0, help='Harm threshold')
    parser.add_argument('--output', type=str, default='exp4_results.json')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    config = Exp4Config(
        n_attack_sequences=args.attacks,
        harm_threshold=args.threshold,
        verbose=not args.quiet
    )
    
    results = run_exp4_zeno(config)
    
    with open(args.output, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    return 0 if results.all_passed else 1


if __name__ == '__main__':
    exit(main())
