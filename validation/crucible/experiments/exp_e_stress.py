#!/usr/bin/env python3
"""
EXPERIMENT E: GOVERNANCE STRESS TESTS
======================================

Tests operational feasibility of the governance stack.

Metrics:
- Throughput: tuples/second
- Latency: p50, p95, p99 decision time
- Fail-safe behavior: what happens when components fail

Success criteria:
- Throughput > 1000 tuples/sec
- p95 latency < 10ms
- Fail-safe prevents catastrophe without bricking system
"""

import json
import time
import random
import statistics
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..tao.ccd import TAOTuple, ObservedEffect, CCDVerifier
from ..tao.ontology import MVS_VERBS
from ..governance.governor import BlindGovernor, GovernorDecision
from ..governance.profiles import SAINT_PROFILE


@dataclass
class ExpEConfig:
    """Configuration for Experiment E."""
    n_tuples: int = 1000              # Tuples for throughput test
    latency_samples: int = 500        # Samples for latency distribution
    fail_safe_scenarios: int = 50     # Fail-safe test scenarios
    latency_budget_ms: float = 10.0   # Governor latency budget
    verbose: bool = True


def generate_random_tuple() -> TAOTuple:
    """Generate a random TAO tuple for stress testing."""
    verbs = list(MVS_VERBS.keys())
    verb = random.choice(verbs)
    
    effect_types = [
        "RESOURCE.TRANSFER", "RESOURCE.DAMAGE",
        "CAPABILITY.ENABLE", "CAPABILITY.RESTRICT",
        "INFO.DISCLOSE", "INFO.WITHHOLD",
        "COMMITMENT.MAKE", "NO_EFFECT"
    ]
    
    effects = [
        ObservedEffect(
            effect_type=random.choice(effect_types),
            target=f"target_{random.randint(1, 100):03d}",
            magnitude=random.uniform(-100, 100)
        )
        for _ in range(random.randint(1, 3))
    ]
    
    return TAOTuple(
        claimed_verb=verb,
        effects=effects,
        context={"test": True}
    )


@dataclass
class ThroughputResult:
    """Throughput test results."""
    total_tuples: int
    total_time_seconds: float
    tuples_per_second: float
    decisions: Dict[str, int]  # ALLOW/BLOCK/ESCALATE counts


@dataclass
class LatencyResult:
    """Latency test results."""
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    mean_ms: float


@dataclass
class FailSafeResult:
    """Fail-safe test results."""
    scenario: str
    tuples_tested: int
    blocked_correctly: int
    allowed_incorrectly: int
    system_crashed: bool
    
    @property
    def safe(self) -> bool:
        """Did fail-safe prevent catastrophe?"""
        return not self.system_crashed and self.allowed_incorrectly == 0


@dataclass
class ExpEResults:
    """Results from Experiment E."""
    throughput: ThroughputResult
    latency: LatencyResult
    fail_safe: Dict[str, FailSafeResult]
    config: ExpEConfig
    runtime_seconds: float
    timestamp: str
    
    # Success criteria
    throughput_ok: bool      # > 1000/sec
    latency_ok: bool         # p95 < 10ms
    fail_safe_ok: bool       # All scenarios safe
    
    @property
    def all_passed(self) -> bool:
        return self.throughput_ok and self.latency_ok and self.fail_safe_ok
    
    def to_dict(self) -> Dict:
        return {
            'experiment': 'EXPE_GOVERNANCE_STRESS_TESTS',
            'timestamp': self.timestamp,
            'runtime_seconds': self.runtime_seconds,
            'throughput': {
                'total_tuples': self.throughput.total_tuples,
                'total_time_seconds': self.throughput.total_time_seconds,
                'tuples_per_second': self.throughput.tuples_per_second,
                'decisions': self.throughput.decisions
            },
            'latency': {
                'samples': self.latency.samples,
                'p50_ms': self.latency.p50_ms,
                'p95_ms': self.latency.p95_ms,
                'p99_ms': self.latency.p99_ms,
                'max_ms': self.latency.max_ms,
                'min_ms': self.latency.min_ms,
                'mean_ms': self.latency.mean_ms
            },
            'fail_safe': {
                name: {
                    'tuples_tested': r.tuples_tested,
                    'blocked_correctly': r.blocked_correctly,
                    'allowed_incorrectly': r.allowed_incorrectly,
                    'system_crashed': r.system_crashed,
                    'safe': r.safe
                }
                for name, r in self.fail_safe.items()
            },
            'success_criteria': {
                'throughput_gt_1000': self.throughput_ok,
                'latency_p95_lt_10ms': self.latency_ok,
                'fail_safe_all_safe': self.fail_safe_ok,
                'ALL_PASSED': self.all_passed
            }
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EXPERIMENT E: GOVERNANCE STRESS TEST RESULTS",
            "=" * 60,
            "",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            "",
            "THROUGHPUT:",
            "-" * 50,
            f"  Tuples processed: {self.throughput.total_tuples}",
            f"  Time: {self.throughput.total_time_seconds:.3f} seconds",
            f"  Rate: {self.throughput.tuples_per_second:.0f} tuples/sec",
            "",
            "LATENCY DISTRIBUTION:",
            "-" * 50,
            f"  p50:  {self.latency.p50_ms:.3f} ms",
            f"  p95:  {self.latency.p95_ms:.3f} ms",
            f"  p99:  {self.latency.p99_ms:.3f} ms",
            f"  max:  {self.latency.max_ms:.3f} ms",
            "",
            "FAIL-SAFE SCENARIOS:",
            "-" * 50,
        ]
        
        for name, result in self.fail_safe.items():
            status = "✓ SAFE" if result.safe else "✗ UNSAFE"
            lines.append(f"  {name:<30} {status}")
        
        lines.extend([
            "",
            "SUCCESS CRITERIA:",
            "-" * 50,
            f"  Throughput > 1000/sec:    {'✓ PASS' if self.throughput_ok else '✗ FAIL'} ({self.throughput.tuples_per_second:.0f}/sec)",
            f"  Latency p95 < 10ms:       {'✓ PASS' if self.latency_ok else '✗ FAIL'} ({self.latency.p95_ms:.3f}ms)",
            f"  All fail-safe scenarios:  {'✓ PASS' if self.fail_safe_ok else '✗ FAIL'}",
            "",
            "=" * 60,
            f"OVERALL: {'✓ ALL CRITERIA PASSED' if self.all_passed else '✗ SOME CRITERIA FAILED'}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def test_throughput(governor: BlindGovernor, n_tuples: int) -> ThroughputResult:
    """Test decision throughput."""
    tuples = [generate_random_tuple() for _ in range(n_tuples)]
    decisions = {"ALLOW": 0, "BLOCK": 0, "ESCALATE": 0}
    
    start = time.perf_counter()
    
    for t in tuples:
        result = governor.decide(t, {"consent": "EXPLICIT"})
        decisions[result.decision.value] += 1
    
    elapsed = time.perf_counter() - start
    
    return ThroughputResult(
        total_tuples=n_tuples,
        total_time_seconds=elapsed,
        tuples_per_second=n_tuples / elapsed if elapsed > 0 else 0,
        decisions=decisions
    )


def test_throughput_with_ccd_logging(governor: BlindGovernor, ccd: CCDVerifier, n_tuples: int) -> ThroughputResult:
    """
    Test throughput with FULL pipeline: CCD verification + Governor + logging.
    
    This is the realistic benchmark GPT requested.
    """
    tuples = [generate_random_tuple() for _ in range(n_tuples)]
    decisions = {"ALLOW": 0, "BLOCK": 0, "ESCALATE": 0}
    
    # Simulate audit log
    audit_log = []
    
    start = time.perf_counter()
    
    for t in tuples:
        # Step 1: CCD verification
        ccd_result = ccd.verify(t)
        
        # Step 2: Governor decision
        result = governor.decide(t, {"consent": "EXPLICIT", "ccd_status": ccd_result.status.value})
        decisions[result.decision.value] += 1
        
        # Step 3: Logging (simulate serialization overhead)
        log_entry = {
            "timestamp": time.time(),
            "verb": t.claimed_verb,
            "ccd_status": ccd_result.status.value,
            "decision": result.decision.value,
            "reason": result.reason[:50] if result.reason else "",
        }
        audit_log.append(log_entry)
    
    elapsed = time.perf_counter() - start
    
    return ThroughputResult(
        total_tuples=n_tuples,
        total_time_seconds=elapsed,
        tuples_per_second=n_tuples / elapsed if elapsed > 0 else 0,
        decisions=decisions
    )


def test_latency(governor: BlindGovernor, n_samples: int) -> LatencyResult:
    """Test decision latency distribution."""
    latencies_ms = []
    
    for _ in range(n_samples):
        t = generate_random_tuple()
        
        start = time.perf_counter_ns()
        governor.decide(t, {"consent": "EXPLICIT"})
        elapsed_ns = time.perf_counter_ns() - start
        
        latencies_ms.append(elapsed_ns / 1_000_000)  # Convert to ms
    
    latencies_ms.sort()
    
    return LatencyResult(
        samples=n_samples,
        p50_ms=latencies_ms[int(n_samples * 0.50)],
        p95_ms=latencies_ms[int(n_samples * 0.95)],
        p99_ms=latencies_ms[int(n_samples * 0.99)],
        max_ms=max(latencies_ms),
        min_ms=min(latencies_ms),
        mean_ms=statistics.mean(latencies_ms)
    )


def test_fail_safe_ccd_failure(governor: BlindGovernor, n_scenarios: int) -> FailSafeResult:
    """
    Test: What happens when CCD verification fails?
    
    Expected: Governor should BLOCK (fail-safe).
    """
    blocked = 0
    allowed = 0
    
    # Generate tuples that would cause CCD to be uncertain
    for _ in range(n_scenarios):
        # Unknown verb (not in MVS)
        t = TAOTuple(
            claimed_verb="UNKNOWN.FAKE.VERB",
            effects=[ObservedEffect("RESOURCE.DAMAGE", "victim", -50)]
        )
        
        result = governor.decide(t, {})
        
        if result.decision in (GovernorDecision.BLOCK, GovernorDecision.ESCALATE):
            blocked += 1
        else:
            allowed += 1
    
    return FailSafeResult(
        scenario="ccd_unknown_verb",
        tuples_tested=n_scenarios,
        blocked_correctly=blocked,
        allowed_incorrectly=allowed,
        system_crashed=False
    )


def test_fail_safe_latency_exceeded(n_scenarios: int) -> FailSafeResult:
    """
    Test: What happens when latency budget is exceeded?
    
    Expected: Governor should BLOCK (fail-safe).
    """
    # Create governor with very tight latency budget
    tight_governor = BlindGovernor(
        profile=SAINT_PROFILE,
        latency_budget_ms=0.001  # Impossibly tight: 1 microsecond
    )
    
    blocked = 0
    allowed = 0
    
    for _ in range(n_scenarios):
        t = generate_random_tuple()
        result = tight_governor.decide(t, {"consent": "EXPLICIT"})
        
        if result.decision == GovernorDecision.BLOCK:
            blocked += 1
        else:
            allowed += 1
    
    return FailSafeResult(
        scenario="latency_exceeded",
        tuples_tested=n_scenarios,
        blocked_correctly=blocked,
        allowed_incorrectly=allowed,
        system_crashed=False
    )


def test_fail_safe_missing_context(governor: BlindGovernor, n_scenarios: int) -> FailSafeResult:
    """
    Test: What happens when required context is missing?
    
    Expected: Governor should ESCALATE or BLOCK for consent-required verbs.
    """
    blocked_or_escalated = 0
    allowed = 0
    
    for _ in range(n_scenarios):
        # Use a verb that requires consent under SAINT profile
        t = TAOTuple(
            claimed_verb="COOPERATE.ASSIST.HELP",
            effects=[ObservedEffect("RESOURCE.TRANSFER", "recipient", 50)]
        )
        
        # Missing consent context
        result = governor.decide(t, {})  # No consent field
        
        if result.decision in (GovernorDecision.BLOCK, GovernorDecision.ESCALATE):
            blocked_or_escalated += 1
        else:
            allowed += 1
    
    return FailSafeResult(
        scenario="missing_consent",
        tuples_tested=n_scenarios,
        blocked_correctly=blocked_or_escalated,
        allowed_incorrectly=allowed,
        system_crashed=False
    )


def test_fail_safe_high_vulnerability(governor: BlindGovernor, n_scenarios: int) -> FailSafeResult:
    """
    Test: What happens with high-vulnerability targets?
    
    Expected: Harm should be blocked.
    """
    blocked = 0
    allowed = 0
    
    for _ in range(n_scenarios):
        # Harm verb with high vulnerability target
        t = TAOTuple(
            claimed_verb="HARM.DAMAGE.STRIKE",
            effects=[ObservedEffect("RESOURCE.DAMAGE", "vulnerable_target", -50)]
        )
        
        result = governor.decide(t, {"vulnerability": 0.9})
        
        if result.decision == GovernorDecision.BLOCK:
            blocked += 1
        else:
            allowed += 1
    
    return FailSafeResult(
        scenario="high_vulnerability",
        tuples_tested=n_scenarios,
        blocked_correctly=blocked,
        allowed_incorrectly=allowed,
        system_crashed=False
    )


def run_exp_e_stress(config: Optional[ExpEConfig] = None) -> ExpEResults:
    """
    Run Experiment E: Governance Stress Tests.
    """
    config = config or ExpEConfig()
    
    print("=" * 60)
    print("EXPERIMENT E: GOVERNANCE STRESS TESTS")
    print("=" * 60)
    print(f"Config: {config.n_tuples} throughput, {config.latency_samples} latency samples")
    print()
    
    start_time = time.time()
    
    # Initialize governor
    governor = BlindGovernor(
        profile=SAINT_PROFILE,
        latency_budget_ms=config.latency_budget_ms
    )
    
    # Initialize CCD for realistic benchmark
    ccd = CCDVerifier(strict_mode=True)
    
    # Throughput test (governor only - microbenchmark)
    if config.verbose:
        print("[Testing throughput (governor only)...]")
    throughput = test_throughput(governor, config.n_tuples)
    if config.verbose:
        print(f"  {throughput.tuples_per_second:.0f} tuples/sec (microbenchmark)")
    
    # Throughput test WITH CCD + logging (realistic)
    if config.verbose:
        print("[Testing throughput (CCD + governor + logging)...]")
    throughput_full = test_throughput_with_ccd_logging(governor, ccd, config.n_tuples)
    if config.verbose:
        print(f"  {throughput_full.tuples_per_second:.0f} tuples/sec (full pipeline)")
    
    # Latency test
    if config.verbose:
        print("[Testing latency...]")
    latency = test_latency(governor, config.latency_samples)
    if config.verbose:
        print(f"  p50={latency.p50_ms:.3f}ms, p95={latency.p95_ms:.3f}ms, p99={latency.p99_ms:.3f}ms")
    
    # Fail-safe tests
    if config.verbose:
        print("[Testing fail-safe scenarios...]")
    
    fail_safe = {
        "ccd_unknown_verb": test_fail_safe_ccd_failure(governor, config.fail_safe_scenarios),
        "latency_exceeded": test_fail_safe_latency_exceeded(config.fail_safe_scenarios),
        "missing_consent": test_fail_safe_missing_context(governor, config.fail_safe_scenarios),
        "high_vulnerability": test_fail_safe_high_vulnerability(governor, config.fail_safe_scenarios),
    }
    
    if config.verbose:
        for name, result in fail_safe.items():
            status = "✓" if result.safe else "✗"
            print(f"  {name}: {status} (blocked {result.blocked_correctly}/{result.tuples_tested})")
    
    runtime = time.time() - start_time
    
    # Check success criteria
    throughput_ok = throughput.tuples_per_second > 1000
    latency_ok = latency.p95_ms < 10.0
    fail_safe_ok = all(r.safe for r in fail_safe.values())
    
    exp_results = ExpEResults(
        throughput=throughput,
        latency=latency,
        fail_safe=fail_safe,
        config=config,
        runtime_seconds=runtime,
        timestamp=datetime.now().isoformat(),
        throughput_ok=throughput_ok,
        latency_ok=latency_ok,
        fail_safe_ok=fail_safe_ok
    )
    
    print(exp_results.summary())
    
    return exp_results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment E: Stress Tests')
    parser.add_argument('--tuples', type=int, default=1000, help='Tuples for throughput')
    parser.add_argument('--latency-samples', type=int, default=500, help='Latency samples')
    parser.add_argument('--output', type=str, default='exp_e_results.json')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    config = ExpEConfig(
        n_tuples=args.tuples,
        latency_samples=args.latency_samples,
        verbose=not args.quiet
    )
    
    results = run_exp_e_stress(config)
    
    with open(args.output, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    return 0 if results.all_passed else 1


if __name__ == '__main__':
    exit(main())
