#!/usr/bin/env python3
"""
EXPERIMENT 3: ANTI-LAUNDERING RED TEAM
=======================================

Tests the CCD (Claim-Check-Delta) verification system.

Claim tested:
- C5: TAO anti-laundering (CCD) catches semantic/mechanical mismatches

Design:
- Generate adversarial TAO tuples (attacks)
- Run through CCD verifier
- Measure detection rate (recall) and false positive rate

Attack types:
1. Semantic relabeling: "HARM claimed as PROTECT"
2. Surgery excuse: Harm without required benefit
3. Context spoofing: Fake consent/authority
4. Benefit with harm claim: Reverse laundering

Success criteria:
- Detection rate > 95%
- False positive rate < 5%
"""

import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..tao.ccd import CCDVerifier, CCDStatus, AttackGenerator, TAOTuple


@dataclass
class Exp3Config:
    """Configuration for Experiment 3."""
    attacks_per_type: int = 100
    clean_tuples: int = 100
    strict_mode: bool = True
    verbose: bool = True


@dataclass
class AttackTypeResult:
    """Results for a single attack type."""
    attack_type: str
    total: int
    detected: int
    detection_rate: float
    false_negatives: int


@dataclass
class Exp3Results:
    """Results from Experiment 3."""
    attack_results: Dict[str, AttackTypeResult]
    overall_detection_rate: float
    false_positive_rate: float
    config: Exp3Config
    runtime_seconds: float
    timestamp: str
    
    # Success criteria
    passes_detection_threshold: bool  # > 95%
    passes_fp_threshold: bool         # < 5%
    
    @property
    def all_passed(self) -> bool:
        return self.passes_detection_threshold and self.passes_fp_threshold
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'experiment': 'EXP3_ANTI_LAUNDERING_RED_TEAM',
            'timestamp': self.timestamp,
            'runtime_seconds': self.runtime_seconds,
            'config': {
                'attacks_per_type': self.config.attacks_per_type,
                'clean_tuples': self.config.clean_tuples,
                'strict_mode': self.config.strict_mode
            },
            'attack_results': {
                name: {
                    'total': r.total,
                    'detected': r.detected,
                    'detection_rate': r.detection_rate,
                    'false_negatives': r.false_negatives
                }
                for name, r in self.attack_results.items()
            },
            'overall_detection_rate': self.overall_detection_rate,
            'false_positive_rate': self.false_positive_rate,
            'success_criteria': {
                'detection_gt_95pct': self.passes_detection_threshold,
                'fp_lt_5pct': self.passes_fp_threshold,
                'ALL_PASSED': self.all_passed
            }
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "EXPERIMENT 3: ANTI-LAUNDERING RED TEAM RESULTS",
            "=" * 60,
            "",
            f"Runtime: {self.runtime_seconds:.1f} seconds",
            f"Timestamp: {self.timestamp}",
            "",
            "ATTACK DETECTION RATES:",
            "-" * 40,
        ]
        
        for name, result in self.attack_results.items():
            status = "✓" if result.detection_rate >= 0.95 else "✗"
            lines.append(
                f"  {name:<25} {result.detection_rate:.1%} "
                f"({result.detected}/{result.total}) {status}"
            )
        
        lines.extend([
            "",
            "AGGREGATE METRICS:",
            "-" * 40,
            f"  Overall detection rate: {self.overall_detection_rate:.1%}",
            f"  False positive rate:    {self.false_positive_rate:.1%}",
            "",
            "SUCCESS CRITERIA:",
            "-" * 40,
            f"  Detection rate > 95%:   {'✓ PASS' if self.passes_detection_threshold else '✗ FAIL'}",
            f"  False positive < 5%:    {'✓ PASS' if self.passes_fp_threshold else '✗ FAIL'}",
            "",
            "=" * 60,
            f"OVERALL: {'✓ ALL CRITERIA PASSED' if self.all_passed else '✗ SOME CRITERIA FAILED'}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def run_exp3_antilaundering(config: Optional[Exp3Config] = None) -> Exp3Results:
    """
    Run Experiment 3: Anti-Laundering Red Team.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Exp3Results with detection rates and success criteria
    """
    config = config or Exp3Config()
    
    print("=" * 60)
    print("EXPERIMENT 3: ANTI-LAUNDERING RED TEAM")
    print("=" * 60)
    print(f"Config: {config.attacks_per_type} attacks/type, {config.clean_tuples} clean")
    print()
    
    start_time = time.time()
    
    # Initialize
    ccd = CCDVerifier(strict_mode=config.strict_mode)
    gen = AttackGenerator()
    
    # Generate attack suite
    if config.verbose:
        print("[Generating attack suite...]")
    suite = gen.generate_attack_suite(n_per_type=config.attacks_per_type)
    
    # Test each attack type
    attack_results = {}
    total_attacks = 0
    total_detected = 0
    
    attack_types = [
        "semantic_relabeling",
        "surgery_excuse", 
        "context_spoofing",
        "benefit_with_harm_claim",
    ]
    
    for attack_type in attack_types:
        if config.verbose:
            print(f"[Testing {attack_type}...]")
        
        attacks = suite[attack_type]
        detection_rate, false_negatives = ccd.compute_detection_rate(
            attacks, expected_contested=True
        )
        
        detected = int(detection_rate * len(attacks))
        attack_results[attack_type] = AttackTypeResult(
            attack_type=attack_type,
            total=len(attacks),
            detected=detected,
            detection_rate=detection_rate,
            false_negatives=len(false_negatives)
        )
        
        total_attacks += len(attacks)
        total_detected += detected
        
        if config.verbose:
            print(f"  → {detection_rate:.1%} detected")
    
    # Test clean tuples (should NOT be flagged)
    if config.verbose:
        print("[Testing clean tuples...]")
    
    clean_tuples = (
        suite["clean_cooperation"] + 
        suite["clean_defense"]
    )
    
    # For clean tuples, we want them VERIFIED (not contested)
    false_positives = 0
    for t in clean_tuples:
        result = ccd.verify(t)
        if result.status == CCDStatus.CONTESTED:
            false_positives += 1
    
    fp_rate = false_positives / len(clean_tuples) if clean_tuples else 0.0
    
    if config.verbose:
        print(f"  → {fp_rate:.1%} false positive rate")
    
    # Compute overall metrics
    overall_detection = total_detected / total_attacks if total_attacks > 0 else 0.0
    
    # FUZZING TEST: Generate randomized attacks
    if config.verbose:
        print("[Running fuzzing test (randomized attacks)...]")
    
    fuzzed = gen.generate_fuzzed_attacks(n_attacks=config.attacks_per_type * 2, seed=42)
    fuzz_correct = 0
    fuzz_total = len(fuzzed)
    
    for tuple_, should_catch in fuzzed:
        result = ccd.verify(tuple_)
        caught = result.status == CCDStatus.CONTESTED
        
        if should_catch and caught:
            fuzz_correct += 1  # True positive
        elif not should_catch and not caught:
            fuzz_correct += 1  # True negative
    
    fuzz_accuracy = fuzz_correct / fuzz_total if fuzz_total > 0 else 0.0
    
    if config.verbose:
        print(f"  → Fuzzing accuracy: {fuzz_accuracy:.1%} ({fuzz_correct}/{fuzz_total})")
    
    runtime = time.time() - start_time
    
    # Check success criteria
    passes_detection = overall_detection >= 0.95
    passes_fp = fp_rate <= 0.05
    
    results = Exp3Results(
        attack_results=attack_results,
        overall_detection_rate=overall_detection,
        false_positive_rate=fp_rate,
        config=config,
        runtime_seconds=runtime,
        timestamp=datetime.now().isoformat(),
        passes_detection_threshold=passes_detection,
        passes_fp_threshold=passes_fp
    )
    
    # Print summary
    print(results.summary())
    
    return results


def export_fuzz_corpus(output_path: str = "ccd_fuzz_corpus.json", n_per_type: int = 50, seed: int = 42):
    """
    Export fuzz corpus as reusable test vectors.
    
    This allows:
    1. Reproducible testing across CCD versions
    2. Regression testing after grammar updates
    3. External validation by other implementations
    
    Format:
    {
        "version": "1.0",
        "generated": "<timestamp>",
        "seed": 42,
        "test_vectors": [
            {
                "id": "fuzz_001",
                "category": "mismatch|missing_required|forbidden_present|clean",
                "claimed_verb": "PROTECT.DEFEND.OTHER",
                "effects": [...],
                "context": {...},
                "expected_status": "CONTESTED|VERIFIED",
                "should_catch": true|false
            },
            ...
        ],
        "fixed_attacks": {
            "semantic_relabeling": {...},
            "surgery_excuse": {...},
            "context_spoofing": {...},
            "benefit_with_harm_claim": {...}
        }
    }
    """
    from ..tao.ccd import AttackGenerator, CCDVerifier, ObservedEffect
    
    generator = AttackGenerator()
    verifier = CCDVerifier()
    
    # Generate fuzzed attacks
    fuzzed = generator.generate_fuzzed_attacks(n_per_type * 4, seed=seed)
    
    test_vectors = []
    for i, (tao_tuple, should_catch) in enumerate(fuzzed):
        # Run through verifier to get expected status
        result = verifier.verify(tao_tuple)
        
        vector = {
            "id": f"fuzz_{i:04d}",
            "category": "attack" if should_catch else "clean",
            "claimed_verb": tao_tuple.claimed_verb,
            "effects": [
                {
                    "effect_type": e.effect_type,
                    "target": e.target,
                    "magnitude": getattr(e, 'magnitude', None),
                    "source": getattr(e, 'source', None),
                }
                for e in tao_tuple.effects
            ],
            "context": tao_tuple.context,
            "expected_status": result.status.value,
            "should_catch": should_catch,
            "actual_caught": result.status.value == "CONTESTED",
            "correct": (result.status.value == "CONTESTED") == should_catch,
        }
        test_vectors.append(vector)
    
    # Add fixed attack patterns
    fixed_attacks = {
        "semantic_relabeling": _tuple_to_dict(generator.semantic_relabeling()),
        "surgery_excuse": _tuple_to_dict(generator.surgery_excuse()),
        "context_spoofing": _tuple_to_dict(generator.context_spoofing()),
        "benefit_with_harm_claim": _tuple_to_dict(generator.benefit_with_harm_claim()),
    }
    
    corpus = {
        "version": "1.0",
        "description": "CCD Anti-Laundering Test Vectors",
        "generated": datetime.now().isoformat(),
        "seed": seed,
        "n_vectors": len(test_vectors),
        "accuracy": sum(1 for v in test_vectors if v["correct"]) / len(test_vectors),
        "test_vectors": test_vectors,
        "fixed_attacks": fixed_attacks,
        "usage": {
            "purpose": "Regression testing for CCD verifier improvements",
            "expected_behavior": "CONTESTED for attacks, VERIFIED for clean",
            "improvement_tracking": "Compare accuracy across CCD versions",
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"Exported {len(test_vectors)} test vectors to {output_path}")
    print(f"Current accuracy: {corpus['accuracy']:.1%}")
    
    return corpus


def _tuple_to_dict(tao_tuple) -> dict:
    """Convert TAOTuple to JSON-serializable dict."""
    return {
        "claimed_verb": tao_tuple.claimed_verb,
        "effects": [
            {
                "effect_type": e.effect_type,
                "target": e.target,
                "magnitude": getattr(e, 'magnitude', None),
                "source": getattr(e, 'source', None),
            }
            for e in tao_tuple.effects
        ],
        "context": tao_tuple.context,
        "justification": getattr(tao_tuple, 'justification', {}),
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment 3: Anti-Laundering Red Team')
    parser.add_argument('--attacks', type=int, default=100, help='Attacks per type')
    parser.add_argument('--clean', type=int, default=100, help='Clean tuples to test')
    parser.add_argument('--output', type=str, default='exp3_results.json', help='Output file')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    parser.add_argument('--export-corpus', action='store_true', help='Export fuzz corpus as test vectors')
    parser.add_argument('--corpus-output', type=str, default='ccd_fuzz_corpus.json', help='Corpus output file')
    
    args = parser.parse_args()
    
    if args.export_corpus:
        export_fuzz_corpus(args.corpus_output)
        return 0
    
    config = Exp3Config(
        attacks_per_type=args.attacks,
        clean_tuples=args.clean,
        verbose=not args.quiet
    )
    
    results = run_exp3_antilaundering(config)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    return 0 if results.all_passed else 1


if __name__ == '__main__':
    exit(main())
