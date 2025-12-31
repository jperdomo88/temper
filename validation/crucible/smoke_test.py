#!/usr/bin/env python3
"""
SMOKE TEST - Verify TEMPER Validation Platform Works
=====================================================

Quick test to make sure everything is wired up correctly.
Run this before running full experiments.
"""

import sys
import time

def test_imports():
    """Test that all modules import correctly."""
    print("[1/5] Testing imports...")
    
    from crucible.core.state import encode_state, N_STATES, AllyStatus, ThreatLevel
    from crucible.core.agents import Action, N_ACTIONS, MaximizerAgent, FrozenAgent, HedonicAgent
    from crucible.core.metrics import cohens_d, compute_effect_size
    from crucible.core.simulation import Simulation, SimulationParams, SwitchboardConfig
    
    print(f"  ✓ State space: {N_STATES} states")
    print(f"  ✓ Action space: {N_ACTIONS} actions")
    print(f"  ✓ All imports successful")
    return True


def test_state_encoding():
    """Test state encoding/decoding."""
    print("\n[2/5] Testing state encoding...")
    
    from crucible.core.state import encode_state, decode_state, AllyStatus, ThreatLevel
    
    # Test encode/decode roundtrip
    state = encode_state(50.0, AllyStatus.HEALTHY, ThreatLevel.PRESENT)
    res, ally, threat = decode_state(state)
    
    assert ally == AllyStatus.HEALTHY, f"Expected HEALTHY, got {ally}"
    assert threat == ThreatLevel.PRESENT, f"Expected PRESENT, got {threat}"
    
    print(f"  ✓ State {state} encodes resources=50, ally=HEALTHY, threat=PRESENT")
    return True


def test_switchboard_configs():
    """Test switchboard configuration."""
    print("\n[3/5] Testing switchboard configs...")
    
    from crucible.core.simulation import SwitchboardConfig
    
    temper = SwitchboardConfig.temper_full()
    maximizer = SwitchboardConfig.maximizer_full()
    
    assert not temper.visible_metric, "TEMPER should not have visible metric"
    assert maximizer.visible_metric, "MAXIMIZER should have visible metric"
    assert temper.name == "TEMPER_FULL", f"Expected TEMPER_FULL, got {temper.name}"
    
    # Test bit encoding
    for i in range(32):
        config = SwitchboardConfig.from_bits(i)
        assert config.to_bits() == i, f"Bit roundtrip failed for {i}"
    
    print(f"  ✓ TEMPER_FULL: visible={temper.visible_metric}, learning={temper.learning_enabled}")
    print(f"  ✓ MAXIMIZER_FULL: visible={maximizer.visible_metric}, learning={maximizer.learning_enabled}")
    print(f"  ✓ All 32 bit configurations valid")
    return True


def test_simulation_runs():
    """Test that simulation actually runs."""
    print("\n[4/5] Testing simulation execution...")
    
    from crucible.core.simulation import Simulation, SimulationParams, SwitchboardConfig
    from crucible.core.agents import AgentType
    
    params = SimulationParams(initial_population=6)
    sw = SwitchboardConfig.temper_full()
    
    sim = Simulation(params, sw, seed=42)
    sim.initialize(AgentType.HEDONIC)
    
    # Run for 10 turns
    for _ in range(10):
        result = sim.step()
        if result.get('ended'):
            break
    
    summary = sim.get_summary()
    
    print(f"  ✓ Simulation ran for {summary['turns']} turns")
    print(f"  ✓ {summary['alive']} agents alive")
    print(f"  ✓ Harm rate: {summary['harm_rate']:.3f}")
    print(f"  ✓ Total events: {summary['total_events']}")
    return True


def test_breeding():
    """Test that breeding produces a kernel."""
    print("\n[5/5] Testing breeding (quick, 5 generations)...")
    
    from crucible.core.simulation import (
        breed_population, saint_fitness, 
        SimulationParams, SwitchboardConfig
    )
    from crucible.core.agents import N_STATES, N_ACTIONS
    
    params = SimulationParams(initial_population=6)
    sw = SwitchboardConfig.temper_full()
    
    kernel = breed_population(
        saint_fitness, params, sw,
        pop_size=5, generations=5, eval_seeds=2, verbose=False
    )
    
    assert kernel is not None, "Breeding returned None"
    assert len(kernel) == N_STATES, f"Expected {N_STATES} states, got {len(kernel)}"
    assert len(kernel[0]) == N_ACTIONS, f"Expected {N_ACTIONS} actions, got {len(kernel[0])}"
    
    print(f"  ✓ Bred kernel: {N_STATES} states × {N_ACTIONS} actions")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("TEMPER VALIDATION PLATFORM - SMOKE TEST")
    print("=" * 60)
    
    start = time.time()
    
    tests = [
        test_imports,
        test_state_encoding,
        test_switchboard_configs,
        test_simulation_runs,
        test_breeding,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed in {elapsed:.1f}s")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ ALL SMOKE TESTS PASSED - Ready to run experiments!")
        return 0
    else:
        print(f"\n✗ {failed} TESTS FAILED - Fix before proceeding")
        return 1


if __name__ == '__main__':
    sys.exit(main())
