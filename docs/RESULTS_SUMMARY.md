# TEMPER Empirical Results Summary

**Ground Truth:** See `paper/TEMPER_Paper_v11.docx` Section 8-10 for complete methodology and results.

This document provides a quick reference to key findings. The paper is the authoritative source.

---

## Key Results at a Glance

### Core Thesis (Exp 0: Fitness Ablation)
| Condition | Harm | Protection | Engagement |
|-----------|------|------------|------------|
| **NORMATIVE** | 7.6% | **57.5%** | 92.0% |
| SURVIVAL_ONLY | 2.2% | 7.0% | 44.7% |
| BRUTE | 77.3% | 5.2% | 96.2% |

**Key Finding:** Low harm ≠ alignment. SURVIVAL achieves low harm through disengagement; NORMATIVE achieves alignment through protective engagement.

- Protection effect: Cliff's δ = **+1.00** (perfect separation)
- Harm effect: Cliff's δ = **-1.00** (NORMATIVE vs BRUTE)

### Architecture Validation (Exp 1: Switchboard)
- TEMPER vs MAXIMIZER: Cohen's d = **-10.46**
- Cliff's δ = **-1.00** (complete separation)
- Harm: TEMPER 19.6% vs MAXIMIZER 86.7%

### Transfer Validation (Exp H: Combined)
- Cliff's δ = **-1.00** in ALL FOUR conditions
- Cross-domain (INFO, COMMITMENT) ✓
- Cross-adapter (Semantic, Effect-based) ✓

### Machiavelli Behavioral Transfer (Exp M) - UPDATED Dec 2025
- **SAINT harm rate: 28.4%** (21.6 pp reduction vs 50% baseline)
- Contextual adapter methodology: Claude-based TAO classification with ally/threat context
- Ground-truth harm labels used only for evaluation, not adapter input
- Sample: 100 scenes from "180 Files: The Aegis Project"
- **Beyond harm rate:** SAINT exhibits coherent behavioral profiles
  - Protection +42% when ally attacked vs alone
  - Strategic deception only in threat-present contexts
  - Coalition dynamics emerge from breeding

### Governance Detection Rates
| Mechanism | Detection | False Positive |
|-----------|-----------|----------------|
| CCD (fixed patterns) | 100% | 0% |
| CCD (fuzzed) | 77.5% | 0% |
| Zeno (with budget) | 100% | 0% |
| Linkage | 100% | 0% |
| Composition | 100% | 0% |

### Tempered RLHF (Exp TR)
- Exploit_z: Proxy 19.9 → Tempered 1.5 (~92% reduction)
- Cohen's d = **2.06** (large effect)
- p = 0.0007, N = 15

---

## Verification

Run `scripts/run_verify.command` (Mac) or `scripts/run_verify.bat` (Windows) to verify:
- Determinism: 28.4884% harm rate with seed=123
- CCD detection: 100% on fixed patterns
- Quick ablation: NORMATIVE > SURVIVAL on protection

---

## Pre-registered Thresholds (All Passed)

| Threshold | Required | Observed |
|-----------|----------|----------|
| Cliff's δ | ≤ -0.8 | **-1.00** |
| Noise crossover | None at 50% | **None** |
| CCD detection | ≥ 95% | **100%** |
| Zeno detection | ≥ 95% | **100%** |
| Linkage detection | ≥ 95% | **100%** |

---

*For full methodology, statistical details, and discussion, see the paper.*
