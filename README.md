# TEMPER Governance Framework

**Zero-Trust Governance for Agentic AI**

Based on the paper "Zero-Trust Governance for Agentic AI: Typed Action Interfaces, Effect Attestation, and Anti-Goodhart Enforcement" by Jorge Perdomo.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

TEMPER is an open-source framework for AI behavioral governance that treats alignment as a **game design problem**, not a training problem. Instead of trying to instill values in AI systems (which they can fake), TEMPER creates environments where defection becomes structurally irrational.

The framework provides:
- **TAO (TEMPER Action Ontology)**: A universal vocabulary for describing AI actions (39 verbs, 9 effect types)
- **Claim-Check-Delta (CCD)**: Verification mechanism binding semantic claims to mechanical effects
- **Mission Profiles**: Domain-specific policy configurations
- **Behavioral Breeding**: Fitness-hidden selection for preference formation

## Key Results

| Experiment | Finding | Effect Size |
|------------|---------|-------------|
| Fitness Ablation | Low harm ≠ alignment; normative selection required | δ = +1.00 (protection) |
| Defense-in-Depth | TEMPER vs MAXIMIZER separation | d = -10.46 |
| Cross-Domain Transfer | Behavioral stability across contexts | 98.2% gap retention, δ = -1.00 |
| Cross-Domain Robustness | Separation persists across domains AND adapters | δ = -1.00 (all 4 conditions) |
| **Machiavelli Transfer** | **Behavioral profiles transfer to text games** | **28.4% harm (21.6 pp reduction)** |
| CCD Detection | Semantic laundering caught | 100% fixed, 77.5% fuzzed |
| Tempered RLHF | Exploit reduction vs proxy training | d = 2.06 (~90% reduction) |

---

## Quick Start

**No terminal needed!** All tests are double-click scripts in the `scripts/` folder.

### Test Scripts

| Script | What It Tests | Time | Paper Section |
|--------|---------------|------|---------------|
| `run_quick_check.*` | Imports + determinism | 1 min | - |
| `run_gold_master.*` | **Comprehensive suite** (see below) | 5-10 min | §9.2-9.5 |
| `run_ablation.*` | Core thesis: low harm ≠ alignment | 5 min | §9.1 |
| `run_cross_domain.*` | Cross-domain robustness | 5 min | §9.4 |
| `run_machiavelli.*` | Text game behavioral transfer | 2 min | §9.6 |

Use `.command` files on Mac/Linux, `.bat` files on Windows.

### What Gold Master Covers

The `run_gold_master` script runs 6 phases in one harness:
1. **Kernel breeding** - Creates SAINT and BRUTE behavioral kernels
2. **Separation** (Exp 1) - Defense-in-depth validation
3. **Noise robustness** (Exp A) - Stability under perturbation
4. **CCD detection** (Exp 3) - Semantic laundering detection
5. **Zeno detection** (Exp 4) - Threshold evasion detection
6. **Linkage detection** (Exp D) - Intent≠outcome verification

### Expected Results

| Test | What Success Looks Like |
|------|-------------------------|
| Quick Check | `✅ Imports OK` + 3 green checkmarks |
| Gold Master | `[PASS] All pre-registered thresholds met` |
| Ablation | NORMATIVE protection (57%) >> SURVIVAL (7%) |
| Cross-Domain | δ ≤ -0.8 in all 4 conditions |
| Machiavelli | ~28% harm rate (21+ pp reduction from baseline) |

### RLHF Experiment (Requires PyTorch)

Located in `tempered_rlhf/` folder:
1. Install dependencies: double-click `install_deps.command` (Mac) or `.bat` (Windows)
2. Run experiment: double-click `run_experiment.command` or `.bat`

**Expect:** ~90% reduction in exploitability (Cohen's d ≈ 2.06)

---

## Repository Structure

```
temper-governance/
├── README.md                  # This file
├── QUICK_TEST.md              # Detailed test guide
├── CHANGELOG.md               # Version history
├── LICENSE                    # CC-BY-4.0 / Apache 2.0
│
├── paper/                     # Academic paper
│   ├── TEMPER_Paper_v11.docx
│   └── TEMPER_Paper_v11.pdf
│
├── spec/                      # TAO Specification
│   ├── TAO_v0_9_0.md
│   ├── TAO_v0_9_0.docx
│   └── TAO_v0_9_0.pdf
│
├── docs/                      # Documentation
│   ├── EMPIRICAL_RESULTS_v6.md  # Complete experimental methodology
│   ├── RESULTS_SUMMARY.md       # Quick reference
│   └── DEVELOPMENT_HISTORY.md   # Project history
│
├── scripts/                   # Double-click test runners
│   ├── run_quick_check.*      # Fast sanity check
│   ├── run_gold_master.*      # Comprehensive suite
│   ├── run_ablation.*         # Core thesis (Exp 0)
│   ├── run_cross_domain.*      # Cross-domain robustness (Exp H)
│   └── run_machiavelli.*      # Text game transfer (Exp M)
│
├── validation/                # Experiment code
│   ├── crucible/              # The Crucible - behavioral breeding environment
│   │   ├── core/              # Simulation, agents, metrics
│   │   ├── environments/      # Switchboard gridworld
│   │   ├── governance/        # Governor, mission profiles
│   │   ├── tao/               # TAO ontology, CCD verification
│   │   └── experiments/       # All experiment implementations
│   │
│   ├── machiavelli/           # Machiavelli behavioral transfer
│   │   ├── kernels/           # SAINT/BRUTE Q-tables
│   │   ├── data/              # AEGIS scenes + classifications
│   │   └── results/           # Test outputs
│   │
│   ├── verify_local.py        # Quick verification
│   ├── gold_master_hostile.py # Comprehensive test suite
│   └── ablation_fitness.py    # Fitness ablation experiment
│
└── tempered_rlhf/             # RLHF experiment (requires PyTorch)
    ├── src/                   # Experiment code
    ├── results/               # Saved results
    ├── install_deps.*         # Dependency installer
    └── run_experiment.*       # Run RLHF test
```

---

## Paper ↔ Code Mapping

| Paper Term | Code Location |
|------------|---------------|
| "Crucible" | `validation/crucible/` |
| "Switchboard environment" | `validation/crucible/environments/switchboard.py` |
| "Fitness-hidden selection" | `validation/crucible/core/simulation.py` |
| "Governor" | `validation/crucible/governance/governor.py` |
| "TAO Ontology" | `validation/crucible/tao/ontology.py` |
| "CCD" | `validation/crucible/tao/ccd.py` |
| "Exp 0 (Ablation)" | `validation/ablation_fitness.py` |
| "Exp 1-D (Crucible)" | `validation/gold_master_hostile.py` |
| "Exp H (Combined)" | `validation/crucible/experiments/exp_h_combined.py` |
| "Exp M (Machiavelli)" | `validation/machiavelli/` |
| "Exp TR (RLHF)" | `tempered_rlhf/` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Extract the full zip - don't cherry-pick files |
| "Permission denied" (Mac) | Right-click → Open → Open Anyway |
| "Python not found" | Install Python 3.8+ from python.org |
| PyTorch errors in RLHF | Run `install_deps.*` first |

---

## Citation

```bibtex
@article{perdomo2025temper,
  title={Zero-Trust Governance for Agentic AI: Typed Action Interfaces, 
         Effect Attestation, and Anti-Goodhart Enforcement},
  author={Perdomo, Jorge},
  year={2025}
}
```

## License

- **Documentation & Prose**: CC-BY-4.0
- **Code & Schemas**: Apache 2.0

You are free to use, modify, and redistribute. Attribution required.

---

*"The goal is not to make AI that wants to be good. The goal is to make environments where being good is the winning strategy."*
