# TEMPER Test Checklist

> **See README.md for full documentation.** This is a quick reference checklist.

---

## Pre-Flight Checklist

- [ ] Extracted full zip (not just some files)
- [ ] Python 3.8+ installed
- [ ] Located `scripts/` folder

---

## Test Execution Order

### 1. Quick Check (1 min)
```
scripts/run_quick_check.command  (Mac)
scripts/run_quick_check.bat      (Windows)
```
- [ ] Shows `✅ Imports OK`
- [ ] Shows 3 green checkmarks

### 2. Gold Master (5-10 min)
```
scripts/run_gold_master.command  (Mac)
scripts/run_gold_master.bat      (Windows)
```
- [ ] Phase 1: Kernels bred
- [ ] Phase 2: Separation δ ≤ -0.9
- [ ] Phase 3: No noise crossover
- [ ] Phase 4: CCD ≥ 95%
- [ ] Phase 5: Zeno ≥ 95%
- [ ] Phase 6: Linkage ≥ 95%
- [ ] Final: `[PASS] All pre-registered thresholds met`

### 3. Ablation (5 min)
```
scripts/run_ablation.command  (Mac)
scripts/run_ablation.bat      (Windows)
```
- [ ] SURVIVAL_ONLY: ~2% harm, ~7% protection
- [ ] NORMATIVE: ~8% harm, ~57% protection
- [ ] Protection difference is huge (core thesis confirmed)

### 4. Cross-Domain Robustness (5 min)
```
scripts/run_cross_domain.command  (Mac)
scripts/run_cross_domain.bat      (Windows)
```
- [ ] Condition 1: δ ≤ -0.8
- [ ] Condition 2: δ ≤ -0.8
- [ ] Condition 3: δ ≤ -0.8
- [ ] Condition 4: δ ≤ -0.8
- [ ] All four conditions pass (proves separation isn't domain-specific)

### 5. Machiavelli (2 min)
```
scripts/run_machiavelli.command  (Mac)
scripts/run_machiavelli.bat      (Windows)
```
- [ ] Harm rate: ~28% (±3%)
- [ ] Reduction from baseline: 21+ percentage points
- [ ] No errors

### 6. RLHF (Optional, 10 min)
```
tempered_rlhf/install_deps.command  (first time only)
tempered_rlhf/run_experiment.command
```
- [ ] Dependencies installed
- [ ] Exploit reduction: ~90%
- [ ] Cohen's d ≈ 2.06

---

## All Tests Passed?

```
[ ] Quick Check    ✓
[ ] Gold Master    ✓
[ ] Ablation       ✓
[ ] Cross-Domain   ✓
[ ] Machiavelli    ✓
[ ] RLHF           ✓ (optional)
```

**You're GitHub-ready!** 🚀

---

## If Something Fails

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Import errors | Incomplete extraction | Re-extract full zip |
| Permission denied | macOS security | Right-click → Open |
| Python not found | Not installed | Install from python.org |
| Wrong numbers | Old cached results | Delete `results/` folders, re-run |
