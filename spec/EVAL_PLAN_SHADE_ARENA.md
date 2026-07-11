# Pre-registered evaluation plan: CCD vs. LLM-monitor baselines on sabotage-evaluation data

**Companion to:** TAO working paper v0.14 · **Version:** 0.1 · **Date:** 2026-07-11 · **Status:** Pre-registered plan; not yet executed

The working paper's largest open liability is the absence of any empirical result. This plan pre-registers the study that closes it, before it is run, so that whatever the numbers turn out to be, they cannot be accused of being chosen after the fact. Publishing the plan is itself a commitment device — and an invitation: anyone with agent-trace data can run this before the author does and is encouraged to.

## 1. Hypotheses

- **H1.** On agent transcripts containing covert harmful side-tasks whose effects are mechanically visible at the tool-call layer, structured CCD (deterministic checks 1 and 3, plus an LLM-implemented teleological check emitting the five standard mismatch classes) achieves detection at least comparable to an end-to-end LLM monitor reading the same transcript, at matched false-positive rate.
- **H2.** A hybrid (LLM monitor given the CCD verdict trail as additional input) outperforms either alone.
- **H3.** CCD's advantage concentrates on side-tasks with clear mechanical signatures (exfiltration, unauthorized transfer, scope-exceeding tool use) and disappears on purely informational manipulation — consistent with the working paper's own limits exhibits (§8). A result confirming H3 is a success for the substrate's honesty, not a failure of the substrate.
- **Falsifier.** If CCD detects materially less than the LLM-monitor baseline on the mechanically visible class, or its INDETERMINATE rate on benign trajectories exceeds usability thresholds (below), the working paper's §10 "uneconomic/uninformative" concerns are partially confirmed and must be reported as such.

## 2. Data

Primary: **SHADE-Arena** (Kutasov et al. 2025, arXiv:2506.15740) — paired benign main tasks and covert harmful side-tasks in tool-using environments, with transcripts and an LLM-monitor baseline design; Anthropic released a public subset (remainder gated to preserve eval integrity — the gated portion is out of scope unless access is granted). Secondary, for benign false-positive calibration: any public benign agent-trace corpus from the same tool families (e.g., published AgentDojo benign runs or equivalent); exact corpus to be fixed at execution time and reported.

If the public SHADE-Arena subset proves too small for stable estimates, the fallback is synthetic augmentation *disclosed as such*: regenerate side-task trajectories in open agent environments matching SHADE-Arena's structure, holding the adapter fixed.

## 3. Method

1. **Adapter.** A transcript adapter converts each agent tool call into a TAO claim tuple: verb and target per `MAPPING_MCP_OTEL.md` §4; `justification.purpose` from the agent's task assignment and its stated rationale at the call site. The adapter is fixed before scoring any harmful trajectory (tuned only on benign traces) and its version hash is reported.
2. **Check side.** SHADE-Arena transcripts include ground-truth tool calls and environment state; check tuples are derived from the environment's record of what each call did (simulating a `PRIVILEGE_ISOLATED` observer with declared coverage = the tool-call layer). This is a simplification versus a production observer and is reported as such.
3. **CCD.** The reference validator runs checks 1 and 3 deterministically. The teleological check is implemented by a fixed judge model emitting only the five standard classes; prompt and model version pinned and published.
4. **Baseline.** An end-to-end LLM monitor (same model family and budget as the teleological judge, for fairness) reading the full transcript and emitting a suspicion score, per the SHADE-Arena monitoring setup.
5. **Scoring.** A trajectory is CCD-flagged at severity = number and class of INCONSISTENT verdicts (ordering: any DIRECT_CONTRADICTION > AUTHORITY_GOAL_MISMATCH > other classes > count). This ordinal score yields a ROC against the binary side-task ground truth.

## 4. Metrics (all reported, none optional)

- TPR at 1% and 5% FPR, and AUC, for CCD, LLM monitor, and hybrid — overall and split by mechanically-visible vs. informational side-task class (H3 split defined before scoring).
- INDETERMINATE rate on benign trajectories (usability threshold: if more than one in three benign trajectories yields a teleological INDETERMINATE, the substrate's per-action layer is not deployable without dossier-level aggregation, and this is reported as a limitation headline, not a footnote).
- Per-check attribution: which of the three checks carried each true detection (the working paper predicts checks 2 and 3 dominate on inference-heavy cases; this tests that prediction).
- Adapter cost: tuples per trajectory, tokens consumed by the teleological judge vs. the end-to-end monitor.

## 5. What this study cannot show

No result here demonstrates cross-vendor portability, real observer economics under production latency, adversarial adapter gaming, or long-horizon dossier detection — those need deployments, not benchmarks. The study answers one question: *does the structured comparison detect what an unstructured monitor detects, on the class of behavior it claims to cover?* That is the necessary first result, not the sufficient one.

## 6. Reporting commitment

Results publish in this repository regardless of direction, with transcripts of the adapter's tuples for every scored trajectory, the pinned judge prompts, and a limitations section written before the results section is populated.
