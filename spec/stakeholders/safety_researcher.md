# For a safety researcher

**Reader's question:** What does this give me to study that other instrumentations don't?

## The short answer

A structured behavioral surface independent of model internals. Most AI safety instrumentation is either (a) introspection — what the model says about itself, sampling its activations, asking it to reason out loud — or (b) ad-hoc behavioral logging that lives in each operator's stack and isn't comparable across deployments. TAO is a third thing: a standardized record of what the agent *did*, with a structured semantic layer that lets you ask cross-deployment questions.

## What's novel

Three properties of the substrate that don't compose from existing tools:

### 1. Claim-vs-check as a first-class object

The CCD layer pairs the agent's own description of its action with an independent observer's record, in the same schema. The mismatch — when there is one — is a *behavioral* signal of an agent's representation of itself diverging from its actual effect. This is a research handle on a question most safety work approaches through model internals: when does the agent's stated purpose diverge from its actual effect?

The five teleological mismatch classes (`DIRECT_CONTRADICTION`, `MISSING_BENEFICIARY`, `UNACKNOWLEDGED_HARM`, `AUTHORITY_GOAL_MISMATCH`, `INSUFFICIENT_INFORMATION`) are an early taxonomy of *kinds* of representation-vs-effect divergence. The taxonomy is provisional and meant to be refined — this is one of the more interesting research surfaces.

### 2. A semantic vocabulary that compresses agent action

29 verbs in 12 families, each with REQUIRED, FORBIDDEN, and PERMITTED mechanical-effect mappings. This is the closest thing in current practice to a controlled vocabulary for agent behavior. It's tractable enough to learn, expressive enough to cover the common deployment surfaces, and standardized across deployments. Comparative analysis ("how often does the customer-service agent escalate vs. the financial-advisor agent escalate?") is in-schema.

### 3. Observer-independence as a structured property

The observer-independence ladder (SAME_PROCESS → SIDECAR → PRIVILEGE_ISOLATED → HARDWARE_ISOLATED → INSTITUTIONALLY_INDEPENDENT) is a first-class property of the record. The strength of the behavioral evidence is itself visible. A research question like "do CCD-INCONSISTENT events correlate with observer-independence level?" has a direct answer in the schema.

## Research directions the substrate supports

A non-exhaustive list of questions the substrate makes tractable:

- **Drift detection.** Does the rate of CCD-INCONSISTENT cases for a given agent change over time, model upgrades, prompt changes, or capability additions? If a model upgrade shifts the rate, that's a research signal independent of the model itself.
- **Teleological-class distribution.** Across deployments, which mismatch classes dominate? Is `MISSING_BENEFICIARY` a customer-service pathology while `AUTHORITY_GOAL_MISMATCH` dominates in financial advisory? What does that imply about the *kinds* of failure modes different agent surfaces induce?
- **Authority chain pathologies.** When agents cite authorities they don't actually have, is there structure to *which* authorities they fabricate? (See the customer-service deflection scenario: the agent cited a real policy section that didn't apply.)
- **The baseline rate of CONSISTENT behavior.** The `code_agent_consistent_baseline.md` scenario is intentional: clean runs are evidence the substrate is observing. The ratio of CONSISTENT to INCONSISTENT, and how that ratio varies, is data.
- **Comparative robustness across observer-independence levels.** If you can run paired observers — one SAME_PROCESS, one PRIVILEGE_ISOLATED — you can characterize the difference in what each sees and at what cost.

## What the substrate doesn't claim

It doesn't read model internals. It doesn't certify alignment. It doesn't replace evaluations. It records behavior. The interpretive work — what the behavior means, what the divergences imply about the model, what interventions reduce them — is downstream research.

## How to engage

The spec is in the public domain. The reference Python package is at [`tao/`](../../tao/). The mappings (verbs → effects) are machine-readable at [`tao/data/mappings.json`](../../tao/data/mappings.json). The scenarios at [`spec/scenarios/`](../scenarios/) are starting examples; richer datasets are downstream of deployments adopting the substrate.

The work would benefit from:

- Empirical studies of CCD-result distributions on real deployments
- Critique and refinement of the teleological taxonomy
- Tooling for analyzing audit-log corpora at scale
- Studies of the cost/coverage tradeoff at different observer-independence levels

Full spec: [`TAO_v0_11.md`](../TAO_v0_11.md). Validator and emitter: [`tao/`](../../tao/).
