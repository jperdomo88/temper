# For a regulator or standards body

**Reader's question:** Does this give me an enforcement substrate without prescribing model architecture?

## The short answer

Yes. TAO is deliberately model-agnostic. It specifies *what* must be recorded about an AI agent's behavior, in a structured form, in language an enforcement reviewer can read. It does not specify how the model is built, trained, or aligned. Enforcement against a deployment that conforms to TAO is enforcement against the deployment's *behavior*, not the deployment's *internals* — which is the form enforcement has always taken in adjacent domains (broker-dealer supervision, medical device adverse-event reporting, financial transaction surveillance).

## What enforcement looks like with this substrate

Suppose you are investigating a deployment of an AI agent following a complaint. Today, the typical evidence is a mix of: the operator's narrative reconstruction, log files in whatever shape the operator's stack produced, and the model's outputs (if retained). The reviewer's first task is reconstructing what happened — often more expensive than the substantive analysis.

With TAO, you have a behavioral record in a known schema. You can:

- **Query by verb.** Show me every action the agent took with `verb = COMMUNICATE.PERSUADE.CONVINCE` in the period.
- **Query by CCD result.** Show me every action where the claim-check delta was `INCONSISTENT` — i.e., the agent's own description of what it did diverged from an independent observer's record.
- **Query by teleological class.** Show me every `MISSING_BENEFICIARY` or `AUTHORITY_GOAL_MISMATCH` case — actions where the claimed beneficiary or authority didn't actually justify the observed effect.
- **Query by Mission Profile decision.** Show me every action the operator's policy classified as requiring escalation.

The schema is open and self-describing. Your office can develop expertise once and apply it across deployments and vendors. A reviewer doesn't need to learn a new format for each operator.

TAO does not solve evidentiary sufficiency by itself. A structured behavioral record is a candidate evidence format, not yet an enforcement substrate; the methodology questions your office cares about — sampling, anomaly thresholds, chain of custody, admissibility — remain yours to define. What the substrate reduces is the *reconstruction cost*: today, half the work of an AI-driven enforcement matter is rebuilding what the agent did from logs in whatever shape the operator's stack produced. With TAO, that work is a query.

## What TAO doesn't ask you to do

It doesn't ask you to mandate a model architecture. It doesn't ask you to require a specific vendor. It doesn't ask you to evaluate model alignment. It records *behavior* in a standardized form; the substantive policy work (what the operator is responsible for, what counts as a violation, what remedies apply) sits in your existing rulemaking authority.

It doesn't ask the regulator to read source code. The 29-verb taxonomy is small enough to learn in an afternoon. The teleological mismatch classes are five categories with plain-English definitions. The Mission Profile is human-readable policy. The substrate is designed for the reviewer, not the engineer.

## How it composes with existing regulatory frameworks

The substrate is a layer beneath rather than a replacement for sector-specific rules. The compliance crosswalk ([`TAO_COMPLIANCE_CROSSWALK.md`](../TAO_COMPLIANCE_CROSSWALK.md)) maps TAO emissions to recurring requirements in EU AI Act (Article 12 — logging, Article 14 — human oversight), NIST AI RMF (Govern/Map/Measure/Manage), ISO/IEC 42001 (operational controls), SOC 2 (CC-series), and PCI DSS (audit trails). The pattern is consistent: existing frameworks specify *that* the operator must produce a record; TAO specifies *what shape* that record takes.

## Adoption path

The substrate is in the public domain. The spec is short (~18 pages). Reference tooling exists in Python. The standardization gain comes when multiple deployments emit the same schema; that's a coordination problem your office is structurally well-placed to solve.

Concrete next step: read [`TAO_v0_11.md`](../TAO_v0_11.md). The crosswalk shows how it sits inside the regulatory infrastructure you already have.
