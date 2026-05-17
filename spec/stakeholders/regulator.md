# For a regulator or standards body

**Reader's question:** Does this give me an enforcement substrate without prescribing model architecture?

## The short answer

It gives you a standardized behavioral record that can support enforcement without prescribing model architecture. TAO is deliberately model-agnostic. It specifies *what* must be recorded about an AI agent's behavior, in a structured form, in language an enforcement reviewer can read. It does not specify how the model is built, trained, or aligned, and it does not by itself carry statutory authority, evidentiary admissibility, or sanction power — those remain in your existing rulemaking and enforcement frameworks. What changes is that enforcement against a TAO-conformant deployment becomes enforcement against the deployment's *behavior*, not the deployment's *internals* — which is the form enforcement has always taken in adjacent domains (broker-dealer supervision, medical device adverse-event reporting, financial transaction surveillance).

## What enforcement looks like with this substrate

Suppose you are investigating a deployment of an AI agent following a complaint. Today, the typical evidence is a mix of: the operator's narrative reconstruction, log files in whatever shape the operator's stack produced, and the model's outputs (if retained). The reviewer's first task is reconstructing what happened — often more expensive than the substantive analysis.

With TAO, you have a behavioral record in a known schema. You can:

- **Query by verb.** Show me every action the agent took with `verb = COMMUNICATE.PERSUADE.CONVINCE` in the period.
- **Query by CCD result.** Show me every action where the claim-check delta was `INCONSISTENT` — i.e., the agent's own description of what it did diverged from an independent observer's record.
- **Query by teleological class.** Show me every `MISSING_BENEFICIARY` or `AUTHORITY_GOAL_MISMATCH` case — actions where the claimed beneficiary or authority didn't actually justify the observed effect.
- **Query by Mission Profile decision.** Show me every action the operator's policy classified as requiring escalation.

The schema is open and self-describing. Your office can develop expertise once and apply it across deployments and vendors. The 29-verb taxonomy and nine effect types are small enough for reviewer training and consistent supervisory use. A reviewer doesn't need to learn a new format for each operator.

TAO does not solve evidentiary sufficiency by itself. A structured behavioral record is a candidate evidence format, not yet an enforcement substrate; the methodology questions your office cares about — sampling, anomaly thresholds, chain of custody, admissibility — remain yours to define. What the substrate reduces is the *reconstruction cost*: today, half the work of an AI-driven enforcement matter is rebuilding what the agent did from logs in whatever shape the operator's stack produced. With conformant TAO emissions, part of the reconstruction work becomes queryable — *part*, because that depends on the operator having emitted useful tuples, observer coverage having been adequate, context having been attested rather than self-reported, retention having been honored, and chain-of-custody being acceptable for the use the records are put to. None of these are technical guarantees; they are deployment commitments your existing rulemaking authority is the right place to require.

## A second-order point: cross-jurisdictional coordination

A shared substrate enables something stronger than per-jurisdiction enforcement. The Chemical Weapons Convention works not because every nation agrees on military doctrine, but because everyone agrees that nerve gas is categorically unacceptable — the molecular structure itself is banned. You don't need consensus on conventional use of force to get consensus on sarin.

TAO enables analogous coordination for AI. Some action patterns are likely controversial (when is `HARM.DAMAGE.STRIKE` against military targets justified?). Others may be candidates for universal prohibition regardless of broader geopolitical disagreement — patterns like unauthorized access to catastrophic systems, human-autonomy capture via preference manipulation, or unauthorized recursive self-improvement at scale. Bannable by mutual interest, even when conventional rules remain in dispute.

The substrate makes the distinction between hard and easy disagreements operational. A regulator working with international counterparts has vocabulary precise enough to write a treaty that bans a structural pattern rather than a narrative. Compliance is verifiable from logs and attested effects. Violations are detectable even when models are black-box, because governance attaches at the interface rather than the internals.

The substrate does not solve the diplomacy. It removes the prior question of whether the records being compared are even compatible.

## What TAO doesn't ask you to do

It doesn't ask you to mandate a model architecture. It doesn't ask you to require a specific vendor. It doesn't ask you to evaluate model alignment. It records *behavior* in a standardized form; the substantive policy work (what the operator is responsible for, what counts as a violation, what remedies apply) sits in your existing rulemaking authority.

It doesn't ask the regulator to read source code. The 29-verb taxonomy is small enough to learn in an afternoon. The teleological mismatch classes are five categories with plain-English definitions. The Mission Profile is human-readable policy. The substrate is designed for the reviewer, not the engineer.

## How it composes with existing regulatory frameworks

The substrate is a layer beneath rather than a replacement for sector-specific rules. The compliance crosswalk ([`TAO_COMPLIANCE_CROSSWALK.md`](../TAO_COMPLIANCE_CROSSWALK.md)) maps TAO emissions to recurring requirements in EU AI Act (Article 12 — logging, Article 14 — human oversight), NIST AI RMF (Govern/Map/Measure/Manage), ISO/IEC 42001 (operational controls), SOC 2 (CC-series), and PCI DSS (audit trails). The pattern is consistent: existing frameworks specify *that* the operator must produce a record; TAO specifies *what shape* that record takes.

## Adoption path

The substrate is in the public domain. The spec is short (~18 pages). Reference tooling exists in Python. The standardization gain comes when multiple deployments emit the same schema; that's a coordination problem your office is structurally well-placed to solve.

Concrete next step: read [`TAO_v0_11.md`](../TAO_v0_11.md). The crosswalk shows how it sits inside the regulatory infrastructure you already have.
