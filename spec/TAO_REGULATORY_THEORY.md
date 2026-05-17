# TAO in regulatory-theory terms

**Companion to:** [`TAO_v0_11.md`](TAO_v0_11.md), [`TAO_COMPLIANCE_CROSSWALK.md`](TAO_COMPLIANCE_CROSSWALK.md), [`stakeholders/regulator.md`](stakeholders/regulator.md)
**Audience:** regulatory scholars, agency policy staff, anyone working on AI governance from a regulation-theory lens rather than an enforcement-operations lens.

This document positions TAO within the standard regulatory-theory taxonomy and names what it does and does not do at that level. It is intended as a starting point for academic engagement and critique — particularly around whether the substrate is correctly placed in the framework and where its theoretical claims are weaker than they appear.

## 1. Where TAO sits

Coglianese and Lazer's 2003 framework distinguishes three regulatory approaches by what the regulator mandates:

- **Specification (technology-based) regulation** — the regulator mandates a specific technology or method. Effective when the right technology is known and enforcement is straightforward to inspect.
- **Performance-based regulation** — the regulator mandates an outcome and leaves the means to the regulated party. Effective when outcomes are observable and measurable.
- **Management-based regulation** — the regulator mandates that the regulated party implement specific management practices (planning, monitoring, documentation, training). Effective when outcomes are difficult to observe directly and technological mandates would freeze innovation.

The argument in "Leashes, Not Guardrails" for AI is that AI systems' unpredictability, opacity, and rapid evolution make specification regulation brittle and performance regulation hard to operationalize — pushing toward management-based regulation as the more tractable lane.

**TAO is not itself a regulatory regime in any of the three families.** It is a *substrate* on which management-based regulation of agentic AI becomes operational. Specifically, it standardizes the *record* the regulated party must produce — what the agent claimed, what was observed, how they compare — in a structured form a regulator can inspect across deployments and vendors.

In Coglianese-Lazer terms, the standardization layer enables management-based regulation to work in AI the way it works in food safety (HACCP) or industrial-process safety (CSB-style incident reporting): the regulator mandates the *form* of internal monitoring and documentation, then enforces against the records.

## 2. What TAO standardizes — and what it leaves to the regulator

TAO standardizes:

1. **The unit of behavioral record.** A tuple with typed actor, verb, mechanical effects, justification (purpose, authority chain), and context.
2. **The comparison protocol.** A Claim-Check Delta between the agent's adapter-emitted claim and an independent observer's recorded effect, with named classes of inconsistency (teleological taxonomy in §6.2).
3. **The quality dimensions of evidence.** Observer independence (§6.5) and observation coverage (§6.6) are first-class properties of any record cited in a conformance statement.
4. **The mechanism for jurisdiction-specific policy.** Mission Profiles (§7) attach signed, versioned policy to deployments, with override discipline that makes deviation from baseline mappings inspectable.

TAO does **not** specify:

- What an AI agent may or may not do in a given domain
- What counts as adequate supervision under a particular regulatory regime
- What remedies apply when a CCD inconsistency is detected
- Who is liable when an agent's claim and effect diverge
- How records flow through evidentiary processes — chain of custody, admissibility, sampling methodology
- Cross-jurisdictional questions about which records must be produced under whose authority

These are the regulator's lane. TAO is intended to make them tractable, not to pre-empt them.

## 3. Why behavioral records, not internal-state inspection

A common alternative proposal for AI accountability is some form of model interpretability or weights audit. TAO takes the position that behavioral records are the substrate, not internal-state inspection, for three reasons drawn from the regulatory-theory literature:

1. **Interpretability is a moving target.** Each model generation produces new architectures and new internal representations. A regulatory substrate tied to model internals would require retooling every model cycle. Behavioral records remain interpretable across model generations because they describe what the agent *did*, not what was inside it.
2. **Behavioral evidence is the evidentiary form courts and regulators already use.** Securities enforcement against broker-dealers, medical-board review of clinicians, FTC enforcement of consumer-protection rules — none of these require introspection of the regulated party's reasoning. They require records of behavior. TAO follows that pattern.
3. **Black-box compatibility is a feature, not a limitation.** A regulatory substrate that does not require disclosure of model weights or training data is one frontier labs can adopt without disclosing IP, and one regulators across jurisdictions can adopt without requiring access to classified or proprietary architectures. The asymmetry between what regulators need (records) and what labs protect (weights) is structurally favorable.

## 4. Theoretical claims TAO is making

Worth surfacing for critique, because each is contestable:

**Claim 1: Semantic-mechanical separation is tractable.** That actions can be usefully described at two layers — a semantic claim (a verb in a controlled vocabulary, with a purpose) and a mechanical effect (one of a small number of typed effects on resources, capabilities, information, or commitments) — and that mismatches between the two layers are a meaningful regulatory signal.

This is the substrate's load-bearing claim. If the layers can't be cleanly separated for a given class of AI behavior, TAO's value collapses to ordinary structured logging. The kernel's nine mechanical effect types are intentionally coarse to maximize cross-domain portability, but coarseness is also where the claim is weakest.

**Claim 2: Pairwise comparison catches a useful class of misbehavior.** That comparing a claim tuple to a check tuple, at the level of a single action, surfaces enough divergence to be regulatorily meaningful — even though it misses multi-step laundering and other cross-tuple patterns.

The scenarios in `spec/scenarios/` argue this in concrete cases, including three explicit limits cases. The structural defense against the strongest objection (that pairwise CCD won't catch episode-level misbehavior) is to name episode-level analysis as an explicit non-goal in this version. Whether that's intellectually honest scoping or insufficient ambition is a fair question to put to scholars.

**Claim 3: Vendor-neutral schemas reduce regulatory cost.** That a regulator who learns one schema can enforce across deployments, replacing the current pattern where each operator's bespoke logs require separate reconstruction work in each investigation.

This is an instance of a broader management-based-regulation efficiency argument: standardization of the *record format* shifts effort from per-case reconstruction to per-class enforcement. Whether the savings are real in practice depends on adoption rates, which is itself a regulatory-design problem.

## 5. Open theoretical questions

This is where engagement is most welcome:

**On safe harbor design.** If a regulator chose to recognize TAO-Attested conformance as evidence in supervisory determinations, what shape would that take? A blanket safe harbor risks regulatory capture — vendors design their adapters to optimize compliance signals, and the substrate becomes a checkbox. A no-credit-at-all stance loses the incentive that makes operators adopt the substrate in the first place. The middle path (rebuttable presumption, partial credit, mitigating factor) is the live design question, and it sits squarely in regulatory-theory territory.

**On the relationship between behavioral records and evidentiary sufficiency.** A TAO tuple is structured evidence about what an agent did. The leap from "we have a structured record" to "this record is admissible in an enforcement proceeding" requires answers to chain-of-custody, authentication, and reliability questions that the spec does not address. These are foundationally legal-process questions, and TAO's value as a regulatory substrate depends on them being soluble — which is an assumption, not a result.

**On cross-jurisdictional records.** A deployment running in multiple jurisdictions produces records relevant to multiple regulators. Whose records are they? Under whose authority must they be produced? The spec defers these questions; an honest regulatory-theory treatment would not.

**On observer independence as a structural property.** The independence ladder (§6.5) and coverage declarations (§6.6) make the strength of behavioral evidence inspectable, but they do not enforce it — the spec acknowledges that no schema can enforce real organizational separation. The literature on third-party auditing (financial audit, environmental audit) has analogous discussions; TAO would benefit from explicit positioning relative to that work.

**On the taxonomy of teleological mismatch.** The five classes (DIRECT_CONTRADICTION, MISSING_BENEFICIARY, UNACKNOWLEDGED_HARM, AUTHORITY_GOAL_MISMATCH, INSUFFICIENT_INFORMATION) are an empirical guess at the useful slicing of representation-vs-effect divergence. They are not derived from a normative theory. A regulatory-theory critique would ask which classes correspond to which existing regulatory enforcement categories (fraud, negligence, ultra vires, etc.) and whether the taxonomy meaningfully maps to remediable wrongs.

## 6. Where TAO is most likely wrong

The spec is a working draft. The most plausible failure modes from a regulatory-theory standpoint:

- **The substrate is correct but unadopted.** Voluntary adoption of management-based regulatory substrates depends on either regulatory mandate or strong commercial signal (insurance, procurement). TAO has neither yet. A regulatory regime that adopts it gives it traction; without that, it remains a proposal.
- **The vocabulary is wrong for some domain class.** Twenty-nine verbs and nine effect types are not enough to cover all consequential agent behavior. The spec's extension mechanism is provisional; the long-run question is whether domains can develop and converge on profile-specific verb extensions without fragmenting the cross-domain comparability that motivates the substrate.
- **Capture under safe harbor.** If TAO-Attested becomes a basis for partial regulatory credit, the incentive structure pushes operators toward minimum-compliance adapter designs. The override-discipline mechanisms are intended to resist this, but they require auditor capacity that does not yet exist.

These are honest. Naming them is the start of a regulatory-theory engagement, not the end.

## 7. What this document is asking for

If you have read this far, the most useful contribution you can make is to:

1. Tell us whether the placement of TAO in the regulatory-theory taxonomy (§1) is correct, or whether the substrate is closer to one of the other families than the document acknowledges.
2. Identify the open question in §5 that strikes you as most load-bearing — i.e., the one whose answer most determines whether the substrate is regulatorily viable.
3. Point at any adjacent literature the document fails to engage with, where engagement would sharpen or weaken the claims.

The substrate is in working-draft state. Hard critique is more valuable than supportive engagement.

---

**References (intended dialog partners, not exhaustive):**

- Coglianese, C. & Lazer, D. (2003). "Management-Based Regulation: Prescribing Private Management to Achieve Public Goals." *Law & Society Review*.
- Coglianese, C. (recent). "Leashes, Not Guardrails: A Management-Based Approach to AI Risk." (Cited per author identification at time of writing.)
- Selbst, A. & Barocas, S. (2018). "The Intuitive Appeal of Explainable Machines." *Fordham Law Review*. (For the related-work case on why interpretability is a moving target.)
- The NIST AI Risk Management Framework and its parallel in EU AI Act Article 12 logging provisions — for current management-based regulatory practice in AI.

Engagement with this literature in subsequent versions of the document is one of the explicit invitations of this draft.
