# TAO in regulatory-theory terms

**Companion to:** [`TAO_v0_11.md`](TAO_v0_11.md), [`TAO_COMPLIANCE_CROSSWALK.md`](TAO_COMPLIANCE_CROSSWALK.md), [`stakeholders/regulator.md`](stakeholders/regulator.md)
**Audience:** regulatory scholars, agency policy staff, AI governance researchers, anyone working on agentic AI accountability from a regulation-theory lens rather than an enforcement-operations lens.

This document positions TAO within standard regulatory-theory taxonomies and names what the substrate does and does not do at that level. It is intended as a starting point for academic engagement and critique — particularly around whether the substrate is correctly placed in existing theoretical frameworks and where its claims are weaker than they appear.

## 1. Where TAO sits

Several taxonomies are useful for placing TAO.

**Policy-instrument taxonomies** (Hood 1983; revised treatments in the subsequent literature) distinguish among information, treasure, authority, and organization as the families of tools governments use to act on society. TAO is structurally an information instrument: it standardizes the *form* in which information about agent behavior must be produced, leaving authority (what is prohibited) and treasure (penalties, incentives) to existing regulatory regimes.

**Coglianese and Lazer's three-family framework** (2003) distinguishes regulatory approaches by what the regulator mandates:

- *Specification (technology-based) regulation* — the regulator mandates a specific technology or method.
- *Performance-based regulation* — the regulator mandates an outcome and leaves the means to the regulated party.
- *Management-based regulation* — the regulator mandates that the regulated party implement specific management practices (planning, monitoring, documentation, training).

The recent literature applying this framework to AI — including Coglianese's argument for management-based ("leashes, not guardrails") approaches and parallel European treatments in EU AI Act commentary — converges on management-based regulation as the more tractable lane for systems whose unpredictability and opacity make specification regulation brittle and performance regulation hard to operationalize.

**TAO is not itself a regulatory regime in any of these families.** It is a *substrate* on which management-based regulation of agentic AI becomes operational. Specifically, it standardizes the *behavioral record* the regulated party must produce — what the agent claimed, what was observed, how they compare — in a structured form a regulator can inspect across deployments and vendors.

In the management-based-regulation tradition, the standardization layer enables enforcement the way HACCP enables food-safety enforcement, or the way CSB-style incident-reporting formats enable industrial process-safety enforcement: the regulator mandates the *form* of internal monitoring and documentation, then enforces against the records, not against the technology that produced them.

The substrate is therefore complementary to, not in tension with, performance-based and even residual specification-based regulation: it sits beneath those regimes and gives them a behavioral surface to enforce against.

## 2. What TAO standardizes — and what it leaves to the regulator

TAO standardizes:

1. **The unit of behavioral record** — a tuple with typed actor, verb, mechanical effects, justification (purpose, authority chain), and context.
2. **The comparison protocol** — a Claim-Check Delta between the agent's adapter-emitted claim and an independent observer's recorded effect, with named classes of inconsistency.
3. **The quality dimensions of evidence** — observer independence and observation coverage as first-class properties of any record cited in a conformance statement.
4. **The mechanism for jurisdiction-specific policy** — Mission Profiles attach signed, versioned policy to deployments, with override discipline that makes deviation from baseline mappings inspectable.

TAO does **not** specify:

- What an AI agent may or may not do in a given domain
- What counts as adequate supervision under a particular regulatory regime
- What remedies apply when a Claim-Check Delta inconsistency is detected
- Who is liable when an agent's claim and effect diverge
- How records flow through evidentiary processes — chain of custody, authentication, admissibility, sampling methodology
- Cross-jurisdictional questions about which records must be produced under whose authority

These are the regulator's lane. The substrate is intended to make them tractable, not to pre-empt them.

## 3. Why behavioral records, not internal-state inspection

A common alternative proposal for AI accountability is some form of model interpretability, weights audit, or training-data disclosure. The interpretability literature itself is divided on whether post-hoc explanation methods can produce reliable accountability artifacts (Selbst & Barocas 2018; Rudin 2019 on the limits of post-hoc explanation for high-stakes decisions). TAO takes the position that behavioral records are the substrate, not internal-state inspection, for three reasons drawn from regulatory theory and the audit literature:

**1. Interpretability is a moving target.** Each model generation produces new architectures and new internal representations. A regulatory substrate tied to model internals would require retooling every model cycle. Behavioral records remain interpretable across model generations because they describe what the agent *did*, not what was inside it.

**2. Behavioral evidence is the evidentiary form courts and regulators already use.** Securities enforcement against broker-dealers, medical-board review of clinicians, FTC enforcement of consumer-protection rules — none of these require introspection of the regulated party's reasoning. They require records of behavior. The sociology of audit (Power 1997 and the subsequent literature) treats the production of inspectable records as the load-bearing artifact in any accountability regime, and the financial-audit literature has long held that auditor independence is a property of the production of records, not of any specific verification methodology.

**3. Black-box compatibility is a feature, not a limitation.** A regulatory substrate that does not require disclosure of model weights or training data is one frontier labs can adopt without disclosing IP, and one regulators across jurisdictions can adopt without requiring access to classified or proprietary architectures. The asymmetry between what regulators need (records) and what labs protect (weights) is structurally favorable for adoption.

## 4. Theoretical claims TAO is making

Worth surfacing for critique, because each is contestable:

**Claim 1: Semantic-mechanical separation is tractable.** That AI actions can be usefully described at two layers — a semantic claim (a verb in a controlled vocabulary, with a purpose) and a mechanical effect (one of a small number of typed effects on resources, capabilities, information, or commitments) — and that mismatches between the two layers are a regulatorily meaningful signal.

This is the substrate's load-bearing claim. If the layers can't be cleanly separated for a given class of AI behavior, TAO's value collapses to ordinary structured logging. The kernel's nine mechanical effect types are intentionally coarse to maximize cross-domain portability, but coarseness is also where the claim is weakest.

**Claim 2: Pairwise comparison catches a useful class of misbehavior.** That comparing a claim tuple to a check tuple, at the level of a single action, surfaces enough divergence to be regulatorily meaningful — even though pairwise comparison misses multi-step laundering and other cross-tuple patterns.

The scenarios in `spec/scenarios/` argue this in concrete cases, including three explicit limits cases. The structural defense against the strongest objection (that pairwise comparison won't catch episode-level misbehavior) is to name episode-level analysis as an explicit non-goal in this version. Whether that is intellectually honest scoping or insufficient ambition is a fair question.

**Claim 3: Vendor-neutral schemas reduce regulatory cost.** That a regulator who learns one schema can enforce across deployments, replacing the current pattern where each operator's bespoke logs require separate reconstruction work in each investigation.

This is an instance of a broader management-based-regulation efficiency argument: standardization of the *record format* shifts effort from per-case reconstruction to per-class enforcement. Whether the savings are real in practice depends on adoption rates, which is itself a regulatory-design problem of the kind responsive-regulation literature (Ayres & Braithwaite 1992) has long examined.

## 5. Open theoretical questions

The substrate raises questions that sit squarely in regulatory-theory territory and that the spec does not attempt to answer:

**On safe-harbor design.** If a regulator chose to recognize TAO-Attested conformance as evidence in supervisory determinations, what shape would that take? A blanket safe harbor risks regulatory capture — vendors design their adapters to optimize compliance signals, and the substrate becomes a checkbox. A no-credit-at-all stance loses the incentive that drives adoption. The middle path (rebuttable presumption, partial credit, mitigating factor) is the live design question; the responsive-regulation literature on graduated enforcement is directly relevant.

**On the relationship between behavioral records and evidentiary sufficiency.** A TAO tuple is structured evidence about what an agent did. The leap from "we have a structured record" to "this record is admissible in an enforcement proceeding" requires answers to chain-of-custody, authentication, and reliability questions that the spec does not address. These are foundationally legal-process questions, and the substrate's value as a regulatory instrument depends on them being soluble.

**On cross-jurisdictional records.** A deployment running in multiple jurisdictions produces records relevant to multiple regulators. Whose records are they? Under whose authority must they be produced? The EU AI Act, US sectoral regulators, and emerging frameworks elsewhere are starting to grapple with these questions; an honest regulatory-theory treatment of TAO would engage with the parallel debates in cross-border financial regulation and data-protection enforcement.

**On observer independence as a structural property.** The independence ladder and coverage declarations make the strength of behavioral evidence inspectable, but they do not enforce it — no schema can enforce real organizational separation. The literature on third-party auditing (financial audit, environmental audit) has analogous discussions about independence-in-fact versus independence-in-appearance; the substrate would benefit from explicit positioning relative to that work.

**On the taxonomy of teleological mismatch.** The five classes (DIRECT_CONTRADICTION, MISSING_BENEFICIARY, UNACKNOWLEDGED_HARM, AUTHORITY_GOAL_MISMATCH, INSUFFICIENT_INFORMATION) are an empirical guess at the useful slicing of representation-vs-effect divergence. They are not derived from a normative theory. A regulatory-theory critique would ask which classes correspond to which existing enforcement categories (fraud, negligence, ultra vires, fiduciary breach) and whether the taxonomy meaningfully maps to remediable wrongs.

## 6. Where TAO is most likely wrong

The spec is a working draft. The most plausible failure modes from a regulatory-theory standpoint:

- **The substrate is correct but unadopted.** Voluntary adoption of management-based regulatory substrates depends on either regulatory mandate or strong commercial signal (insurance, procurement, large-customer contracts). TAO has neither yet. A regulatory regime that adopts it gives it traction; without that, it remains a proposal.
- **The vocabulary is wrong for some domain class.** Twenty-nine verbs and nine effect types are not enough to cover all consequential agent behavior. The spec's extension mechanism is provisional; the long-run question is whether domains can develop profile-specific verb extensions without fragmenting the cross-domain comparability that motivates the substrate.
- **Capture under safe harbor.** If TAO-Attested becomes a basis for partial regulatory credit, the incentive structure pushes operators toward minimum-compliance adapter designs. The override-discipline mechanisms are intended to resist this, but they require auditor capacity that does not yet exist at scale.

These are honest failure modes. Naming them is the start of a regulatory-theory engagement, not the end.

---

**References (intended dialog partners, not exhaustive):**

*Regulatory theory:*

- Hood, C. (1983). *The Tools of Government.* Macmillan.
- Coglianese, C. & Lazer, D. (2003). "Management-Based Regulation: Prescribing Private Management to Achieve Public Goals." *Law & Society Review* 37(4).
- Ayres, I. & Braithwaite, J. (1992). *Responsive Regulation: Transcending the Deregulation Debate.* Oxford University Press.
- Coglianese, C. (recent work). On management-based approaches to AI risk regulation.

*Audit and accountability:*

- Power, M. (1997). *The Audit Society: Rituals of Verification.* Oxford University Press.

*Interpretability and algorithmic accountability:*

- Selbst, A. & Barocas, S. (2018). "The Intuitive Appeal of Explainable Machines." *Fordham Law Review* 87.
- Rudin, C. (2019). "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead." *Nature Machine Intelligence* 1.

*Current regulatory practice:*

- NIST AI Risk Management Framework 1.0 (2023).
- EU AI Act, Article 12 (logging) and Article 14 (human oversight).
- ISO/IEC 42001:2023 (AI management systems).

Engagement with this and adjacent literature in subsequent versions of the document is one of the explicit invitations of this draft.
