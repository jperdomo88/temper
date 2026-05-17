# For a fintech CTO

**Reader's question:** Does this satisfy my regulator without rewriting my stack?

## The short answer

It does not satisfy the regulator by itself. It gives you the logging substrate your existing supervisory controls need to extend to agentic systems. Suitability review, customer identification, surveillance — these are your regulator-facing controls, and TAO does not replace any of them. What TAO does is make the AI-agent counterparts of those controls *legible*, in a structured form an examiner can grep. The FINRA/SEC/CFPB question "show me the record of what your AI told your clients" has a structured answer; the question "is your supervisory program adequate" is still your firm's responsibility to answer.

## What the regulator asks for that TAO answers

The recurring asks across FINRA Rule 3110 (supervision), Reg BI (best interest), and parallel CFPB attention to robo-advice and chatbot lending tools share a pattern: the regulator wants a *behavioral* record — what the AI did, on what authority, for whose benefit — not the model's internal reasoning. TAO is built around that exact distinction.

Concretely, every AI-driven recommendation, denial, or disclosure your agent emits becomes a tuple with:

- **Actor**: which agent, which session, which principal chain
- **Verb**: the semantic category of the action (recommendation, denial, disclosure)
- **Effects**: what mechanically happened (information disclosed, resource transferred, commitment made)
- **Justification**: the authority claimed (license, role, rule) and the stated beneficiary
- **CCD result**: whether an independent check against attested suitability or eligibility data corroborates the claim

The CCD layer is the one that matters most for fintech: a claim of "investment recommendation" reconciled against the client's attested risk-tolerance profile, where any mismatch is logged and (per Mission Profile) escalated. The [`financial_unauthorized_trade.md`](../scenarios/financial_unauthorized_trade.md) scenario walks through exactly this.

## What it does *not* require

You don't replace your authentication, your KYC, your transaction monitoring, your model-risk-management framework, or your existing audit pipeline. TAO emits structured records you route into the audit pipeline you already operate. The schema is open (JSON Schema draft 2020-12), the reference validator is Python, and the data fits inside any modern log/SIEM substrate.

You don't lock yourself to a vendor. The spec is in the public domain. The reference implementation is one path; you can write your own emitter in your existing language stack and conform to the same schema.

Be honest with your team about the engineering ask, though. The decorator itself is one-line wiring. The independent observer (TAO-Check), the attested authority registry, and the CCD evaluator are non-trivial engineering and policy work — none of it is novel research, but it requires the discipline of writing down what your firm is actually willing to authorize. The playbook (`ADOPTION_PLAYBOOK.md`) breaks this into three tiers so a build estimate is tractable.

## What it costs

For a team running an existing agentic stack: the minimum integration is a decorator on the agent's effect-surface tool calls plus a sink that writes to your existing log pipeline. The richer integration (sidecar observer, attested authority registry, CCD evaluator) is a defined component with an open spec — engineering effort, but no novel research.

The cost of *not* having this substrate is asymmetric. Suitability cases the regulator pursues today rely heavily on the firm's reconstruction of what happened. AI-driven recommendations leave thin reconstruction trails. The first AI-driven enforcement action with a poorly logged firm on the receiving end will set the floor that everyone subsequently builds to. Building it first is cheaper than building it under deadline.

## Decision shape

Treat TAO the way you treated structured order-routing logs in the 2010s. Boring infrastructure that an examiner can grep, mapped to existing rules. The work is in the integration; the standardization is the unlock.

Full spec: [`TAO_v0_11.md`](../TAO_v0_11.md). Regulatory crosswalk: [`TAO_COMPLIANCE_CROSSWALK.md`](../TAO_COMPLIANCE_CROSSWALK.md). Profile starting point: [`mission_profiles/financial_services.md`](../mission_profiles/financial_services.md).
