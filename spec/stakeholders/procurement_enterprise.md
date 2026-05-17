# For enterprise procurement and risk

**Reader's question:** What do I require in the contract?

## The short answer

Require the vendor's agent to emit TAO-conformant tuples to a sink your team controls, with an attested observer-independence level you can reason about. The substrate gives you a contractual hook that doesn't depend on the vendor's internal model choices — and a behavioral record you can audit without their cooperation.

## Suggested contract clauses

The following are starting points, not legal advice. Engage counsel to adapt them to your standard agreement.

### 1. Logging conformance

> "Covered Action" means any tool call, external communication, data access, resource transfer, commitment creation, account modification, message send, or other action class enumerated in **Exhibit A — Covered Action Surface**.
>
> Provider shall emit, for every Covered Action taken by the AI agent in the course of the Service, a record conforming to the TAO v0.12 tuple schema (or a successor version mutually agreed). Records shall be delivered to a logging endpoint designated by Customer, in near-real-time, no later than 60 seconds after the Covered Action.

This makes the audit trail Customer-controlled, not Provider-controlled. The record is in a schema the regulator can read directly. The Covered Action surface is negotiated explicitly rather than left as "every action" — without that scoping, vendors will either reject the clause or comply by flooding logs with no-ops.

### 2. Observer independence AND coverage

> Provider shall declare (a) the observer-independence level of the recording mechanism (one of SAME_PROCESS, SIDECAR, PRIVILEGE_ISOLATED, HARDWARE_ISOLATED, INSTITUTIONALLY_INDEPENDENT) and (b) the observer's coverage declaration per TAO v0.11 §6.6, naming the channels observed, the channels explicitly not observed, the observation point, and whether blocked attempts are visible. Provider shall not downgrade either property without 30 days written notice.

Independence and coverage are distinct properties. Independence is where the observer runs; coverage is what it can see. A `PRIVILEGE_ISOLATED` observer that only watches wrapper-level tool returns is technically isolated and substantively uninformative. Forcing both declarations prevents the procurement-gaming pattern where a vendor satisfies the independence clause while quietly excluding the channels that matter.

### 3. Claim-check delta (where the agent's surface is wide)

> Where Provider's agent has discretion over actions with effects beyond an immutable interface (e.g., external network calls, payments, account modifications, communication to third parties), Provider shall additionally emit Check tuples from an independent observer, and a CCD record evaluating consistency between Claim and Check.

This applies where the cost of an unrecorded effect is high. The browser-agent scenario ([`browser_agent_subscription.md`](../scenarios/browser_agent_subscription.md)) is the canonical example: a payment agent that fuzzy-matches a subscription opt-in needs an observer that catches the `COMMITMENT.MAKE` side-effect the agent didn't claim.

### 4. Mission Profile review

> Provider shall provide Customer with a copy of any Mission Profile the agent operates under (including verb-to-decision mappings, escalation policies, and any deviations from the published Mission Profile schema's defaults). Material changes to the Mission Profile require Customer review.

The Mission Profile is the operator's policy. You should see it the way you would see any policy document the vendor operates under.

### 5. Audit access

> Customer shall have read-only access to the audit log for the term of the agreement plus a retention period of [N] years. Provider shall not redact records of actions affecting Customer or Customer's data without notice.

The substrate is only as useful as your right to read it.

## What you get out of this

Three things, ordered by how soon you'll need them:

1. **A vendor-neutral audit substrate.** When you have agents from three vendors, you have one log format. Your team doesn't relearn each vendor's bespoke trace format.
2. **A breach-investigation surface that doesn't require vendor cooperation.** When something goes wrong, you have the structured record. You don't have to ask the vendor to reconstruct.
3. **A control your compliance team can map to existing frameworks.** The crosswalk ([`TAO_COMPLIANCE_CROSSWALK.md`](../TAO_COMPLIANCE_CROSSWALK.md)) shows how TAO emissions satisfy logging and oversight requirements in EU AI Act, NIST AI RMF, ISO 42001, SOC 2, and PCI DSS — frameworks you're already operating under.

## What it doesn't replace

This isn't a substitute for vendor due diligence, model evaluation, or your standard security review. It's the *behavioral* logging layer that sits on top. Treat it like access logs: necessary, not sufficient.

Full spec: [`TAO_v0_11.md`](../TAO_v0_11.md). Crosswalk: [`TAO_COMPLIANCE_CROSSWALK.md`](../TAO_COMPLIANCE_CROSSWALK.md). Enterprise scenario: [`scenarios/enterprise_wrong_channel.md`](../scenarios/enterprise_wrong_channel.md).
