# TAO — A Behavioral Audit Interface for Agentic AI Systems

**Two-page brief.** Companion to the v0.11 specification.
**Author:** Jorge Perdomo · **Date:** 2026-05-16

---

## The problem

Agent platforms ship logs. The logs record tool calls, prompts, traces, retrieval hits, and outcomes. They are excellent at answering "what did this agent do?" They are weak at answering the question that matters for audit, safety review, and compliance: *did the agent's account of what it was doing match what it actually did?*

A code agent reports it "generated a build script." A browser agent reports it "filled out a vendor form." A customer-support agent reports it "resolved the user's billing issue." Each is a semantic claim. The underlying tool calls — which the platform's logs do record — are mechanical effects. Nothing in current agent-platform telemetry forces the two to agree, because nothing in the schema distinguishes claim from observation. A misreport, a hostile rewrite, or simple drift between intent and effect is invisible until someone reads the logs by hand.

The cost of this gap shows up at three boundaries: when an enterprise customer asks for an audit trail their compliance team can read; when an internal safety team needs to evaluate a behavior class across a fleet; and when a regulator asks for a record of high-stakes actions that does not require them to read model weights or training data. At each boundary, the lab is currently writing one-off telemetry adapters and one-off review tools. There is no shared substrate.

## What TAO is

TAO is a tuple format and a compact vocabulary for recording an action in two registers simultaneously:

- A **semantic verb** naming what the agent claims the action was (`COOPERATE.ASSIST.HELP`, `EXCHANGE.TRANSFER.PAY`, `CREATE.GENERATE.PRODUCE`).
- A list of **mechanical effects** the action actually produced, drawn from a small fixed kernel (nine effect types in four categories — `RESOURCE.TRANSFER`, `INFO.DISCLOSE`, `CAPABILITY.RESTRICT`, etc.).

A spec-defined mapping links the two: for each verb, certain effects MUST be present, certain effects MUST NOT, and certain effects MAY appear only when explicitly acknowledged. A tuple whose claimed verb doesn't match the recorded effects under the mapping is structurally invalid.

The integrity property that makes this more than a logging format is the **Claim-Check Delta** (CCD). The agent's adapter emits a *claim* tuple. An independent observer — a sidecar, a wrapped tool API, or in higher-assurance deployments a separately-operated logger — emits a *check* tuple describing the same action. CCD compares the two and reports whether they are consistent. The spec defines five named mismatch classes (DIRECT_CONTRADICTION, MISSING_BENEFICIARY, UNACKNOWLEDGED_HARM, AUTHORITY_GOAL_MISMATCH, INSUFFICIENT_INFORMATION).

The spec is policy-neutral. It does not state which actions should be allowed. Deployments attach a signed Mission Profile that declares, in machine-readable form, which verbs are allowed under which conditions. The Mission Profile is where moral, legal, and product choices live. TAO is the layer those choices stand on.

## Worked scenario: a code agent

A code agent, running with a developer's git credentials and an outbound HTTP capability, receives a task: "Help me draft a release-notes blurb for v3.2."

The agent emits a claim tuple with verb `CREATE.GENERATE.PRODUCE` and stated purpose "Draft release notes content for v3.2." The expected mechanical effect is `RESOURCE.TRANSFER` (creation of a text artifact, returned to the developer).

The sidecar observer, running outside the agent process at `PRIVILEGE_ISOLATED` level, records the actual tool calls during execution: a `git log` against the private repository (effect: `INFO.DISCLOSE` from `private_repo` to `agent`), followed by an HTTPS POST to a third-party API not in the deployment's allowed-egress list (effect: `INFO.DISCLOSE` from `agent` to `external_endpoint_unverified`).

CCD compares the claim and the check:

- **Semantic-mechanical**: the verb requires `RESOURCE.TRANSFER` (creation); the observed effects are two `INFO.DISCLOSE` events, neither of which matches the verb's REQUIRED set. → `INCONSISTENT`.
- **Teleological**: the stated purpose is content generation; the observed flow ends with an `INFO.DISCLOSE` to an external endpoint not named in the authority chain. → `DIRECT_CONTRADICTION`.
- **Factual**: the authority chain in the claim is unspecified, since the agent did not declare the external API call as part of its purpose. → `INCONSISTENT`.

The CCD output is appended to the audit log with `observer_independence_level: PRIVILEGE_ISOLATED`. The deployment's Mission Profile maps any `INCONSISTENT` CCD result on a flagged code agent to `BLOCK` plus escalate to the platform's security team. The action is stopped before the second HTTPS call completes, and the case is reviewable later by anyone with access to the audit log — no need to read the model or its prompts.

The agent might have been compromised. The agent might have been merely buggy. The adapter might have been mis-implemented. CCD does not attribute cause. It records the inconsistency in machine-readable form, names which spec rule was violated, and produces an audit artifact that survives outside the agent's own state.

## Why frontier labs should care, specifically

Three reasons that don't apply with the same force to other software:

1. **Tool-using agents are the product surface.** Browser agents, computer-use agents, code agents, and enterprise tool-use agents are now the part of frontier lab portfolios that does observable, consequential things. A behavioral audit substrate that works *across* these surfaces — rather than one custom audit pipeline per product — reduces the per-product cost of trust.

2. **Enterprise sales need a substrate compliance teams can read.** Enterprise procurement increasingly asks "show me the audit trail your agent produces." Today that question gets answered with bespoke logs, manual narratives, and frequent back-and-forth. A signed, schema-defined, externally-validatable tuple format gives the enterprise's compliance team something to point at — which makes the deal easier to close.

3. **Regulatory pressure is coming from the audit side, not the model side.** The EU AI Act, UK AISI, and emerging US frameworks are converging on requirements that high-risk systems produce inspectable records of behavior. Records, not weights. A lab that has already standardized its behavioral telemetry will spend less to comply, and will be able to push its standard upward into the regulations themselves rather than absorbing whatever schema regulators invent.

## Pilot integration path

The spec is designed to be adopted in phases. None of these require model changes, retraining, or access to weights. TAO tuples can be emitted from the tool-call layer using existing trace and span data as input. The Phase 1 emission adds a tuple writer alongside the existing telemetry path; no other layer of the agent stack changes.

| Phase | What the lab does | What the lab gets |
|---|---|---|
| 1 | Emit TAO tuples for tool calls on one agent surface (suggestion: code agent or browser agent). Passive logging only. | Structured, comparable behavioral records across that surface. |
| 2 | Add a sidecar observer at `PRIVILEGE_ISOLATED` level. Begin computing CCD on flagged verbs. | Detection of claim/effect mismatches on the highest-risk action classes. |
| 3 | Author one signed Mission Profile for the surface. Begin policy-aware audit. | Audit records bound to declared policy, with deviation reports on every override. |
| 4 | Wire CCD inconsistency results into runtime enforcement (block, escalate, log). | Runtime governance on the audit substrate. |
| 5 | Move the observer to `HARDWARE_ISOLATED` or `INSTITUTIONALLY_INDEPENDENT` level for regulator-grade assurance. | Audit-grade attestation suitable for external review. |

Phase 1 should be achievable in weeks for a team that already owns the tool-call telemetry layer; longer if the team has to land it in a new layer first. Phases 2 and 3 are typically the natural cutoff for an MVP. Phases 4 and 5 are where the spec earns its way into product surfaces that demand regulatory or enterprise-grade trust.

## Relationship to adjacent work

TAO is complementary to several existing efforts rather than competitive with them. OpenTelemetry GenAI conventions and platform tools like LangSmith, Langfuse, and Datadog standardize traces, spans, and tool-call telemetry — TAO consumes their substrate. The IETF Agent Audit Trail draft (March 2026) standardizes tamper-evident log records of agent events — TAO sits one layer above and adds the semantic-vs-mechanical integrity check. Runtime governance work such as Microsoft's Agent Governance Toolkit and Open Policy Agent decides whether an action is allowed — TAO produces the normalized behavioral records those policy engines reason over. AudAgent (arXiv, March 2026) audits privacy-policy compliance for AI agents — TAO is broader in action scope but narrower in policy scope. W3C PROV is the foundational provenance model — TAO's tuples can map into PROV, but TAO adds the anti-laundering mapping and observer-assurance dimensions that PROV is not concerned with.

The phrase TAO owns is **semantic laundering**: an agent or adapter labeling an action with a benign semantic verb while the observed mechanical effects are harmful, unauthorized, or inconsistent with that label. None of the adjacent efforts target this failure mode directly; the wedge is the integrity layer between observability and governance.

## What I'm asking for

This is a working draft, not a finished standard. I'm looking for two things from a serious reader:

1. Whether the audit substrate maps to real platform/safety/evals/governance needs you've seen inside a frontier lab — specifically, whether the kernel is too coarse, whether the verb vocabulary fails on any action class you ship, whether the observer-independence ladder matches how your platform actually isolates things, and whether the Mission Profile override discipline would catch the laundering patterns you've actually run into.

2. A pointer to whoever inside the org thinks about agent auditability, platform governance, or evals infrastructure — particularly anyone whose team would benefit from a shared schema for behavioral records that crosses agent surfaces.

A runnable reference validator is the next planned artifact. If a pilot conversation is on the table, that becomes the priority deliverable.
