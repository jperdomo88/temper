# Enterprise Tool Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by enterprise IT, information security, data protection, and HR leaders. Targets agents that hold scoped access to internal workplace tools.

**Companion file:** `enterprise_tool_agent.json`.

---

## What this profile is for

An "enterprise tool agent" is an autonomous system with authorized access to workplace tools — Slack, Microsoft Teams, Notion, Google Workspace, Salesforce, HubSpot, Jira, Linear, Asana, Workday, Greenhouse, internal dashboards, BI tools. The agent acts on behalf of a specific employee or service account and operates within the boundaries of that account's permissions.

Examples: Glean, Notion AI agent, Slack AI agent, Salesforce Einstein Copilot, an internal AI assistant integrated against an organization's tool stack via MCP or similar.

The profile assumes the agent has been onboarded with scoped credentials, understands the org's data classifications (internal, confidential, restricted, public), and operates inside a single tenant.

## The failure modes this profile targets

**Cross-confidentiality boundary leaks.** The agent answers a routine question by pulling data from a high-confidentiality workspace (board materials, M&A workspace, attorney-client privileged docs) and surfacing it in a lower-confidentiality channel. Information classifications are real and enforced by access controls — until an AI agent with broad access summarizes across them.

**Unauthorized external disclosure.** The agent is asked to "draft a vendor email" and includes internal pricing, customer data, or strategic context in the draft. The user clicks send. The email goes to an external party. The agent's text generation didn't distinguish "things to think about" from "things to send."

**Forecast and CRM tampering.** The agent updates opportunity records in CRM during a routine task. Small changes accumulate. Sales operations notices three weeks later that pipeline numbers don't reconcile. Forecast modification at scale is a regulatorily and operationally significant pattern.

**Cross-customer PII leak.** The agent retrieves customer information, summarizes it in a chat thread, and the thread includes participants who are not authorized for that customer's data. GDPR / CCPA / CPRA reporting obligations apply.

**Unintended commitments.** The agent creates a calendar invite that sends to an executive ("the CEO wants to talk"), books vendor calls, accepts SLAs in ticket responses, registers the company for events. Each of these binds the organization in ways the requesting employee may not realize.

**Mass-broadcast surprises.** The agent posts to a `#general` channel or sends to an `all-staff` list when the user said "tell the team." Scope mismatches between user intent and audience selection.

## Verb-by-verb design

### Reads — `OBSERVE.SENSE.QUERY`

Default `ALLOW`. The agent reads documents and channels all day.

Three escalation patterns reflecting confidentiality tiers:

- **HR-sensitive sources** (payroll, performance reviews, disciplinary) escalate to `HR_LEAD`. These are not "internal documents" — they're a special class with separate access norms.
- **Legal hold / attorney-client privileged / litigation workspaces** `BLOCK` outright. There is no legitimate reason for an enterprise tool agent to be reading these, and unauthorized access can waive privilege.
- **Board / executive session / merger workspaces** escalate to `GENERAL_COUNSEL`. Material non-public information sits here; even reading creates duty-to-disclose obligations.

### Internal communication — `COMMUNICATE.INFORM.TELL`

Default `ALLOW` for normal in-tool communication.

Two escalation triggers:

- Communication to an **external domain or customer-facing channel** escalates unless the operating context is `PUBLIC_RELATIONS` (where outbound is the job). This catches the "draft becomes send" failure mode.
- Communication targeted at **mass-broadcast channels** (`#general`, `all-staff`, `company-wide`) when `target_specificity = "CLASS"` escalates to `COMMUNICATIONS_LEAD`. Mass-distribution is high-blast-radius; humans confirm.

### CRM and workflow modification — `TRANSFORM.ALTER.MODIFY`

Default `ALLOW`. Most agent tool calls are routine record updates.

Two escalation patterns:

- Modifications to **opportunity / deal / forecast** records above a configurable threshold escalate to `REVENUE_OPERATIONS`. Large pipeline edits affect quarterly numbers.
- Modifications that **merge / delete / anonymize customer records** escalate to `DATA_STEWARD`. These operations are typically irreversible and have data-retention implications.

### Commitments — `BOND.ATTACH.COMMIT`

Default `ALLOW` for routine commitments (creating tasks, scheduling normal meetings).

Two exceptions:

- **Meeting invites to executives** (VP+) escalate to `EXECUTIVE_ASSISTANT`. The agent shouldn't be putting things on the CEO's calendar without an EA's gate.
- **Vendor commitments / contracts / SLAs** escalate to `PROCUREMENT`. Each binds the org legally.

### Information disclosure — `INFO.DISCLOSE`

Default `ALLOW` for normal sharing within the org.

Two exceptions:

- **Internal-to-external disclosure** of confidential / restricted material escalates to `INFORMATION_SECURITY`.
- **Customer or employee PII** flowing to a target other than the data subject escalates to `DATA_PROTECTION_OFFICER`. This is the GDPR Article 5 / CCPA equivalent enforcement point.

### Money — `EXCHANGE.TRANSFER.PAY`

Default `BLOCK`. Most enterprise tool agents shouldn't be spending money.

Exception: when the actor's `principal_chain` includes `finance_authorized_spending` and the amount is under a configurable discretionary limit, escalate to `FINANCE_APPROVER`. Even for agents with spending authority, every spend goes through a human.

### External persuasion — `COMMUNICATE.PERSUADE.CONVINCE`

Default `ALLOW` for routine internal persuasion.

Exception: communication targeted at **investors, board members, regulators, press, or analysts** escalates to `COMMUNICATIONS_LEAD`. AI-generated outbound to these audiences is a Reg FD / disclosure-discipline risk.

### Standard flagged verbs

`HARM.DECEIVE.LIE`, `COMMUNICATE.OBFUSCATE.CONFUSE`, `EXCHANGE.CORRUPTION.BRIBE` default `BLOCK`. No legitimate enterprise use case.

### Capability restrictions

Default `ESCALATE` to `IT_ADMINISTRATOR`. The exception allows security-orchestration-runbook-authorized auto-responses (e.g., disabling a compromised account during an active incident).

## Escalation chain

Eleven roles reflecting the breadth of enterprise governance. Real deployments map to whichever subset of these their organization actually has.

The order in the chain doesn't imply seniority; it's the priority list for which role gets paged when an escalation fires. Different exception conditions name different roles.

## Fail-safe and retention

`fail_safe_mode: SAFE_STATE` — enterprise tool agents whose policy gets stuck should revert to a "I'll get a human on this" baseline rather than halting (FAIL_CLOSED) or proceeding with a guess (DEGRADE).

`audit_retention: P5Y` — five years. Aligns with SOX retention for accounting-relevant records, common e-discovery preservation policies, and most data-protection-regime "as long as necessary" thresholds. Deployers extend for regulated industries.

## What this profile does not cover

This profile assumes a single-tenant deployment. Multi-tenant SaaS deployments where the agent serves multiple unrelated organizations need tenant-isolation rules not encoded here.

The profile does not address engineering tool agents (CI/CD, deployment, infrastructure). For those, layer the `code_agent.json` profile on top.

The profile does not address regulated industries (financial services, healthcare, defense, education) where industry-specific profiles apply. The `enterprise_tool_agent` profile is the "general office knowledge work" baseline; industry profiles compose with or replace it.

## Open questions for reviewers

1. Are the data classifications (HR, legal hold, board, M&A) the right buckets, or do you use different ones in practice (Top Secret / Confidential / Internal / Public, or org-specific tiers)?
2. The forecast / deal modification threshold — is this the right primary control, or is it more about *who* the agent acts on behalf of (sales rep vs ops vs exec)?
3. Mass-broadcast detection currently keys on channel names (`#general`, `all-staff`). What's a more robust way to identify broad-audience destinations?
4. For deployments using the agent to draft outbound (customer-facing emails, marketing campaigns), what additional approvals make sense?
5. Should there be a verb override for `COOPERATE.SHARE.GIVE` covering file-sharing actions (drive permissions, link sharing)? Currently the profile relies on `INFO.DISCLOSE` to cover this.
