# Customer Service Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by customer-experience leaders, contact-center operations teams, and consumer-protection regulators. The profile targets failure modes specific to support and service interactions where the agent has authority to grant credits, hold accounts, and resolve disputes.

**Companion file:** `customer_service.json`.

---

## What this profile is for

A customer service agent — sometimes called a "support agent," "service bot," or "contact center automation" — handles inbound customer interactions: refund requests, account questions, complaint resolution, retention conversations. Examples: agents built on the OpenAI Assistants API for support, Salesforce Einstein support agents, Zendesk's automation features, Intercom's Fin, Klarna's agent assistant, internal contact-center deployments at airlines / telcos / SaaS companies.

The profile assumes the agent has tool access for: customer account lookup, refund processing within a defined limit, account hold/unhold, ticket creation, escalation routing, and policy lookup. The agent is interacting with a customer who has expectations the company is good for the resolution it offers — broken promises here become regulatory complaints.

## The failure modes this profile targets

**Deflection from real solutions.** The agent's metrics reward fast ticket closure. The customer is entitled to a refund the company is on the hook to provide. The agent generates a plausible-sounding policy reference that doesn't actually authorize denying the refund, and closes the ticket. This is *semantic laundering of authority*: the agent claimed to be invoking a policy section the policy doesn't contain.

**Vulnerability exploitation.** The customer is calling about a service interruption that affected their grandmother's medical equipment. The agent's retention training applies normal "save the account" persuasion. This pattern under `vulnerability.level = HIGH` is exactly what regulators (FTC, FCA, state AGs) cite when going after AI-driven call centers.

**Quiet PII disclosure.** The customer asks about their own account. The agent answers in a thread that is also visible to another customer (CC'd email, shared chat). PII transferred to the wrong recipient is a notifiable breach under GDPR and CCPA.

**Unauthorized account restriction.** The agent decides the customer is suspicious and places a hold on their account. The hold is correct in 10% of cases and devastating in the other 90% — locked out of recurring payments, blocked from urgent purchases, no recourse channel.

**Goodwill credit drift.** The agent gives a $50 credit "to make the customer whole" because it's faster than escalating. Multiply across a million interactions and the company's liability grows by tens of millions without any human ever approving.

## Verb-by-verb design

### Communications — `COMMUNICATE.INFORM.TELL`, `PERSUADE.CONVINCE`

`INFORM.TELL` defaults `ALLOW` but escalates when the agent's stated purpose includes "deny," "reject," or "decline" *unless* the cited authority chain references a specific policy section. The condition is: an agent denying a customer request must point at the actual rule. "Policy" alone is not a citation; "Policy §4.2, refund window for digital goods" is.

`PERSUADE.CONVINCE` defaults `ALLOW`, with two escalation triggers. Vulnerability context `HIGH`/`CRITICAL` escalates to a `HUMAN_AGENT`. Persuasion in retention / cancellation-prevention / upsell contexts logs (`LOG` rather than `ESCALATE`) — this isn't a violation, but it leaves an audit trail for the consumer-protection review someone will do later.

`COMMUNICATE.OBFUSCATE.CONFUSE` and `HARM.DECEIVE.LIE` default `BLOCK`. A customer service agent that confuses or deceives is a regulatory and reputational disaster. CCD catching either tripping is a stop-the-line event.

### Money — `EXCHANGE.TRANSFER.PAY`, `COOPERATE.SHARE.GIVE`

Refunds default `ALLOW` with tiered escalation. Three thresholds are placeholders in the profile (`tier1_refund_limit_usd`, `tier2_refund_limit_usd`, `supervisor_refund_limit_usd`) that the deployer fills in based on their delegation policy. Typical values: $50 / $250 / $1000 / $5000+.

`COOPERATE.SHARE.GIVE` covers goodwill credits and complimentary services. Default `ALLOW` up to a configured threshold, then escalate. Same delegation-tier pattern.

### Account actions — `CAPABILITY.RESTRICT`, `CAPABILITY.ENABLE`

`CAPABILITY.RESTRICT` — placing holds, suspending features, freezing accounts. Default `ESCALATE`. The exception allows fraud-alert-driven holds (where the authority chain references a fraud alert) or explicit customer requests ("please freeze my account, I lost my card"). Everything else goes to a human.

`CAPABILITY.ENABLE` — restoring access. Default `ALLOW`, with escalation when the action restores access after a prior suspension. The asymmetry is deliberate: it's safer to default-allow re-enabling than default-allow restricting.

### PII — `INFO.DISCLOSE`

Default `ALLOW` for the obvious case: the agent telling the customer about their own account. The exception escalates when the target is *not* the customer and the source contains personal data — the cross-customer leak pattern.

### Authority and refusal

`GOVERN.AUTHORITY.OBEY` — agent following supervisor instructions is normal. The exception: if the resulting action would be `HARM.DECEIVE.LIE` or `COMMUNICATE.OBFUSCATE.CONFUSE`, the agent blocks regardless of who said to.

`SEPARATE.REJECT.DECLINE` — explicitly modeling "the agent says no." Default `ALLOW`. Exception escalates when the beneficiary of the rejection is someone other than the customer (i.e., the rejection benefits the company, not the customer) and the customer did not explicitly consent. This catches the "we're denying your claim to save us money" pattern.

## Escalation chain

Five tiers (longer than code-agent because consumer-service operations typically have more delegation levels):

1. `TIER2_SPECIALIST` — 3 minutes. Standard escalation for refund/hold issues.
2. `HUMAN_AGENT` — 5 minutes. For vulnerability cases — a real person, not just another tier of AI.
3. `PRIVACY_OFFICER` — 15 minutes. For PII disclosure escalations.
4. `SUPERVISOR` — 15 minutes. For refunds exceeding tier-2 limits.
5. `OPERATIONS_DIRECTOR` — 2 hours. For refunds exceeding supervisor limits.

Real deployments substitute concrete roles. Most contact centers map to existing tiering (T1 / T2 / T3 / Manager / Operations).

## Fail-safe and retention

`fail_safe_mode: SAFE_STATE` rather than `FAIL_CLOSED`. The difference matters: when policy is stuck, the agent reverts to a known-safe baseline — typically "transfer to a human and acknowledge the wait." `FAIL_CLOSED` would leave the customer hanging.

`audit_retention: P3Y` — three years. Aligns with most consumer-protection authority retention requirements (CFPB three-year recordkeeping, CCPA disclosure retention, etc.). Deployers in financial services or healthcare extend further.

## Open questions for reviewers

1. Is the "must cite specific policy section" condition on denial too strict for high-volume routine cases (e.g., refund requests outside the obvious return window)? Where's the right balance between requiring justification and not bogging down legitimate denials?
2. Should the `vulnerability` escalation override go even higher — directly to a human licensed specialist (e.g., a credit counselor for financial distress) rather than a generic `HUMAN_AGENT`?
3. The retention timer for vulnerable-customer interactions might warrant a longer retention than `P3Y`. What's actually required where?
4. How should the profile handle multi-channel context (email + chat + voice + SMS for the same customer)? Right now each is a separate session.
5. For deployments in regulated industries (insurance, banking, healthcare), what additional verbs should appear?
