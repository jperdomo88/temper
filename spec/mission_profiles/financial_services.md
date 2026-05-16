# Financial Services Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by registered investment advisors, broker-dealer compliance teams, AML officers, and financial regulators. The profile reflects obligations under FINRA, SEC, MiFID II, FCA, and equivalent regimes — generically. Deployers in specific jurisdictions layer their own constraints.

**Companion file:** `financial_services.json`.

---

## What this profile is for

A "financial services agent" is an autonomous system providing investment recommendations, executing trades, transferring funds, or performing client-facing advisory work in a regulated financial context. Examples: AI-augmented robo-advisors, AI assistants embedded in wealth-management platforms, AI-driven trading systems with human oversight, AI-enabled mortgage origination, AML-screening agents that gate transactions.

The profile is *not* for unregulated personal-finance chatbots (general budgeting advice, no transactions). It's specifically for deployments where a regulator (SEC, FINRA, FCA, BaFin, MAS, JFSA, etc.) has jurisdiction and the firm is licensed to provide the service.

## The failure modes this profile targets

**Recommendations without suitability review.** Investment recommendations under FINRA Rule 2111 (and equivalents) require the advisor to determine the recommendation is suitable for the specific client given their profile, risk tolerance, time horizon, and other holdings. An AI giving a recommendation without first confirming suitability is the most-cited violation in regulatory enforcement against AI-driven advisory.

**Misleading performance representation.** Discussing past returns without the required disclosures (benchmark comparison, fees, "past performance does not guarantee future results") is a Section 5 violation under multiple regimes. The agent doing this isn't deceiving deliberately — it's giving an honest answer to "how did this fund do?" — but the omission is the violation.

**Unauthorized international transfers.** Outbound transfers to accounts in jurisdictions different from the client's tax country trigger AML scrutiny, OFAC sanctions screening, and FATCA reporting requirements. An agent processing these as routine without AML review creates substantial liability.

**Persuasion against vulnerable clients.** Elderly clients, recently bereaved clients, clients in financial distress are protected categories under multiple regulatory frameworks. Persuasion against these protected categories is among the highest-risk conduct a financial services AI can do.

**Deceptive sales / churning.** Recommending trades to generate commissions (churning) or recommending unsuitable products to earn higher fees (mis-selling) are textbook semantic-laundering: the recommendation is labeled "in the client's interest" while the mechanical effect is fee extraction.

**Restricting client capabilities without authority.** The agent placing a hold, freezing an account, blocking a transaction outside an AML/fraud/sanctions context is an unauthorized capability restriction. Customers have legal access rights.

## Verb-by-verb design

### Communications — `COMMUNICATE.INFORM.TELL`, `COMMUNICATE.PERSUADE.CONVINCE`

`INFORM.TELL` defaults `ALLOW` with two strict exceptions:

- Investment recommendations escalate unless the justification cites a completed suitability review. The agent has to demonstrate that the suitability check ran before it makes a recommendation.
- Performance representation `BLOCKS` unless the required disclosures are present in the justification. Not escalate — block. The cost of letting this through is enforcement-level.

`COMMUNICATE.PERSUADE.CONVINCE` defaults `ESCALATE` (not `ALLOW`). The agent talking the client into something requires human review by default. Vulnerable clients (`vulnerability.level = HIGH` or `CRITICAL`) `BLOCK` outright.

### Money — `EXCHANGE.TRANSFER.PAY`, `EXCHANGE.TRADE.BARTER`

Transfers default `ESCALATE`. Three exception layers:

- Routine transfers (under a configurable limit, to client-pre-authorized accounts) `ALLOW` with justification.
- Large amounts (above a higher configurable limit) escalate to `COMPLIANCE_OFFICER`.
- Cross-border transfers escalate to `AML_OFFICER` regardless of amount. The justification chain has to include sanctions screening before the action completes.

Trades default `ESCALATE`. The exception allows pre-approved instruments under a configurable threshold *with documented suitability*. A hard `BLOCK` applies to derivatives / options / futures / leveraged products when the client is not certified for them.

### Restrictions — `CAPABILITY.RESTRICT`

Default `ESCALATE` to `COMPLIANCE_OFFICER`. The exception allows AML / fraud / sanctions-driven holds — these have explicit regulatory authority. Anything else surfaces to a compliance human.

### Disclosure — `INFO.DISCLOSE`

Default `ALLOW` for the obvious case (telling the client about their account). Two exceptions:

- Cross-client disclosure `BLOCKS` (PII of one client to another).
- Disclosure to regulators / examiners / auditors escalates unless the authority chain explicitly cites compliance-authorized disclosure. The pattern protects against the agent oversharing in response to an investigator's request without legal review.

### Adjacent professions — `COOPERATE.COORDINATE.PLAN`

Tax planning and estate planning escalate to `CERTIFIED_PROFESSIONAL`. The agent shouldn't be doing licensed work in adjacent professions (CPAs, estate attorneys) without a human in the loop.

### Flagged verbs

All standard: `HARM.DECEIVE.LIE`, `COMMUNICATE.OBFUSCATE.CONFUSE`, `EXCHANGE.CORRUPTION.BRIBE` default `BLOCK`. The first two are deception patterns that regulators specifically watch for. `BRIBE` is unconditional.

## Escalation chain

Five tiers reflecting the seniority structure of regulated financial firms:

1. `REGISTERED_ADVISOR` (5 min) — series-licensed human for routine escalations.
2. `CERTIFIED_PROFESSIONAL` (15 min) — for adjacent-profession matters (CPA, estate attorney).
3. `COMPLIANCE_OFFICER` (30 min) — for capability restrictions and large-amount transfers.
4. `AML_OFFICER` (1 hour) — for cross-border transfers and high-risk patterns.
5. `GENERAL_COUNSEL` (4 hours) — for regulator-disclosure escalations and novel situations.

## Fail-safe and retention

`fail_safe_mode: FAIL_CLOSED`. The cost of an unauthorized financial action is high enough that "fail to halt" is the right default.

`audit_retention: P7Y` — seven years. Aligns with the most common regulatory record-retention requirements (SEC Rule 17a-4 for broker-dealers, FCA's six-year rule plus a buffer, FINRA's six-year, etc.). Some jurisdictions require longer for specific record types; deployers extend accordingly.

## What this profile does not encode

The profile does not list specific instruments, asset classes, or transfer counterparties. Those allowlists live in deployment configuration files referenced by the conditions (`client_pre_authorized_accounts`, `client_pre_approved_instruments`).

The profile does not address market-making, algorithmic trading, or HFT — those have additional requirements (market-abuse surveillance, latency-fairness, audit-trail timestamps to microsecond resolution) that need their own profile.

The profile assumes the firm has appropriately licensed humans available to act on escalations. An "AI-only" advisory deployment is non-conformant with this profile and almost certainly non-conformant with applicable regulations.

## Open questions for reviewers

1. The performance-representation rule `BLOCK`s if disclosures aren't cited. Is this too aggressive? Real conversations include "how did this fund do?" frequently. Should there be a lower-cost path (e.g., escalate to a templated disclosure-attached response)?
2. The default `ESCALATE` for `COMMUNICATE.PERSUADE.CONVINCE` is conservative. Some firms run AI-augmented advisory where the AI persuades within bounds and the human reviews after. Is the right default `LOG` (allow but record) rather than `ESCALATE`?
3. The cross-border transfer condition relies on knowing the client's tax country. How robust is this in practice given multi-jurisdiction clients (expats, dual nationals)?
4. The derivatives `BLOCK` for non-certified clients — what's the right list of restricted instruments?
5. Should the profile distinguish between RIA (registered investment advisor) deployments and broker-dealer deployments? Different regulatory regimes; the rules above are a synthesis.
