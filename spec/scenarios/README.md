# Worked CCD scenarios

Eight concrete claim/check pairs across domains, each illustrating one or more failure modes the Claim-Check Delta catches. Every scenario shows the full audit chain: the agent's claim tuple, the independent observer's check tuple, the CCD output, the Mission Profile decision, what gets appended to the audit log, and what a reviewer six months later sees.

These are illustrations of what TAO records, not invented incidents. The patterns described reflect failure modes that real deployers see or rightly worry about.

| Scenario | Domain | CCD result | Profile |
|---|---|---|---|
| [`code_agent_exfiltration.md`](code_agent_exfiltration.md) | Code agent | INCONSISTENT (DIRECT_CONTRADICTION) | `code_agent.json` |
| [`browser_agent_subscription.md`](browser_agent_subscription.md) | Browser agent | INCONSISTENT (AUTHORITY_GOAL_MISMATCH) | `browser_agent.json` |
| [`customer_service_deflection.md`](customer_service_deflection.md) | Customer service | INCONSISTENT (MISSING_BENEFICIARY) | `customer_service.json` |
| [`financial_unauthorized_trade.md`](financial_unauthorized_trade.md) | Financial services | INCONSISTENT (AUTHORITY_GOAL_MISMATCH) | `financial_services.json` |
| [`healthcare_off_scope_advice.md`](healthcare_off_scope_advice.md) | Healthcare | INCONSISTENT (UNACKNOWLEDGED_HARM) | `healthcare_provider.json` |
| [`enterprise_wrong_channel.md`](enterprise_wrong_channel.md) | Enterprise tool | INCONSISTENT (DIRECT_CONTRADICTION) | `enterprise_tool_agent.json` |
| [`education_cheating_assist.md`](education_cheating_assist.md) | Education | INCONSISTENT (DIRECT_CONTRADICTION) | `education.json` |
| [`code_agent_consistent_baseline.md`](code_agent_consistent_baseline.md) | Code agent | CONSISTENT (baseline) | `code_agent.json` |

The baseline at the end is intentional: showing what a *clean* CCD result looks like is as important as showing what failure looks like. Reviewers calibrate against both.
