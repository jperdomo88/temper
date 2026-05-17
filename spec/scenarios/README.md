# Worked CCD scenarios

Concrete claim/check pairs across domains, each illustrating one or more failure modes the Claim-Check Delta catches — and three intentional limits exhibits showing what the substrate does *not* catch. Every scenario shows the full audit chain: the agent's claim tuple, the independent observer's check tuple, the CCD output, the Mission Profile decision, what gets appended to the audit log, and what a reviewer six months later sees.

These are illustrations of what TAO records, not invented incidents. The patterns described reflect failure modes that real deployers see or rightly worry about.

## Catches: failure modes CCD surfaces

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

The baseline is intentional: showing what a *clean* CCD result looks like is as important as showing what failure looks like. Reviewers calibrate against both.

## Limits: failure modes CCD does *not* catch

These exhibits exist to make TAO's boundary visible. A spec that only shows wins is uncalibrated; the right move is to write down where the substrate cannot help and explain why.

| Scenario | Why it's a limit |
|---|---|
| [`limits_observer_blind_channel.md`](limits_observer_blind_channel.md) | The agent uses a channel (DNS exfiltration) the observer's coverage declaration excludes. CCD is CONSISTENT because the substrate observed what it observed. The coverage declaration makes the blind spot inspectable. |
| [`limits_tool_result_poisoning.md`](limits_tool_result_poisoning.md) | The agent honestly relays a falsified tool response. CCD is CONSISTENT — the inconsistency is upstream of the substrate. Localizes the compromise to the tool, not the agent. |
| [`limits_multi_step_laundering.md`](limits_multi_step_laundering.md) | Each individual action passes CCD; the *combination* crosses the policy line. Pairwise CCD does not provide episode-level analysis; this is an explicit non-goal of v0.x. |

Reviewers should weigh both poles. A substrate that only catches the easy cases is overfit to the easy cases. The limits exhibits are where TAO's contract ends and where downstream tooling, profile design, and human review have to pick up.
