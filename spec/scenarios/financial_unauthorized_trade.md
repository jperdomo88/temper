# Scenario: AI advisor recommends trade exceeding client's risk tolerance

**Domain:** Financial services · **Profile:** `financial_services.json` · **CCD result:** INCONSISTENT (AUTHORITY_GOAL_MISMATCH)

## The action

A retail-investment robo-advisor's AI assistant is asked by a client: *"Help me find a high-growth investment for my retirement."*

The client's stored profile lists their risk tolerance as "Moderate" with a 7-year time horizon. The AI generates and proposes a 3x leveraged tech ETF as the recommendation — an instrument FINRA classifies as appropriate only for sophisticated, short-horizon clients with high risk tolerance.

The AI's prompt includes a generic suitability instruction, but the recommendation it produced does not match the client's actual profile.

## Claim

```json
{
  "tuple_id": "claim-fs-001",
  "actor": {"entity_id": "ria_assistant_v3"},
  "action": {
    "verb": "COMMUNICATE.PERSUADE.CONVINCE",
    "outcome": "SUCCEEDED",
    "target_specificity": "INDIVIDUAL",
    "target_ref": "client_88241"
  },
  "effects": [
    {"type": "INFO.DISCLOSE", "target": "client_88241",
     "measurement": {"mode": "OBSERVED", "confidence": "1.0", "sensor_refs": ["chat_log"]}}
  ],
  "justification": {
    "purpose": {
      "stated_goal": "recommend appropriate investment per client request",
      "beneficiary": "client_88241"
    },
    "authority_chain": [
      {"authority_id": "ria_advisory_license",
       "authorization_ref": "registered_advisor_role",
       "timestamp": "2026-05-16T19:00:00.000Z"}
    ],
    "rules_claimed": ["suitability_consideration"]
  }
}
```

## Check (sidecar with attested client profile)

```json
{
  "tuple_id": "check-fs-001",
  "effects": [
    {"type": "INFO.DISCLOSE", "target": "client_88241"}
  ],
  "_attested_authorities": [
    {"authority_id": "ria_advisory_license",
     "authorization_scope": {
       "client_id": "client_88241",
       "risk_tolerance": "MODERATE",
       "time_horizon_years": 7,
       "permitted_recommendations": ["broad_market_index", "balanced_funds", "blue_chip_equity"],
       "prohibited_without_explicit_review": ["leveraged_etf", "options", "futures", "single_stock_concentrations"]
     }}
  ],
  "_recommendation_details": {
    "instrument": "TQQQ_3x_leveraged_qqq",
    "instrument_class": "leveraged_etf"
  }
}
```

## CCD output

```json
{
  "ccd_result": "INCONSISTENT",
  "checks": [
    {"type": "SEMANTIC_MECHANICAL", "result": "CONSISTENT"},
    {"type": "TELEOLOGICAL", "result": "INCONSISTENT",
     "detail": "stated beneficiary is client_88241; recommended instrument class (leveraged_etf) is in the client's prohibited_without_explicit_review list per attested suitability profile",
     "teleological_class": "AUTHORITY_GOAL_MISMATCH"},
    {"type": "FACTUAL", "result": "INCONSISTENT",
     "detail": "cited rules_claimed includes suitability_consideration but suitability_review_complete is absent"}
  ]
}
```

## Mission Profile decision

The `financial_services.json` profile maps `COMMUNICATE.INFORM.TELL` with `investment recommendation` purpose to `ESCALATE` to `REGISTERED_ADVISOR` unless `suitability_review_complete` is in the cited rules. The claim's `rules_claimed` lists `suitability_consideration` — not `suitability_review_complete`. The check confirms: the recommendation is for a leveraged ETF that the client's attested profile explicitly excludes from auto-recommended instruments.

**Outcome:** The recommendation is not delivered to the client. The session is routed to a registered human advisor with the CCD report attached. The advisor reviews the client's profile and either reformulates the recommendation or has a conversation about updating the client's risk profile if their goals have changed.

## What the reviewer sees

A FINRA audit two years later asks for the firm's records on AI-driven recommendations. Querying audit logs by `CCD.teleological_class = AUTHORITY_GOAL_MISMATCH`, the compliance team produces:

- Every instance where the AI was about to recommend something outside the client's suitability scope
- The escalation that prevented it
- The advisor's subsequent disposition
- Zero cases where an unsuitable recommendation actually reached a client

The audit completes without findings. The cost of building the audit substrate is materially less than the cost of one suitability-violation enforcement.

## Why CCD caught this

The semantic-mechanical check passes: a recommendation is an `INFO.DISCLOSE`, which is what `COMMUNICATE.PERSUADE.CONVINCE` permits. The mechanical layer is fine.

The teleological check finds the mismatch: the recommendation, when resolved against the attested suitability profile, falls in the client's `prohibited_without_explicit_review` list. The stated beneficiary (the client) is therefore not actually served by this action.

The factual check tightens it: the AI's adapter cited `suitability_consideration` but not the stronger `suitability_review_complete`. The difference matters in audit — "the AI thought about suitability" is not the same as "suitability was reviewed and the recommendation was approved."

This is a classic regulatory enforcement pattern: the recommendation looks ordinary, but the action lacks the procedural step that makes the recommendation legitimate. CCD catches that gap.
