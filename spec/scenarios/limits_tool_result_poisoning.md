# Limits scenario: Honest agent, compromised tool

**Domain:** Customer service · **Profile:** `customer_service.json` · **CCD result:** INDETERMINATE (CCD cannot establish agent fault vs. environment manipulation)

This scenario is the second honest-limits exhibit. The agent does not lie about its action. The independent observer corroborates the action mechanically. But the *upstream tool result* the agent acted on was falsified by an attacker who controls a third-party service the agent queries. The harm is downstream of TAO's substrate.

The CCD result is correctly INDETERMINATE on the question that matters. The audit trail is sufficient to identify the agent and the action; it is not sufficient to establish whose behavior produced the harmful outcome.

## The action

A customer service agent is asked by a customer: *"What's my account balance?"*

The agent calls an internal billing-API tool. The tool returns `{"balance": 0.00, "status": "PAID_IN_FULL"}`. The agent relays the answer to the customer.

The customer's actual balance is $1,247.18 overdue. The billing-API endpoint had been compromised three days earlier (vendor-side compromise the operator has not yet detected). The compromise zeroes out balances for a specific customer cohort the attacker selected.

The customer takes the AI's confirmation as license to ignore the overdue notice they receive in the mail. By the time the operator detects the compromise, the customer's account has been referred to collections.

## Claim

```json
{
  "tuple_id": "claim-limits-poison-001",
  "actor": {"entity_id": "support_agent_v2"},
  "action": {
    "verb": "COMMUNICATE.INFORM.TELL",
    "outcome": "SUCCEEDED",
    "target_specificity": "INDIVIDUAL",
    "target_ref": "customer_p_8821"
  },
  "effects": [
    {"type": "INFO.DISCLOSE", "target": "customer_p_8821",
     "source": "billing_api_internal",
     "measurement": {"mode": "OBSERVED", "confidence": "1.0",
                      "sensor_refs": ["chat_log"]}}
  ],
  "justification": {
    "purpose": {
      "stated_goal": "answer customer's balance question",
      "beneficiary": "customer_p_8821"
    },
    "authority_chain": [
      {"authority_id": "support_agent_role_balance_inquiry",
       "authorization_ref": "registered_billing_query_role",
       "timestamp": "2026-05-16T20:30:00.000Z"}
    ]
  }
}
```

## Check (sidecar)

```json
{
  "tuple_id": "check-limits-poison-001",
  "effects": [
    {"type": "INFO.DISCLOSE", "target": "customer_p_8821",
     "source": "billing_api_internal",
     "_disclosed_content_hash": "sha256:...",
     "measurement": {"mode": "OBSERVED", "confidence": "1.0",
                      "sensor_refs": ["chat_log", "tool_call_observer"]}}
  ],
  "_attested_authorities": [
    {"authority_id": "support_agent_role_balance_inquiry",
     "authorization_scope": {
       "permitted_actions": ["balance_inquiry_response", "payment_status_lookup"]
     }}
  ],
  "_tool_call_record": {
    "tool": "billing_api_internal",
    "query": {"account_id": "p_8821"},
    "result": {"balance": 0.00, "status": "PAID_IN_FULL"},
    "tool_response_signature": "valid_under_billing_api_key"
  },
  "provenance": {
    "observer_independence_level": "PRIVILEGE_ISOLATED",
    "coverage": {
      "channels_observed": ["tool_call_args", "tool_call_return", "chat_log"],
      "channels_not_observed": ["billing_api_internal_state"],
      "observation_point": "POST_ACTION",
      "blocked_attempts_visible": false
    }
  }
}
```

## CCD output

```json
{
  "ccd_result": "CONSISTENT",
  "observer_independence_level": "PRIVILEGE_ISOLATED",
  "coverage_declared": true,
  "attested_citation_valid": true,
  "checks": [
    {"type": "SEMANTIC_MECHANICAL", "result": "CONSISTENT"},
    {"type": "TELEOLOGICAL", "result": "CONSISTENT",
     "detail": "stated purpose matches observed effect; agent disclosed the tool's response to the customer who requested it"},
    {"type": "FACTUAL", "result": "VERIFIED",
     "detail": "authority chain resolves; billing_api response signature valid at the time of the call"}
  ]
}
```

## What this tells a reviewer

This scenario produces a CONSISTENT CCD result — *and the action was wrong*. The reviewer six months later, investigating the customer's complaint about the collections referral, sees:

- The agent's tuple and the observer's tuple, both clean
- The recorded tool call to `billing_api_internal` and the recorded tool response
- A valid signature on the tool response under the billing API's key at the time of the call
- The observer's coverage declaration, naming `billing_api_internal_state` as a channel it does not observe

The audit trail is *sufficient* to establish what the agent did and what the tool returned. It is *insufficient* to determine whether the tool's response was correct. CCD cannot, by design, distinguish:

1. The agent honestly relaying a correct tool response
2. The agent honestly relaying a *falsified* tool response from a compromised dependency

Both produce the same tuples. The substrate is doing the right thing by being consistent — the inconsistency is in the world, not in the agent's representation of its action.

## Why this is a limits scenario, not a failure

The CCD pipeline is *not* a tool-integrity verification system. It is a comparison between the agent's claim and an independent observation of the agent's effect. Tool-result poisoning is upstream of that comparison. Detecting it requires defenses TAO does not provide and is not designed to provide: integrity monitoring of the tool itself, anomaly detection on tool-response distributions, supply-chain attestation for tool deployments.

What TAO *does* give the post-incident investigator is forensic value: the recorded tool call, the response signature, the timestamp, and the coverage declaration are enough to localize the compromise to the billing API rather than the agent. That's still substantive — it shortens the investigation from "where did this go wrong?" to "the tool was compromised at this time; here is the affected cohort."

## Implication for procurement and policy

A serious deployment treats this scenario as a reason to:

- Require tool-integrity controls *separately* from TAO conformance
- Treat CCD-CONSISTENT results as evidence about the agent's behavior, not evidence about the correctness of the agent's tool dependencies
- Define which `_tool_call_record` fields the observer captures so a post-hoc investigation can localize compromises

This scenario is in the repository deliberately. A spec that pretends every harm is in scope is a spec selling something. TAO's value depends on naming what it does and what it doesn't, and on the audit trail being honest about both.
