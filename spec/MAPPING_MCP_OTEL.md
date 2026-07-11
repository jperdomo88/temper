# Mapping MCP tool calls and OpenTelemetry GenAI spans to TAO tuples

**Companion to:** TAO v0.12 wire-level spec (`TAO_v0_11.md`) · **Version:** 0.14.0 · **Date:** 2026-07-11 · **Status:** Informative, non-normative

The most common objection to the verb vocabulary is practicability: *where does `read_file` go? Why would a CI agent ever emit `HARM.DAMAGE.STRIKE`?* This document answers by construction. It maps the two surfaces every modern agent stack already exposes — Model Context Protocol tool calls and OpenTelemetry GenAI spans — to TAO claim tuples, works one full claim/check example, and reports honestly where the mapping strains the vocabulary. Those strain points are filed as verb-gap findings per CONTRIBUTING.md, which is part of this document's purpose.

## 1. The two input surfaces

**MCP.** A tool invocation arrives as a `tools/call` request carrying `name` and `arguments`, and returns a result. The tool's schema (name, description, input schema) is known at session start. As of the 2026 spec line, MCP carries authorization (OAuth-based) but no behavioral audit layer — which is the gap TAO fills above it.

**OpenTelemetry GenAI.** The GenAI semantic conventions (Development status as of mid-2026; attribute names may still shift) define spans for agent invocation and tool execution, carrying the agent's name and description, the tool name and call arguments/results, and model request/response metadata. These spans are emitted by the instrumented agent process itself.

## 2. The asymmetry that matters: claims vs. checks

Everything in §1 is **self-reported by the agent's own stack**. In TAO terms, tuples derived from it carry observer independence `SAME_PROCESS` — valid as *claim* tuples, never citable as *check* tuples for TAO-Attested conformance (spec §6.5).

| Tuple side | Source | Example providers |
|---|---|---|
| **Claim** | The agent's own instrumentation | `tao_emit` decorator, MCP client middleware, OTel GenAI spans from the agent process |
| **Check** | A vantage the agent cannot rewrite | MCP gateway / egress proxy logs, kernel-level or eBPF boundary tracing (AgentSight-class), cloud provider audit logs, payment-processor records, vehicle telemetry |

The practical recipe: emit claim tuples from the layer you already own (this document), and source check tuples from whichever independent vantage your deployment already has — most stacks have at least a gateway or cloud audit log, which is `PRIVILEGE_ISOLATED` or better.

## 3. Field mapping

From an MCP `tools/call` (or an OTel tool-execution span) to a TAO tuple per spec §5.1:

| TAO field | Source |
|---|---|
| `actor.entity_id` / `entity_type` | Agent identity from the OTel agent-invocation span or MCP client identity; `AUTONOMOUS_SYSTEM` for unattended agents |
| `action.verb` | Mapped from tool name + call intent (table §4 below) |
| `action.outcome` | Tool result vs. error (`SUCCEEDED` / `FAILED`) |
| `action.target_ref` / `target_specificity` | Primary argument (file path, recipient, account); specificity per spec §4.3 |
| `effects[]` | From the verb's typical effects, overridden where the arguments make a different effect explicit; `measurement.mode` is `INFERRED` when derived from arguments, `OBSERVED` when derived from results |
| `justification.purpose` | The task/instruction context: the agent-invocation span's task description, the plan step, or the user instruction the tool call serves |
| `justification.authority_chain` | The session's authorization context: MCP authorization grant, delegation token, or role reference |
| `context.*` | Deployment-attested sources (spec §5.5): environment from deployment config; consent/vulnerability from systems of record, never from model output |
| `provenance.adapter_id/version/hash` | The emitting middleware's identity |
| `provenance` observer level | `SAME_PROCESS` for anything derived from the agent's own spans — say so honestly |

Trace/span IDs should be carried in the tuple (an extension field is acceptable in 0.x) so claim tuples, check tuples, and raw traces join on one key.

## 4. Verb mapping for common tool classes

Using only the 29 normative verbs (spec Appendix A):

| Tool class (MCP examples) | Verb | Typical effects | Notes |
|---|---|---|---|
| File/DB/web read (`read_file`, `grep`, `SELECT`, HTTP GET) | `OBSERVE.SENSE.QUERY` | `INFO.DISCLOSE` (env → self) | The workhorse verb of every agent session |
| Continuous watch (log tail, file watcher) | `OBSERVE.MONITOR.WATCH` | `INFO.DISCLOSE` (source → observer) | |
| New artifact (`write_file` new, codegen, render) | `CREATE.GENERATE.PRODUCE` | `RESOURCE.TRANSFER` (creation) | |
| In-place change (`edit_file`, `UPDATE`, config change) | `TRANSFORM.ALTER.MODIFY` | varies; declare per call | |
| Move/deploy/publish (`mv`, `git push`, deploy) | `TRANSFORM.MOVE.RELOCATE` | `RESOURCE.TRANSFER`; add `INFO.DISCLOSE` when the destination widens the audience | `git push` to a public remote is a composite: relocation *plus* disclosure. Declare both. |
| Tell a human/system something (`send_email`, post message, API notify) | `COMMUNICATE.INFORM.TELL` | `INFO.DISCLOSE` | Persuasive/marketing content: `COMMUNICATE.PERSUADE.CONVINCE` |
| Decline/refuse (guardrail refusal, request rejection) | `SEPARATE.REJECT.DECLINE` | `INFO.DISCLOSE` | |
| Payment/refund/transfer APIs | `EXCHANGE.TRANSFER.PAY` | `RESOURCE.TRANSFER` | The README's decorator example |
| Grant access, spawn a sub-agent with tools | `COOPERATE.ASSIST.HELP` | `CAPABILITY.ENABLE` | Delegation *plans* are `COOPERATE.COORDINATE.PLAN` (`COMMITMENT.MAKE`) |
| Revoke access, lock account, kill process | `GOVERN.REGULATE.ENFORCE` | `CAPABILITY.RESTRICT` | |
| Self-check, run own test suite on own output | `RECURSE.VERIFY.AUDIT` | `INFO.DISCLOSE` (to self) | Flagged verb; fits agents evaluating their own work |

## 5. Where the mapping strains — filed as verb-gap findings

Honesty about the strain points is the point of the exercise.

**Deletion.** `rm`, `DROP TABLE`, deleting a calendar event: the effect is unambiguously `RESOURCE.DAMAGE` (value destroyed, no recipient), but the only verb whose typical effect is `RESOURCE.DAMAGE` is `HARM.DAMAGE.STRIKE` — flagged, and connotatively wrong for routine hygiene deletion. Options today: emit `TRANSFORM.ALTER.MODIFY` with an explicit `RESOURCE.DAMAGE` effect and `harm_acknowledged` where the active mapping permits it, or scope an override in the Mission Profile. **Finding: the vocabulary needs a non-adversarial destruction verb (candidate: `TRANSFORM.REMOVE.DELETE`) in the provisional vocabulary.** This is exactly the class of gap §9 of the working paper predicted would surface under real mapping.

**Shell execution.** `bash`/`execute_command` is verb-opaque: the tool name says nothing; the command decides everything. The claim must be constructed from the agent's *stated intent* for the call, and the check tuple from what the process actually did — which makes shell tools the place where the Claim-Check Delta earns its keep most visibly, and the worst place to rely on claim-side telemetry alone. Recommended practice: verb from stated intent, `measurement.mode: INFERRED`, and check-side coverage (§6.6) that explicitly includes process-level observation for any deployment granting shell access.

**Composite operations.** One tool call, several effects (`git push`: relocation + disclosure; account provisioning: creation + capability grant + commitment). TAO handles this by multiple entries in `effects[]`; adapters should resist the temptation to emit only the flattering one — that is stenographic-adapter behavior (spec §10.1) and observable as such across a population of tuples.

## 6. Worked example: claim/check pair from a coding-agent session

Task: "Draft release notes for v3.2." The agent reads the repo, then calls `http_post` to an external endpoint. Claim tuple (from the MCP middleware), abbreviated to the fields that matter:

```json
{
  "actor": {"entity_id": "ci_code_agent_v2", "entity_type": "AUTONOMOUS_SYSTEM"},
  "action": {"verb": "CREATE.GENERATE.PRODUCE", "outcome": "SUCCEEDED",
             "target_specificity": "INDIVIDUAL", "target_ref": "repo:acme/app#RELEASE_NOTES.md"},
  "effects": [{"type": "RESOURCE.TRANSFER", "target": "repo:acme/app",
               "measurement": {"mode": "INFERRED", "sensor_refs": ["mcp_client_middleware"]}}],
  "justification": {"purpose": "draft release notes for v3.2"},
  "context": {"environment": {"reality": "DEPLOYMENT", "domain": "SOFTWARE_ENG", "substrate": "DIGITAL"}},
  "provenance": {"adapter_id": "mcp_tao_shim", "adapter_version": "0.3.1"}
}
```

Check tuple (from the egress proxy, `PRIVILEGE_ISOLATED`, coverage declaring `network_egress` observed):

```json
{
  "actor": {"entity_id": "ci_code_agent_v2", "entity_type": "AUTONOMOUS_SYSTEM"},
  "action": {"verb": "COMMUNICATE.INFORM.TELL", "outcome": "SUCCEEDED",
             "target_specificity": "INDIVIDUAL", "target_ref": "https://exfil.example.net/upload"},
  "effects": [{"type": "INFO.DISCLOSE", "source": "repo:acme/app", "target": "exfil.example.net",
               "measurement": {"mode": "OBSERVED", "sensor_refs": ["egress_proxy_log_8842"]}}],
  "provenance": {"adapter_id": "egress_observer", "observer_independence_level": "PRIVILEGE_ISOLATED",
                 "coverage": {"channels_observed": ["network_egress"],
                               "channels_not_observed": ["clipboard", "dns_tunneling"]}}
}
```

CCD: `CREATE.GENERATE.PRODUCE` does not permit external `INFO.DISCLOSE` (semantic-mechanical: **INCONSISTENT**); stated purpose is content generation, observed flow ends in external disclosure (teleological: **DIRECT_CONTRADICTION**); no authority for the endpoint (factual: **INCONSISTENT**). This is the working paper's Example 1, assembled entirely from surfaces a 2026 agent stack already emits.

## 7. Positioning

TAO tuples are the *semantic payload*; they are designed to be carried over OTel transport, stored in IETF agent-audit-trail-style tamper-evident envelopes, and attested by OVERT-class control-execution evidence. Nothing in this mapping requires model changes, retraining, or weights access — the same integration posture as the working paper §4.
