# TAO — Detecting Semantic Laundering in Agentic AI

**A standard interface for comparing an agent's declared action against independently observed effects — so operators, auditors, and regulators can detect when the label on behavior stops matching the behavior itself.**

[![License: CC BY 4.0](https://img.shields.io/badge/spec-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
![Working draft](https://img.shields.io/badge/status-working%20draft%20v0.12-orange)

---

## The problem in one sentence

An agent reports it was drafting release notes. The trace records a read from a private repo and an HTTPS POST to an endpoint nobody authorized. Today's agent telemetry records both halves of that story. No standard layer asks whether the agent's label for the action matches what the action mechanically did. We call this gap **semantic laundering** — a benign label sitting on top of effects that don't support it.

## What this repository contains

| | |
|---|---|
| `spec/TAO_Semantic_Laundering_Overview.pdf` | **Start here.** Two pages. The hook, the comparison chart, the worked scenario. |
| `spec/TAO_v0_11.md` | The specification. ~18 RFC-style pages, normative. |
| `spec/TAO_ADOPTION_BRIEF.md` | Two-page brief for PMs and platform leaders with a five-phase pilot path. |
| `spec/ADOPTION_PLAYBOOK.md` | Three-stage practical guide for teams integrating TAO into an existing deployment, plus a "drop into Claude" prompt template. |
| `spec/TAO_COMPLIANCE_CROSSWALK.md` | Maps TAO features to EU AI Act articles, NIST AI RMF, ISO/IEC 42001, SOC 2, and PCI DSS. |
| `spec/TAO_REGULATORY_THEORY.md` | Positions TAO in regulatory-theory terms — policy-instrument taxonomies, management-based regulation, the audit-society literature, and current AI-governance practice. For regulatory scholars and agency policy staff. |
| `spec/REFERENCE_VALIDATOR_SPEC.md` | Specification of the validator implemented in this repo. |
| `spec/tao_tuple.schema.json` | JSON Schema for the action tuple. |
| `spec/tao_mission_profile.schema.json` | JSON Schema for Mission Profiles. |
| `spec/test_vectors.json` | 22 conformance test vectors (positive, negative, CCD, profile override). |
| `spec/mission_profiles/` | **Mission Profile starting points** for code agents, browser agents, customer service, financial services, healthcare, enterprise tools, and education. Draft starting points pending industry validation. |
| `spec/scenarios/` | **Worked CCD scenarios.** Eight concrete claim/check pairs across domains, each illustrating one failure mode (or, for the baseline, a clean run). |
| `spec/stakeholders/` | **One-pagers for specific readers** — frontier-lab PM, fintech CTO, healthcare compliance officer, regulator, enterprise procurement, safety researcher. Same substrate, audience-local language. |
| `tao/` | Reference validator and `tao_emit` decorator (Python). |

## Quick start

```bash
pip install -e .
tao check-suite spec/test_vectors.json
```

Expected output: **22 / 22 vectors passed.** That's the canonical evidence that an implementation conforms.

### Emit TAO tuples from your agent in one line

The `tao_emit` decorator turns any function into a TAO emission point:

```python
from tao.adapters import tao_emit, configure_emitter, JsonlSink

configure_emitter(
    actor={"entity_id": "support_agent_v3", "entity_type": "AUTONOMOUS_SYSTEM"},
    sink=JsonlSink("audit.jsonl"),
)

@tao_emit("EXCHANGE.TRANSFER.PAY",
          target="$customer_id",
          context={"environment": {"domain": "RETAIL"}})
def refund(customer_id, amount):
    # ... real payment logic
    return {"refunded": amount}

refund(customer_id="cust_88241", amount="29.99")
# A TAO tuple lands in audit.jsonl with verb, target, effects, context,
# outcome, timestamp. If the function raises, the tuple is emitted with
# outcome=FAILED and the exception re-raised.
```

The decorator emits valid tuples by default. Override effects, target, justification, or context as needed (see `tests/test_adapter.py` for a full set of patterns).

### Validate or compare tuples from the CLI

```bash
tao validate spec/examples/my_tuple.json
tao ccd claim.json check.json --observer-level PRIVILEGE_ISOLATED
```

## Where TAO fits

TAO is not a telemetry layer (OpenTelemetry GenAI handles that), not a policy engine (OPA and Microsoft's Agent Governance Toolkit do that at runtime), and not an audit-trail format competing with the IETF Agent Audit Trail draft. TAO operates one layer above raw telemetry: it records whether the *semantic label* on an action matches the *mechanical effects* the action produced, and emits a structured comparison through the Claim-Check Delta.

See `spec/TAO_Semantic_Laundering_Overview.pdf` for the comparison chart against existing work, or read the spec's §1.2 and §10 for non-goals and threat model.

## Conformance

Two conformance levels:

- **TAO** — emits valid tuples using the normative vocabulary, honors the semantic-mechanical mapping, performs CCD checks when claim/check pairs are available.
- **TAO-Attested** — adds canonical serialization (RFC 8785 JCS), tuple signatures, hash-chained append-only logs, signed Mission Profiles with override-discipline enforcement, and a minimum observer-independence level (`PRIVILEGE_ISOLATED`).

A claim of conformance includes the implementation's results on `spec/test_vectors.json`. The reference validator passes all 21 vectors.

## Status

Working draft. Vocabulary may change in the 0.x line; structural pieces (tuple format, kernel, conformance levels, override discipline) are intended to be stable across 0.x.

**Not yet implemented (planned for v0.12):**

- RFC 8785 canonical serialization and signature verification
- Authority-chain resolution against an attested registry
- Mission Profile signature verification

These are explicit limitations disclosed in `spec/REFERENCE_VALIDATOR_SPEC.md` §7.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most useful contributions right now are:

- Verb-gap or effect-gap issues — concrete cases where the kernel or vocabulary fails on a real action class you've shipped.
- Adversarial Mission Profiles — examples that pass §7.3 but launder something the discipline should catch.
- Additional test vectors — concrete claim/check pairs with expected CCD output.

Hard critique is the most valuable thing you can offer. The author is an outside systems builder looking for the kind of objections a senior reader sees in five minutes that he would miss after months in it.

## License

- Specification text (everything in `spec/` other than `.json`) — CC BY 4.0.
- Code, schemas, and test vectors — Apache License 2.0.

See [LICENSE](LICENSE) for the full text.

## Author and contact

Jorge Perdomo · jorgeperdom@gmail.com

This repository is the public form of a working draft I have been developing on
the side. I am not presenting it as a finished standard. Feedback,
collaborators, and adversarial review are all welcome.
