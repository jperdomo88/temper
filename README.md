# TAO — Detecting Semantic Laundering in Agentic AI

**A standard interface for comparing an agent's declared action against independently observed effects — so operators, auditors, and regulators can detect when the label on behavior stops matching the behavior itself.**

[![License: CC BY 4.0](https://img.shields.io/badge/spec-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
![Working draft](https://img.shields.io/badge/status-working%20draft%20v0.14-orange)

---

# 📄 [Read the working paper](TAO_Working_Paper.pdf)

> **The primary entry-point to this project is the 23-page working paper.** It covers the problem, the substrate, four worked examples, what TAO does *and* does not catch, the theoretical positioning, and the substrate's named failure modes. It stands on its own — you don't need to browse this repository to read it.
>
> If you only have five minutes: the [2-page overview](spec/TAO_Semantic_Laundering_Overview.pdf) carries the hook, the comparison chart, and one worked scenario.
>
> If you're **not technical** — or you're deciding whether to route this to someone who is — start with the [plain-language brief](TAO_Plain_Language_Brief.pdf): two pages, no jargon, what TAO is and why it matters.

---

## The problem in one sentence

An agent reports it was drafting release notes. The trace records a read from a private repo and an HTTPS POST to an endpoint nobody authorized. Today's agent telemetry records both halves of that story. No standard layer asks whether the agent's label for the action matches what the action mechanically did. We call this gap **semantic laundering** — a benign label sitting on top of effects that don't support it.

## What this repository contains

| | |
|---|---|
| `TAO_Plain_Language_Brief.pdf` | **Start here if you're non-technical.** A 2-page, jargon-free doorway: the concept, why it matters, and what's new — for reviewers, executives, and anyone deciding whether to pass this along. |
| `TAO_Working_Paper.pdf` | **Start here if you're technical.** A 23-page working paper introducing TAO in narrative form. Linked prominently at the top of this README. |
| `spec/TAO_Semantic_Laundering_Overview.pdf` | The 2-page entry-point: the hook, comparison chart, one worked scenario. Faster than the working paper if you only have five minutes. |
| `spec/TAO_v0_11.md` | The specification proper. ~18 RFC-style pages, normative. For engineers implementing TAO. |
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
| `spec/MAPPING_MCP_OTEL.md` | **How TAO rides on real agent stacks** — worked mapping from MCP tool calls and OpenTelemetry GenAI spans to TAO tuples, with a full claim/check example from a coding-agent session. |
| `spec/EVAL_PLAN_SHADE_ARENA.md` | **Pre-registered evaluation plan** — testing CCD detection against LLM-monitor baselines on public sabotage-evaluation data. The empirical study the substrate needs next. |
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

The layers around this problem all have serious public efforts now: telemetry (OpenTelemetry GenAI), observation (AgentSight's eBPF boundary tracing), runtime policy (OPA, Microsoft's Agent Governance Toolkit), attestation that controls executed (OVERT 1.1), tamper-evident record formats (the IETF agent-audit-trail and audit-architecture drafts), and payment-scoped intent verification (AP2, Mastercard/Google Verifiable Intent). TAO competes with none of them. It defines the one object none of them define: a portable, deterministic comparison between the *semantic label* an agent put on an action and the *mechanical effects* the action produced — emitted as a structured verdict through the Claim-Check Delta. Adjacent systems observe, attest, transport, authorize, and judge; TAO specifies the comparison between them. See the working paper's §5 landscape section for the full positioning.

See `spec/TAO_Semantic_Laundering_Overview.pdf` for the comparison chart against existing work, or read the spec's §1.2 and §10 for non-goals and threat model.

## Conformance

Two conformance levels:

- **TAO** — emits valid tuples using the normative vocabulary, honors the semantic-mechanical mapping, performs CCD checks when claim/check pairs are available.
- **TAO-Attested** — adds canonical serialization (RFC 8785 JCS), tuple signatures, hash-chained append-only logs, signed Mission Profiles with override-discipline enforcement, and a minimum observer-independence level (`PRIVILEGE_ISOLATED`).

A claim of conformance includes the implementation's results on `spec/test_vectors.json`. The reference validator passes all 22 vectors.

## Status

Working draft. Vocabulary may change in the 0.x line; structural pieces (tuple format, kernel, conformance levels, override discipline) are intended to be stable across 0.x.

**Not yet implemented (targeted for v0.15):**

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

---

## Version status

| Artifact | Current version | Notes |
|---|---|---|
| Working paper | v0.14 (2026-07-11) | The primary narrative entry-point. v0.14 adds the 2025–26 landscape positioning (AgentSight, OVERT, IETF drafts, AI-control literature, payments protocols), the term disambiguation against Romanchuk & Bondar (2026), the policy-as-code section, and the CCD check tiering. |
| Normative spec | v0.12 wire-level (file: `spec/TAO_v0_11.md`) | File retains the v0.11 filename for stable links; contents include §6.6 Observation Coverage and the broadened DIRECT_CONTRADICTION classifier introduced in v0.12. |
| JSON schemas | v0.12 (`spec/tao_tuple.schema.json`, `spec/tao_mission_profile.schema.json`) | Include `provenance.coverage` declaration. |
| Reference validator | v0.13 (`tao/` Python package, `pyproject.toml` v0.13.x) | Passes all 22 conformance test vectors. |
| Test vectors | 22 vectors (`spec/test_vectors.json`) | Aligned to validator v0.13. |

A v0.14 cycle will likely re-anchor the spec filename, formalize the Behavioral Dossier as a separate analytical layer, and expand the `INSTITUTIONALLY_INDEPENDENT` rung's subdimensions.
