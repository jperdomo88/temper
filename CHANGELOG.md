# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project versioning is consistent across the spec and the reference
implementation: the validator version tracks the spec version it implements.

## [0.11.3] — 2026-05-16

Expands the Mission Profile collection from one to seven domain templates.

- `browser_agent` — for computer-use and browser-controlling agents (Claude
  computer-use, Operator, Browser Use, etc.). 19 verbs, 13 exceptions.
  3-tier escalation. Hard caps on purchases above thresholds; subscription
  enrollment always escalates.
- `customer_service` — for contact-center and support automation. 14 verbs,
  12 exceptions. 5-tier delegation. SAFE_STATE fail-safe so customers
  aren't left hanging. Vulnerability-context escalation routes to a real
  human, not another AI tier.
- `financial_services` — for AI advisors, robo-advisors, trading agents
  under FINRA / SEC / MiFID II / FCA regimes. 14 verbs, 14 exceptions.
  Performance representation BLOCKS without required disclosures.
  Cross-border transfers escalate to AML officer. P7Y retention.
- `healthcare_provider` — for clinical decision support and patient-facing
  agents under HIPAA-equivalent regimes. 15 verbs, 15 exceptions.
  Treatment defaults to ESCALATE; HARM.DAMAGE permitted only with
  acknowledged harm and explicit consent / emergency-doctrine authority.
  P10Y retention.
- `enterprise_tool_agent` — for agents with scoped access to internal
  workplace tools (Slack, Notion, Salesforce, etc.). 14 verbs, 15
  exceptions. 11-role escalation chain reflecting enterprise governance
  breadth. Cross-classification reads (board materials, legal hold) hard
  blocked.
- `education` — for AI tutors, TAs, study tools, and assessment systems
  under FERPA / COPPA. 13 verbs, 13 exceptions. Academic-integrity guard
  on COMMUNICATE.INFORM.TELL and COOPERATE.ASSIST.HELP. Biometric
  monitoring of minors escalates to principal. Proctoring requires
  explicit consent in justification.

All seven profiles validate against `tao_mission_profile.schema.json`.

`spec/mission_profiles/README.md` updated with the catalog table.

No spec or validator changes.

## [0.11.2] — 2026-05-16

Adds the first Mission Profile template — a deployable code-agent profile —
and the directory structure for future profiles.

- **`spec/mission_profiles/code_agent.json`** — schema-valid Mission Profile
  template for code agents (Claude Code, Cursor, Cline, Aider, GitHub
  Copilot Workspace, Continue, Devin, etc.). 18 verb overrides, 18 exception
  rules, 4-tier escalation chain, FAIL_CLOSED fail-safe, 1-year audit
  retention. The `authority` block is intentionally omitted; deployers sign
  before deploying.
- **`spec/mission_profiles/code_agent.md`** — human-readable companion
  walking verb-by-verb through every choice. Names three concrete failure
  modes the profile targets: exfiltration via tool composition, silent
  CI/credential modification, deceptive task framing.
- **`spec/mission_profiles/README.md`** — directory index with the
  "draft starting points pending industry validation" framing. Explicit
  about what these templates are and are not.

The profile reflects threats common in tool-using code agents today. It is
published as a working draft inviting platform engineers, security leads,
and AI-product teams to push back, refine, and contribute additional
profiles for their domains. The companion `.md` ends with five open
questions whose answers most usefully shape v0.2.

No spec changes. Tuple format, vocabulary, mapping rules, CCD pipeline,
and validator behavior are unchanged from 0.11.0.

## [0.11.1] — 2026-05-16

Adds the first end-to-end integration adapter so a developer can wire TAO into
an agent in one line.

- **`tao.adapters.tao_emit` decorator** — wraps any function with a TAO verb;
  every call emits a conformant tuple to a configurable sink. Default sink is
  stdout; production sinks (`ListSink`, `JsonlSink`, `CallableSink`) ship with
  the package. Failures produce `outcome=FAILED` tuples with empty effects and
  re-raise.
- **Effect derivation** — when no explicit effects are supplied, the decorator
  emits a minimal effect drawn from the verb's REQUIRED set in the reference
  mapping. Custom effects can be passed as a list (with `$argname` placeholder
  substitution) or as a callable receiving `(args, kwargs, result)`.
- **Module-level config via `configure_emitter`** — set the actor identity and
  sink once at startup; per-decorator overrides remain available.
- **Pytest suite** — `tests/test_adapter.py` validates every emitted tuple
  against the full pipeline. CI now runs pytest in addition to the conformance
  suite.

No spec changes. Tuple format, vocabulary, mapping rules, and CCD pipeline are
unchanged from 0.11.0.

## [0.11.0] — 2026-05-16

First public release of TAO.

### Specification

- `TAO_v0_11.md` — working-draft spec, ~18 RFC-style pages.
- Two-layer model: semantic claim + mechanical effects, linked by REQUIRED /
  FORBIDDEN / PERMITTED mapping rules.
- 29-verb vocabulary across 12 families (Appendix A); 10 verbs in a
  provisional vocabulary appendix (A.2).
- Nine mechanical effect types in four categories (resource, capability,
  information, commitment).
- Five conformance-relevant context fields plus measurement metadata.
- Claim-Check Delta with five teleological mismatch classes:
  CONSISTENT, DIRECT_CONTRADICTION, MISSING_BENEFICIARY,
  UNACKNOWLEDGED_HARM, AUTHORITY_GOAL_MISMATCH, INSUFFICIENT_INFORMATION.
- Observer-independence ladder: SAME_PROCESS → SIDECAR → PRIVILEGE_ISOLATED →
  HARDWARE_ISOLATED → INSTITUTIONALLY_INDEPENDENT.
- Mission Profile schema and override discipline (machine-readable diff,
  weakening flag, deviation report).
- Two conformance levels: TAO and TAO-Attested.
- Threat model (§10) naming six adversary classes and what the spec does
  about each.

### Companion documents

- `TAO_Semantic_Laundering_Overview.pdf` — 2-page entry point.
- `TAO_ADOPTION_BRIEF.md` — 2-page brief for PMs and platform leaders.
- `TAO_COMPLIANCE_CROSSWALK.md` — maps TAO features to EU AI Act articles,
  NIST AI RMF subcategories, ISO/IEC 42001 Annex A controls, SOC 2 Trust
  Services Criteria, and PCI DSS v4.0 Requirement 10.
- `REFERENCE_VALIDATOR_SPEC.md` — specification of the planned validator
  pipeline (this release implements it).

### Reference implementation

- `tao` Python package implementing schema validation, placeholder rejection,
  semantic-mechanical mapping enforcement, justification checks, Mission
  Profile override discipline, and the Claim-Check Delta pipeline.
- CLI: `tao validate`, `tao ccd`, `tao check-suite`.
- 21 published test vectors covering positive, negative, CCD (all five
  teleological classes plus observer-independence), and profile-override
  cases. The reference implementation passes all 21.
- GitHub Actions CI runs the test suite on Python 3.10–3.12.

### Not yet implemented

- RFC 8785 (JCS) canonical serialization and signature verification.
- Authority-chain resolution against an attested registry.
- Mission Profile signature verification.

These remain planned for v0.12 and are explicitly disclosed as offline
limitations in REFERENCE_VALIDATOR_SPEC.md §7.
