# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project versioning is consistent across the spec and the reference
implementation: the validator version tracks the spec version it implements.

## [0.12.0] — 2026-05-17

Addresses the substantive critiques in the first external review. Minor
version bump because of the schema addition (`provenance.coverage` on check
tuples) and the broadened CCD classifier.

### Spec changes

- **§6.6 Observation coverage (new):** Separates the existing
  independence-level ladder (where the observer runs) from a new coverage
  declaration (what channels the observer watches). A check tuple MUST now
  declare `provenance.coverage` with `channels_observed`,
  `channels_not_observed`, `observation_point`, and
  `blocked_attempts_visible`. Attested citation requires BOTH adequate
  independence AND a coverage declaration. The intent is to close the
  procurement-gaming surface where a vendor declares high independence but
  quietly narrow coverage.
- **REFERENCE_VALIDATOR_SPEC.md:** Updated to reflect current implementation
  reality. Sections now marked as Implemented or Planned. Removes the
  pre-implementation "executable code is the next planned artifact" framing
  that conflicted with the shipping code.

### Validator and adapter changes

- **DIRECT_CONTRADICTION classifier broadened.** Previously gated on
  `PROTECT.*` / `COOPERATE.*` verbs; now also flags any observed effect
  outside the verb's REQUIRED/PERMITTED envelope, with a specific path for
  `INFO.DISCLOSE` to external/unverified/unauthorized targets. This matches
  the behavior the flagship `code_agent_exfiltration.md` scenario assumes
  and makes the scenarios reproducible against the reference validator.
- **Decorator default context made honest.** Defaults changed from
  reassuring (`consent: IMPLICIT`, `vulnerability: NONE`,
  `legitimacy: VERIFIED`, `reversibility: REVERSIBLE`) to honest uncertainty
  (`consent: UNKNOWN`, `vulnerability: UNKNOWN`, `legitimacy: CLAIMED`,
  `reversibility: UNKNOWN`). The decorator runs inside the agent and cannot
  in fact verify these properties; quiet certification by default was a
  laundering vector.
- **CCD output extended.** `coverage_declared` and `coverage_summary` fields
  added to CCD output. `attested_citation_valid` now requires both
  independence ≥ PRIVILEGE_ISOLATED AND a coverage declaration.

### Test vectors

- **CCD-009 (new):** Code-agent exfiltration on `CREATE.GENERATE.PRODUCE`
  with observed external `INFO.DISCLOSE`. Locks in the broadened
  DIRECT_CONTRADICTION classifier behavior. Vector count: 21 → 22.

### Adoption playbook restructured

- **Three operational tiers introduced.** TAO-Log (decorator only),
  TAO-Check (independent observer + CCD), TAO-Governed (CCD + Mission
  Profile wired to enforcement). Stops conflating decorator-only adoption
  with audit-grade assurance.
- **"Who does what" section added.** Engineer, security, compliance,
  domain owner, procurement, incident response — each role's distinct
  ownership of the substrate.
- **Stage 2 step 3 (new):** Declare the observer's coverage. Explicit
  step in the playbook reflecting the §6.6 schema requirement.
- **Stage 2 shadow mode added:** Two-sprint shadow run before wiring CCD
  output to escalation/blocking. Replaces "graduate to CCD when an
  incident makes the case" framing.

### Stakeholder pages softened

- **PM:** Acknowledges decorator-only is TAO-Log, not audit-grade.
  Replaces "wait for an incident" framing with shadow-mode sequencing.
- **Fintech CTO:** "Does not satisfy regulator by itself" — gives the
  logging substrate existing supervisory controls need.
- **Healthcare compliance:** TAO logs may themselves be regulated
  records; inherit existing access controls.
- **Regulator:** Substrate reduces reconstruction cost; does not solve
  evidentiary sufficiency by itself.
- **Procurement:** Contract clauses now define "Covered Action"
  explicitly via Exhibit reference. Observer clause requires both
  independence and coverage declarations.
- **Safety researcher:** Vocabulary claim downgraded to "proposed
  compact controlled vocabulary"; acknowledges adjacent literature.

### New scenarios — limits exhibits

The scenarios directory now includes three exhibits showing what TAO
does *not* catch, alongside the eight catches and one CONSISTENT
baseline. A spec that only shows wins is uncalibrated.

- `limits_observer_blind_channel.md` — Agent uses DNS exfiltration;
  observer's coverage declaration excludes DNS queries. CCD CONSISTENT
  with the gap explicitly recorded.
- `limits_tool_result_poisoning.md` — Honest agent relays a falsified
  tool response. CCD cannot, by design, distinguish honest relay of
  correct vs. compromised dependency state.
- `limits_multi_step_laundering.md` — Three individually CONSISTENT
  actions compose into a re-identification chain. Pairwise CCD does
  not provide episode-level analysis; explicit non-goal of v0.x.

### README positioning sharpened

The lead now reads: "A standard interface for comparing an agent's
declared action against independently observed effects — so operators,
auditors, and regulators can detect when the label on behavior stops
matching the behavior itself." Replaces the previous "behavioral audit
interface" phrasing.

## [0.11.4] — 2026-05-16

Completes the conceptual package: eight worked CCD scenarios, six stakeholder
one-pagers, and a three-stage adoption playbook. The aim is a repository that
a team can drop into an AI assistant and ask "help us adopt this" — with every
artifact the assistant needs already present.

- **`spec/scenarios/`** — eight worked CCD scenarios, each a complete chain
  of claim tuple, check tuple, CCD output, Mission Profile decision, what a
  reviewer sees later, and why CCD caught the case:
  - `code_agent_exfiltration.md` — DIRECT_CONTRADICTION
  - `browser_agent_subscription.md` — AUTHORITY_GOAL_MISMATCH
  - `customer_service_deflection.md` — MISSING_BENEFICIARY
  - `financial_unauthorized_trade.md` — AUTHORITY_GOAL_MISMATCH
  - `healthcare_off_scope_advice.md` — UNACKNOWLEDGED_HARM
  - `enterprise_wrong_channel.md` — DIRECT_CONTRADICTION
  - `education_cheating_assist.md` — DIRECT_CONTRADICTION
  - `code_agent_consistent_baseline.md` — CONSISTENT (calibration baseline)

  The baseline is intentional: reviewers calibrate against clean runs as well
  as failures.

- **`spec/stakeholders/`** — six one-pagers translating the same substrate
  into local idioms:
  - `pm_frontier_lab.md` — for a frontier-lab product manager
  - `cto_fintech.md` — for a fintech CTO
  - `compliance_healthcare.md` — for a healthcare compliance officer
  - `regulator.md` — for a regulator or standards body
  - `procurement_enterprise.md` — for enterprise procurement and risk
  - `safety_researcher.md` — for a safety researcher

  Each answers one question that audience would actually ask, in language
  they already use, in roughly a single screen.

- **`spec/ADOPTION_PLAYBOOK.md`** — three-stage adoption guide:
  Stage 1 (drop in the decorator), Stage 2 (add an independent observer),
  Stage 3 (write a Mission Profile). Ends with a "drop into Claude" prompt
  template that operationalizes the assumed adoption pattern: hand the
  repository to an AI assistant and ask it to help with integration.

- README updated with the expanded directory layout.

No spec, schema, or validator changes. Tuple format, vocabulary, mapping
rules, CCD pipeline, and Mission Profile schema are unchanged from 0.11.0.

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
