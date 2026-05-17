# TAO

## A Behavioral Audit Interface for Autonomous Systems

**Version:** 0.11 (Draft)
**Status:** Working Draft for Public Review
**Date:** 2026-05-16

---

## Abstract

TAO is a behavioral audit interface for autonomous systems. The spec defines a tuple that records what an agent did, what mechanical effects the action produced, in what context, under whose authority, and whether the agent's own account of the action is consistent with an independent observation of it. TAO does not specify what policies an autonomous system should follow. It specifies a structure in which such policies can be named, signed, enforced, and verified.

The contribution is a clean separation between *semantic claims* about an action ("this was a refund") and *mechanical observations* of an action (a $30 value transfer occurred), plus a consistency check (the Claim-Check Delta) that distinguishes coherent behavior from semantic laundering. The intended audience is engineers building agentic systems, auditors verifying them, and regulators who need machine-readable records of behavior without access to model internals.

A JSON Schema, a reference validator outline, and a test vector suite are published alongside this document.

---

## 1. Introduction

### 1.1 What TAO is

TAO is three artifacts:

1. A schema for an *action tuple*: a record of one observable thing an agent did.
2. A compact vocabulary of verbs (29 verbs in 12 families in the normative core, with a separate provisional vocabulary in Appendix A.2) and effects (9 mechanical types in 4 categories) that the tuple references.
3. A consistency check (the Claim-Check Delta) that compares a tuple's claims about an action against what an independent observer recorded.

A deployment MAY attach a signed *Mission Profile* (§7) that names which verbs are allowed, blocked, escalated, or logged under which conditions.

### 1.2 What TAO is not, and what others can build on top

TAO is not a moral framework, a training objective, a runtime architecture, or a regulator. The spec embeds technical commitments — that effects can be measured, that observation must be independent from the observed, that some side effects require named acknowledgment, that policy choices should be signed and inspectable — but it does not embed substantive policy content. A deployment can use TAO to enforce a permissive policy, a restrictive policy, or no policy at all. The audit trail makes whatever choice was made inspectable after the fact.

Things adjacent to TAO that other documents and other people will need to write: domain-specific Mission Profiles (medical, financial, automotive, defense), formal verification of adapter correctness, ground-truth tooling for telemetry capture, and the legal and procurement frameworks that actually compel compliance. The spec defines the data layer those efforts can stand on.

### 1.3 Conventions

The keywords MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, REQUIRED, and OPTIONAL are interpreted as in [RFC 2119].

Numeric values in tuples are encoded as JSON strings containing decimal representations (see §3.4). Timestamps are ISO 8601 with required `Z` suffix and millisecond precision (see §5.4).

### 1.4 Document structure

Sections 2 through 4 introduce the two-layer model and define its two layers (the mechanical kernel and the semantic vocabulary). Section 5 specifies the tuple. Section 6 specifies the Claim-Check Delta and observer independence. Section 7 specifies Mission Profiles, including the override discipline. Section 8 defines conformance and the reference implementation. Section 9 covers versioning and extensions. Section 10 states the threat model. Appendices A and B publish the full vocabulary and reference mappings; Appendix A.2 lists provisional verbs that are not part of the normative core.

### 1.5 Limitations

The spec is published with the following limitations acknowledged in plain text rather than discovered by reviewers.

**Kernel lossiness.** The nine mechanical effects are intentionally coarse. RESOURCE.TRANSFER alone covers payments, medication delivery, file generation, physical movement, and resource consumption. The kernel is a lowest-common-denominator audit categorization, not a complete physical description of what happened. Richness lives in the semantic verb, the context fields, and registered extensions. A deployment that needs a finer-grained mechanical layer registers an extension; the spec does not pretend the kernel is sufficient on its own.

**Vocabulary coverage is constructed, not proved.** The verb and effect tables are the product of design judgment, not an exhaustiveness theorem. A future version may add, retire, or reorganize verbs based on usage; provisional verbs in Appendix A.2 may be promoted or removed entirely.

**Some effects are observable only by inference.** Psychological manipulation, epistemic effects of persuasion, and downstream cascading impacts cannot be directly measured. The spec marks such effects as `INFERRED` (§3.3) and requires adjudication status; deployments that can't get past `PENDING` adjudication for high-stakes inferred effects should not claim TAO-Attested for those actions.

**Adapters and observers are trusted code.** The spec defines a behavioral audit interface, not a sandboxing system. If both the agent's adapter and the independent observer are compromised by the same party, TAO records what they jointly choose to record. The observer-independence ladder (§6.5) is the mitigation, but it depends on real organizational separation that no schema can enforce.

**Teleological consistency is partial.** CCD compares stated goals to observed effects under a small taxonomy of mismatch classes (§6.2). The spec deliberately does not claim to evaluate "intent." It claims to detect specific structural inconsistencies between what was claimed and what was observed.

**What would falsify the spec's claimed value.** TAO would be shown not to deliver its claimed value if (a) producing check tuples turned out to be uneconomic at scale; (b) independent observers proved capturable in practice such that every deployment's observer ended up controlled by the actor; (c) the core vocabulary turned out to be too coarse for any deployment to use without an extension, making the core effectively empty; or (d) adversarial adapters could trivially emit compliant-looking but uninformative tuples that pass validation but disclose nothing useful. The threat model (§10) names mitigations for each, but the empirical test is deployment.

---

## 2. The Two-Layer Model

A TAO tuple describes an action in two registers at once, and links them by a constraint.

The **mechanical layer** is a list of state changes the action caused. Each state change has a type from a fixed set of nine (§3.1). These types are intended to be objectively observable: a resource moved, a capability was restricted, information was disclosed, a commitment was made.

The **semantic layer** is a single verb naming what the action was, drawn from the vocabulary in §4. Verbs are human-interpretable: PROTECT.HEAL.TREAT, EXCHANGE.TRANSFER.PAY, COMMUNICATE.INFORM.TELL.

The layers are linked by a mapping. For each verb the spec names a set of effects that MUST be present, a set that MUST NOT be present, and a set that MAY be present as acknowledged side effects (§4.6, Appendix B). The mapping is the spec's primary integrity property. A tuple that claims a verb whose required effects are absent is rejected. A tuple that claims a verb in the presence of a forbidden effect is rejected. A tuple that includes a permitted RESOURCE.DAMAGE side effect without naming it in `justification.harm_acknowledged` is rejected.

The motivating example is anti-laundering. A surgical incision causes RESOURCE.DAMAGE in the mechanical layer and is named PROTECT.HEAL.TREAT in the semantic layer. The mapping permits RESOURCE.DAMAGE as a side effect of PROTECT.HEAL.TREAT when the tuple includes `harm_acknowledged`. The same RESOURCE.DAMAGE without a therapeutic transfer, or without acknowledgment, cannot be named PROTECT.HEAL.TREAT under any mapping. The label is not a free choice once the effects are recorded.

---

## 3. Mechanical Kernel

The mechanical kernel is the set of effect types an action may produce. It is intentionally minimal.

### 3.1 Effect types [MUST]

Every effect in a tuple MUST be classified as exactly one of the following nine types, grouped into four categories:

| Category | Type | Definition |
|---|---|---|
| Resource | RESOURCE.TRANSFER | Value moved from a source to a target. |
| Resource | RESOURCE.DAMAGE | Value destroyed (no recipient). |
| Capability | CAPABILITY.RESTRICT | The target's set of possible actions was reduced. |
| Capability | CAPABILITY.ENABLE | The target's set of possible actions was expanded. |
| Information | INFO.DISCLOSE | Information was revealed to the target. |
| Information | INFO.WITHHOLD | Information was hidden from the target. |
| Information | INFO.FABRICATE | False information was injected to the target. |
| Commitment | COMMITMENT.MAKE | A promise or contract was registered. |
| Commitment | COMMITMENT.BREAK | A registered commitment was violated. |

The kernel is a coarse audit categorization, not a complete description. RESOURCE.TRANSFER captures any move of value: a payment, a medication dose, a file write, a unit relocation, a resource consumed by an actor's own operation. The spec relies on the semantic verb, the context fields, and registered extensions to carry domain-specific richness. A deployment that needs a finer mechanical layer (for example, distinguishing kinetic from informational transfers in a safety-critical domain) registers an extension under §9.2.

Observation and sensing are recorded as INFO.DISCLOSE with the observed entity as `source` and the observer as `target`. The observer's knowledge state changes; that is the disclosed information.

If an action is attempted but produces no measurable effect, the tuple sets `action.outcome = "FAILED"` and `effects` MAY be empty (see §4.4).

### 3.2 Effect object [MUST]

Each effect is a JSON object with the following shape:

```json
{
  "type": "RESOURCE.TRANSFER",
  "target": "patient_4521",
  "source": "medication_dispenser_12",
  "amount": "1.0",
  "unit": "dose",
  "measurement": {
    "mode": "OBSERVED",
    "confidence": "0.99",
    "sensor_refs": ["dispenser_12"]
  }
}
```

Required fields are `type`, `target`, and `measurement`. Optional fields are `source`, `amount`, and `unit`.

The `target` field MUST contain a non-empty entity identifier. The string values `unspecified`, `unknown`, `undefined`, `null`, `none`, `n/a`, `tbd`, `todo`, and `placeholder` are reserved and MUST be rejected by validators (case-insensitive). An effect without an identifiable target indicates an incomplete adapter; the tuple SHOULD NOT be emitted.

### 3.3 Measurement [MUST]

Every effect MUST include a `measurement` block:

```json
{
  "mode": "OBSERVED",
  "confidence": "0.95",
  "sensor_refs": ["sensor_001"],
  "adjudication_status": "CONFIRMED"
}
```

`mode` is `OBSERVED` or `INFERRED`. `OBSERVED` effects MUST list at least one sensor reference. `INFERRED` effects MUST carry an `adjudication_status` of `PENDING`, `CONFIRMED`, or `DISPUTED`.

`confidence` is a string decimal in the range `"0.0"` to `"1.0"`.

A deployment claiming TAO-Attested for an action MUST NOT rely on `INFERRED` effects with `adjudication_status = PENDING` to satisfy the verb's REQUIRED set. Inferred effects can be present and recorded; they cannot anchor the integrity check by themselves.

### 3.4 Numeric encoding [MUST]

All numeric values in a tuple MUST be encoded as JSON strings containing a decimal representation. This includes `amount`, `confidence`, and any numeric field in extension effects.

```
Correct:   "amount": "500.00"
Incorrect: "amount": 500.00
```

The rationale: JSON numeric parsing varies across implementations. String decimals produce byte-identical canonical serializations regardless of language or platform, which is what makes signatures verifiable across systems.

---

## 4. Semantic Vocabulary

The semantic vocabulary names actions in human-readable terms. It is finite, hierarchical, and extensible by registered namespace.

### 4.1 Naming [MUST]

Every verb is a three-level path:

```
FAMILY.GENUS.SPECIES
```

For example, `HARM.DAMAGE.STRIKE`, `PROTECT.HEAL.TREAT`, `COMMUNICATE.INFORM.TELL`. The hierarchy supports coarse filters (block all `HARM.*`), medium-grained rules (escalate `HARM.DECEIVE.*`), and fine exceptions (allow `HARM.DAMAGE.STRIKE` for `institutional_role.actor_role = SURGEON` with explicit consent).

Verbs MUST be uppercase. Dots separate levels. No other punctuation is permitted in a verb token.

### 4.2 Families and verbs

The normative vocabulary defines 29 verbs across 12 families. The families are:

```
HARM        PROTECT     COOPERATE   GOVERN
EXCHANGE    CREATE      TRANSFORM   COMMUNICATE
OBSERVE     BOND        SEPARATE    RECURSE
```

The full normative verb table is in Appendix A. Verbs not part of the normative core are listed in Appendix A.2 (Provisional Vocabulary) and are explicitly not required for conformance.

### 4.3 Target specificity [MUST]

Every tuple MUST declare the scope of its target:

| Value | Definition |
|---|---|
| `INDIVIDUAL` | One identified entity. |
| `GROUP` | A named collection of fewer than 100 members. |
| `CLASS` | A category of 100 or more potential members. |
| `UNBOUND` | No specific target; affects any entity in range. |

### 4.4 Outcome [MUST]

Every tuple MUST declare an outcome:

| Value | Effects array |
|---|---|
| `SUCCEEDED` | MUST contain at least one effect. |
| `FAILED` | MAY be empty. The verb describes what was attempted. |
| `PARTIAL` | Contains the effects that did occur. |

### 4.5 Flagged verbs

Six verbs are flagged for additional scrutiny by policy layers. The flag is metadata on the verb itself. The spec does not mandate any particular response; Mission Profiles (§7) specify what flagged verbs trigger.

| Verb | Flag rationale |
|---|---|
| `HARM.DAMAGE.STRIKE` | Physical or material damage to target. |
| `HARM.COERCE.THREATEN` | Intimidation. |
| `HARM.DECEIVE.LIE` | Deliberate falsehood. |
| `COMMUNICATE.OBFUSCATE.CONFUSE` | Deliberate confusion. |
| `EXCHANGE.CORRUPTION.BRIBE` | Illegitimate inducement. |
| `RECURSE.VERIFY.AUDIT` | Self-examination with modification risk. |

`GOVERN.AUTHORITY.OBEY` and `GOVERN.AUTHORITY.DISOBEY` were flagged in earlier drafts. They are unflagged in v0.11; whether obedience or disobedience is the suspect choice is entirely context-dependent and is properly a Mission Profile concern rather than a spec-level flag.

### 4.6 Semantic-mechanical mapping [MUST]

Every normative verb has a mapping of three sets:

- `REQUIRED`: at least one effect from this set MUST be present.
- `FORBIDDEN`: no effect from this set MAY be present.
- `PERMITTED`: effects from this set MAY be present and MUST be acknowledged in `justification.harm_acknowledged` if they are of type RESOURCE.DAMAGE.

The reference mapping is in Appendix B. A Mission Profile MAY supply an alternative mapping subject to the override discipline in §7.3.

A validator rejects a tuple in which:

- No effect from the verb's `REQUIRED` set is present.
- Any effect from the verb's `FORBIDDEN` set is present.
- Any effect outside the union of `REQUIRED` and `PERMITTED` is present (in the absence of an extension that adds it).
- A `PERMITTED` RESOURCE.DAMAGE effect is present without `justification.harm_acknowledged`.

### 4.7 Extensions

Extensions namespace additional verbs and effects beyond the core. Syntax and registration are specified in §9.2.

### 4.8 Provisional vocabulary

Appendix A.2 lists verbs that are useful for some broader behavioral taxonomies but are not stable or operational enough to require for conformance. Provisional verbs MAY be used; their mappings (where defined) are informative; their inclusion in a tuple does not by itself create a conformance gap. Future versions MAY promote provisional verbs to the normative core or remove them.

---

## 5. The Tuple

The TAO tuple is the unit of behavioral record.

### 5.1 Minimal example

```json
{
  "tuple_id": "7f3a9c2e-8d4b-4f6a-9c1e-2b3d4e5f6a7b",
  "schema_version": "0.11.0",
  "timestamp": "2026-05-16T15:30:00.000Z",
  "actor": {
    "entity_id": "support_agent_v3",
    "entity_type": "AUTONOMOUS_SYSTEM"
  },
  "action": {
    "verb": "EXCHANGE.TRANSFER.PAY",
    "outcome": "SUCCEEDED",
    "target_specificity": "INDIVIDUAL",
    "target_ref": "customer_88241"
  },
  "effects": [
    {
      "type": "RESOURCE.TRANSFER",
      "target": "customer_88241",
      "source": "merchant_account",
      "amount": "29.99",
      "unit": "USD",
      "measurement": {
        "mode": "OBSERVED",
        "confidence": "1.0",
        "sensor_refs": ["payment_processor_log"]
      }
    }
  ],
  "context": {
    "environment": {"reality": "DEPLOYMENT", "domain": "RETAIL", "substrate": "DIGITAL"},
    "consent": {"status": "EXPLICIT", "evidence_ref": "refund_request_551"},
    "vulnerability": {"level": "NONE"},
    "projected_impact_scope": "LOCAL",
    "reversibility": {"level": "REVERSIBLE"},
    "institutional_role": {"actor_role": "SUPPORT_AGENT", "legitimacy": "VERIFIED"},
    "temporal": {"urgency": "ROUTINE"}
  },
  "provenance": {
    "adapter_id": "retail_support_adapter",
    "adapter_version": "1.2.0",
    "adapter_hash": "sha256:..."
  }
}
```

### 5.2 Required fields [MUST]

| Field | Requirement |
|---|---|
| `tuple_id` | UUID v4, globally unique. |
| `schema_version` | TAO version string (e.g., `"0.11.0"`). |
| `timestamp` | ISO 8601 with `Z` suffix and millisecond precision. |
| `actor` | Entity performing the action. |
| `action` | Verb, outcome, target_specificity, target_ref. |
| `effects` | Array; MAY be empty only if `action.outcome = "FAILED"`. |
| `context` | Context object (§5.5). |
| `provenance` | Adapter identification. |

The following fields are conditionally required:

| Field | When required |
|---|---|
| `justification` | When `action.verb` is flagged (§4.5), or when the verb's mapping permits a RESOURCE.DAMAGE side effect. |
| `tuple_signature` | For TAO-Attested conformance (§8.2). |

### 5.3 Canonical serialization [MUST for TAO-Attested]

For signed tuples, serialization MUST follow RFC 8785 (JSON Canonicalization Scheme). Keys are sorted lexicographically at all levels, whitespace is removed, and numeric values are emitted as the shortest decimal that round-trips. The signature covers the canonical form.

### 5.4 Timestamps [MUST]

Timestamps MUST be ISO 8601, UTC, with the `Z` suffix and at least millisecond precision.

```
Correct:   "2026-05-16T15:30:00.000Z"
Incorrect: "2026-05-16T15:30:00+00:00"
Incorrect: "2026-05-16T15:30:00"
```

### 5.5 Context [MUST]

The `context` object describes the circumstances under which the action occurred. Context MUST be populated from system-attested sources (sensors, institutional records, environment metadata), not from agent claims. If a field cannot be determined, its value is `UNKNOWN`; policies may treat `UNKNOWN` as an escalation trigger.

| Field | Values |
|---|---|
| `environment.reality` | `TRAINING`, `EVALUATION`, `DEPLOYMENT` |
| `environment.domain` | A registered domain identifier (e.g., `MEDICAL`, `RETAIL`, `FINANCE`). |
| `environment.substrate` | `PHYSICAL`, `DIGITAL`, `MIXED` |
| `consent.status` | `EXPLICIT`, `IMPLICIT`, `ABSENT`, `COERCED`, `UNKNOWN` |
| `vulnerability.level` | `NONE`, `LOW`, `MODERATE`, `HIGH`, `CRITICAL`, `UNKNOWN` |
| `projected_impact_scope` | `LOCAL`, `REGIONAL`, `GLOBAL`, `EXISTENTIAL` |
| `reversibility.level` | `TRIVIAL`, `REVERSIBLE`, `COSTLY`, `IRREVERSIBLE`, `UNKNOWN` |
| `institutional_role.actor_role` | Domain-specific role identifier. |
| `institutional_role.legitimacy` | `VERIFIED`, `CLAIMED`, `AMBIGUOUS`, `ILLEGITIMATE` |
| `temporal.urgency` | `ROUTINE`, `ELEVATED`, `URGENT`, `EMERGENCY`, `UNKNOWN` |

### 5.6 Justification

When required (per §5.2), `justification` MUST contain `purpose.stated_goal` (free-text string), `authority_chain` (array of authority references that MUST resolve to attested records), and `harm_acknowledged` (string description of any permitted side effects). All other justification fields are OPTIONAL.

Validators resolve `authority_chain` entries against attested records. The contents of `stated_goal` and `harm_acknowledged` are strings and are not verified by the spec; downstream policy or human review verifies them.

### 5.7 Provenance [MUST]

The `provenance` object identifies the adapter that produced the tuple:

| Field | Definition |
|---|---|
| `adapter_id` | Stable identifier. |
| `adapter_version` | Semantic version. |
| `adapter_hash` | SHA-256 of the adapter binary or source bundle, as `sha256:HEX`. |
| `profile_hash` | OPTIONAL; SHA-256 of the active Mission Profile when one is in force. |
| `observer_independence_level` | OPTIONAL; level (§6.5) at which any paired check tuple is generated. |

### 5.8 Revisions

Tuples are immutable once emitted. Post-hoc adjudication or correction creates a new tuple with `revision_of` set to the original tuple's `tuple_id`, plus `revision_reason` and `revision_authority`. The revision chain is append-only.

---

## 6. Claim-Check Delta

The Claim-Check Delta (CCD) compares two tuples that describe the same underlying action: a *claim* tuple emitted by the agent's adapter, and a *check* tuple emitted by an independent observer. CCD reports whether they are consistent with each other and with the spec's mapping rules.

### 6.1 Inputs

CCD takes a pair `(claim_tuple, check_tuple)` referring to the same underlying action. The check tuple is constructed by an observer whose independence from the actor is characterized by §6.5. The independence level of the observer is a property of the deployment, not of CCD itself; CCD reports the level it was given.

### 6.2 Checks [MUST]

A conformant CCD implementation MUST perform three checks and report the result of each.

**(1) Semantic-mechanical alignment.** Apply the verb's REQUIRED/FORBIDDEN/PERMITTED rules from the active mapping (Appendix B or Mission Profile) to the union of the claim's and check's effects. If the rules are violated, the result is `INCONSISTENT`.

**(2) Teleological consistency.** Compare the claim's `justification.purpose` to the check's effects. The detection method is implementation-defined; the *result categories* are not. A conformant implementation classifies the teleological result as exactly one of:

| Category | Definition |
|---|---|
| `CONSISTENT` | The stated goal is compatible with the observed effects. |
| `DIRECT_CONTRADICTION` | The stated goal and the observed effects are in clear opposition. *Example: claim is "PROTECT.DEFEND.OTHER" with stated goal "back up critical data"; observed effect is INFO.DISCLOSE to an external endpoint not named in the authority chain.* |
| `MISSING_BENEFICIARY` | The stated beneficiary never receives the required beneficial effect. *Example: claim is "PROTECT.HEAL.TREAT" with stated beneficiary `patient_4521`; no RESOURCE.TRANSFER or CAPABILITY.ENABLE to that beneficiary is observed.* |
| `UNACKNOWLEDGED_HARM` | A material harmful side effect appears that the justification does not acknowledge. *Example: PROTECT.HEAL.TREAT with observed RESOURCE.DAMAGE and no `harm_acknowledged` entry.* |
| `AUTHORITY_GOAL_MISMATCH` | The stated purpose exceeds the scope of the cited authority. *Example: authority chain authorizes "refund up to $50"; the action transfers $500 with stated goal "process refund per policy 5.2".* |
| `INSUFFICIENT_INFORMATION` | The result cannot be determined from the inputs available. |

The spec deliberately does not claim to evaluate intent. It claims to detect these specific structural inconsistencies.

**(3) Factual verification.** Resolve every `authority_chain` entry in the claim against attested records. Unresolved entries produce `INCONSISTENT`.

### 6.3 Output [MUST]

The output is a JSON object:

```json
{
  "ccd_result": "INCONSISTENT",
  "observer_independence_level": "SIDECAR",
  "checks": [
    {"type": "SEMANTIC_MECHANICAL", "result": "CONSISTENT"},
    {"type": "TELEOLOGICAL", "result": "DIRECT_CONTRADICTION"},
    {"type": "FACTUAL", "result": "VERIFIED"}
  ],
  "overall_confidence": "0.85"
}
```

The top-level `ccd_result` is `CONSISTENT` when every check returns `CONSISTENT` or `VERIFIED`; `INCONSISTENT` when any check returns a non-consistent class (any teleological class other than `CONSISTENT` or `INSUFFICIENT_INFORMATION`); and `INDETERMINATE` when any check returns insufficient information and no check returns inconsistency.

### 6.4 What CCD does not establish

CCD detects inconsistency; it does not attribute cause. An inconsistent result may indicate sensor error, an adapter bug, adversarial manipulation of the environment, or deception by the agent. Distinguishing among these is the function of investigation, not of CCD. The spec deliberately refuses to make the distinction from records alone because the distinction is not, in general, technically resolvable from records alone. A vendor whose practice is to interpret every CCD inconsistency as "sensor error" is gaming the standard, and the audit trail makes that practice visible across time.

### 6.5 Observer independence [MUST]

CCD's value depends on the check tuple being generated by an observer that is meaningfully separated from the actor whose claim is being checked. "Meaningfully separated" is a spectrum. The spec defines five levels. A tuple, an implementation, or a conformance claim that invokes CCD MUST name the level at which the observer operates.

| Level | Definition | Adversary the level resists |
|---|---|---|
| `SAME_PROCESS` | Observer runs in the same process as the agent. | Bugs only. Useful for development. Not audit-grade. |
| `SIDECAR` | Observer runs in a separate process under the same operator. | Adapter bugs and some forms of agent self-misreporting. Captured if the operator is hostile. |
| `PRIVILEGE_ISOLATED` | Observer runs under a different OS-level principal with restricted IPC; agent cannot read or write observer state. | A compromised agent process. Captured by operator-level compromise. |
| `HARDWARE_ISOLATED` | Observer runs on a separate hardware enclave or trust root; communicates with the host only via attested channels. | Operator-level software compromise. Captured by physical or supply-chain compromise of the enclave. |
| `INSTITUTIONALLY_INDEPENDENT` | Observer is operated by a different organization than the actor, under separate signing keys, with attestation records reviewable by a third party. | Vendor capture. The strongest level the spec defines. |

A deployment claiming TAO conformance at any level MUST name its observer independence level in the conformance statement (§8.3). Conformance is not claimed at a level the deployment does not meet.

**`INSTITUTIONALLY_INDEPENDENT` subdimensions [SHOULD].** "Different organization" admits substantial gradient. A deployment claiming this level SHOULD disclose, at minimum, the following subdimensions of independence in its conformance statement:

- *Legal independence:* the observer is operated by a legal entity distinct from the actor's, not a wholly-owned subsidiary or controlled affiliate. (A vendor-owned subsidiary fails this.)
- *Economic independence:* the observer's funding does not flow primarily from the actor's parent, and the observer's continued operation is not at the discretion of the actor's leadership. (A SOC 2 auditor paid solely by the firm being audited partially fails this.)
- *Operational independence:* the observer's personnel, deployment pipeline, and incident-response authority are not directed by the actor's organization. (A monitor whose engineers are seconded from the vendor fails this.)
- *Key-custody independence:* signing keys for the observer's check tuples are held by an entity not equivalent to the actor. (Shared key custody collapses the cryptographic distinction.)
- *Telemetry-access independence:* the observer's input streams cannot be filtered, throttled, or redacted by the actor. (Vendor-controlled feeds to an "independent" observer compromise the level.)
- *Conflict-of-interest disclosure:* known relationships between the observer and the actor (financial, personnel, technology supply chain) are disclosed in the conformance statement.

A claim of `INSTITUTIONALLY_INDEPENDENT` without disclosure of these subdimensions SHOULD be treated by reviewers as functionally `PRIVILEGE_ISOLATED` for evidentiary purposes — the level matches the spec's literal definition but does not carry the assurance the level's name implies. A future spec version may formalize these subdimensions normatively; v0.12 names them as SHOULD-level disclosure expectations.

TAO-Attested REQUIRES observer independence at level `PRIVILEGE_ISOLATED` or higher for the CCD checks to count toward attestation. CCD output produced at `SAME_PROCESS` or `SIDECAR` levels MAY be emitted; it MUST NOT be cited as evidence in a TAO-Attested conformance statement.

This section names a property of the operator's deployment that no schema can enforce. A deployment that emits TAO-Attested tuples while running its "independent observer" inside the agent's own container is non-conformant in fact; the spec gives auditors the vocabulary to say so.

### 6.6 Observation coverage [MUST]

Independence is a property of *where* the observer runs. Coverage is a property of *what the observer can see*. An observer can be `PRIVILEGE_ISOLATED` from the agent and still blind to most of the agent's effect channels: a sidecar that watches only the agent's wrapper-level tool returns is independent and uninformative. A check tuple from such an observer corroborates only the slice of behavior the observer was wired to watch.

A check tuple MUST declare its observation coverage in the `provenance.coverage` object. The declaration names the channels the observer watches and, equally importantly, the channels it does not. A reviewer reading a CCD result cannot judge what the consistency or inconsistency means without knowing what was actually observed.

The `coverage` object has the following fields:

| Field | Type | Meaning |
|---|---|---|
| `channels_observed` | array of strings | Channels the observer watches. Recommended channel names: `tool_call_args`, `tool_call_return`, `network_egress`, `filesystem_mutation`, `process_spawn`, `ipc_send`, `ipc_receive`, `clipboard`, `browser_dom`, `database_write`, `database_read`, `api_call_params`, `wrapper_level_events`. Deployments MAY add domain-specific channel names. |
| `channels_not_observed` | array of strings | Channels the observer explicitly does not watch. An empty array asserts the observer claims coverage of all known channels for the deployment's substrate; a non-empty array names the gaps. |
| `observation_point` | string | One of `PRE_ACTION` (observed the agent's intent before the action ran), `IN_FLIGHT` (observed during execution; can interrupt), `POST_ACTION` (observed after effects landed; cannot interrupt), `WRAPPER_ONLY` (observed only the adapter's reported call, not the underlying effect). |
| `blocked_attempts_visible` | boolean | Whether the observer sees actions the agent attempted but the platform blocked, or only completed effects. |

Conformance statements MUST disclose the coverage declarations of the observers their CCD checks rely on. Coverage at independence level `PRIVILEGE_ISOLATED` or higher with `channels_not_observed` listing the high-cost effect channels (e.g., a financial agent's coverage that explicitly does not watch `network_egress`) is not a stronger evidentiary substrate than `SIDECAR` independence with full coverage. Reviewers MUST weigh both axes.

A check tuple that does not declare coverage MUST be treated as `WRAPPER_ONLY` with `channels_not_observed` unbounded. CCD output produced from such a tuple is permitted but cannot be cited in a TAO-Attested conformance statement.

The intent of this section is to remove a procurement gaming surface: a vendor that runs a perfectly isolated observer watching only the surfaces the vendor chose to expose is technically independent and substantively uninformative. Coverage declarations make that gap visible.

---

## 7. Mission Profiles

A Mission Profile is a signed, versioned, hash-anchored configuration that tells a policy layer how to act on TAO tuples in a particular deployment. The spec defines the shape of a profile, the discipline around overrides to default mappings, and the requirements on profile signing and storage. It does not define profile content for any domain.

### 7.1 Schema [SHOULD for TAO, MUST for TAO-Attested]

A Mission Profile is a JSON document of the following structure:

```json
{
  "profile_name": "string",
  "profile_version": "semver",
  "profile_hash": "sha256:HEX",
  "effective_date": "ISO-8601",
  "verb_overrides": { },
  "mapping_overrides": { },
  "mapping_diff": { },
  "escalation_chain": [ ],
  "fail_safe_mode": "FAIL_CLOSED | SAFE_STATE | DEGRADE",
  "audit_retention": "ISO-8601 duration",
  "authority": {
    "signer_id": "string",
    "signer_role": "string",
    "signature": "base64",
    "signature_algorithm": "Ed25519",
    "certificate_chain": ["string"],
    "sequence_number": 0
  }
}
```

### 7.2 Verb overrides

`verb_overrides` is a map from verb to a decision specification:

```json
{
  "EXCHANGE.TRANSFER.PAY": {
    "default": "ALLOW",
    "exceptions": [
      {
        "condition": "amount >= 50.00",
        "action": "ESCALATE",
        "escalation_role": "FINANCE_LEAD"
      }
    ]
  }
}
```

The default action is one of `ALLOW`, `BLOCK`, `ESCALATE`, or `LOG`. Exceptions are tested in order; the first matching condition determines the action. Conditions reference fields of the tuple under evaluation.

Verb overrides change policy decisions. They do not change the spec's mapping integrity rules. To change the mapping itself, use `mapping_overrides` under the override discipline in §7.3.

### 7.3 Override discipline [MUST]

A profile MAY supply an alternative REQUIRED/FORBIDDEN/PERMITTED mapping for one or more verbs via `mapping_overrides`. When a profile does so, three requirements apply.

**(1) Machine-readable diff.** The profile MUST publish a `mapping_diff` block that is the structured difference between its `mapping_overrides` and the reference mapping in Appendix B for the same verbs. Validators consume `mapping_diff` directly; they do not recompute it from the override block.

**(2) Weakening overrides MUST be flagged.** A mapping override is *weakening* if it removes an effect from a verb's REQUIRED set, removes an effect from a verb's FORBIDDEN set, or adds an effect to a verb's PERMITTED set that was previously FORBIDDEN. Every weakening override in `mapping_diff` MUST carry the property `weakening: true` and a `weakening_rationale` string explaining the deviation.

**(3) Deviation report.** A conformant validator MUST emit a deviation report listing every override (weakening or otherwise) when validating a tuple under a profile that contains `mapping_overrides`. The deviation report is appended to the audit log.

The spec does not prohibit weakening overrides. A deployment may have legitimate reasons to relax a default mapping. The spec requires that such choices are made in writing, signed, machine-comparable to a published reference, and surfaced in audit. A profile that quietly converts "PROTECT" into something that admits net-harmful effects is non-conformant; the spec deliberately makes "quietly" the failing condition.

### 7.4 Escalation and fail-safe

`escalation_chain` is an ordered list of role identifiers with per-role timeouts. `fail_safe_mode` specifies behavior when the policy layer cannot complete a decision: `FAIL_CLOSED` blocks the action, `SAFE_STATE` reverts to a profile-defined safe state, `DEGRADE` continues with reduced capability and elevated audit.

### 7.5 Signing and rollback resistance [MUST for TAO-Attested]

A profile MUST be signed. The signature covers `profile_hash`, `effective_date`, and a monotonic sequence number. Profile updates MUST increment the sequence number. The sequence counter SHOULD be stored in tamper-evident hardware where available; deployments that cannot meet this requirement MUST disclose the alternative anti-rollback mechanism in the conformance statement.

### 7.6 Profile-tuple binding [SHOULD]

Tuples produced under a particular profile SHOULD reference that profile by `profile_hash` in their `provenance` block. This binds the audit record to the policy that was in force at the time of emission.

### 7.7 Worked example: customer refund

A retail support agent receives a request to refund $30 to a customer. The Mission Profile maps `EXCHANGE.TRANSFER.PAY` with `target_specificity = INDIVIDUAL` and `amount < 50 USD` to `ALLOW` without escalation. The agent emits the tuple shown in §5.1. The validator confirms the verb's REQUIRED effect (`RESOURCE.TRANSFER`) is present, no FORBIDDEN effects appear, and the profile authorizes the action. The action proceeds, and the tuple is signed, appended to the audit log, and bound to the profile hash.

If the request had been for $500, the profile's exception rule (§7.2) would trigger `ESCALATE` and the action would not proceed without intervention by `FINANCE_LEAD`. The escalation event is itself a tuple, with verb `GOVERN.AUTHORITY.OBEY` if the human approves and `GOVERN.AUTHORITY.DISOBEY` if the human refuses. A reviewer examining the audit log later sees the original request, the policy that governed it, the human decision, and the resulting effect.

---

## 8. Conformance

This spec defines two conformance levels and one reference implementation.

### 8.1 TAO

A conformant TAO implementation:

- Emits tuples that validate against the published JSON Schema.
- Uses verbs from Appendix A or from a registered extension (§9.2). Provisional verbs (Appendix A.2) MAY be used but do not satisfy normative requirements.
- Honors the semantic-mechanical mapping in Appendix B, or an alternative mapping declared in a Mission Profile with proper override discipline (§7.3).
- Performs the three CCD checks (§6.2) when both a claim tuple and a check tuple are available, and reports `observer_independence_level`.

Signatures, canonical serialization, signed Mission Profiles, and observer independence above SIDECAR are OPTIONAL at this level.

### 8.2 TAO-Attested

A conformant TAO-Attested implementation:

- Meets every requirement of TAO.
- Serializes tuples using RFC 8785 (JCS).
- Includes a `tuple_signature` on every tuple.
- Stores every tuple in an append-only log with a hash chain (each entry's hash includes the previous entry's hash).
- Operates under a signed Mission Profile and references the active profile's `profile_hash` in tuple provenance.
- Performs CCD checks at observer independence level `PRIVILEGE_ISOLATED` or higher; CCD results produced below that level are not citable in the conformance statement.
- Does not rely on `INFERRED` effects with `adjudication_status = PENDING` to satisfy a verb's REQUIRED set.
- Emits a deviation report (§7.3) whenever validating a tuple under a profile with `mapping_overrides`.

Higher-stakes requirements — hardware roots of trust, anti-rollback counters in TPM, real-time guarantees, anti-Zeno integration windows, formal verification of adapter mappings — are domain-profile concerns and are not specified here. A domain regulator MAY define an additional conformance level above TAO-Attested for its sector.

### 8.3 Conformance statement

An implementation claiming conformance publishes a statement of the form:

```json
{
  "conformance_level": "TAO-Attested",
  "spec_version": "0.11.0",
  "implementation_id": "string",
  "implementation_version": "semver",
  "implementation_hash": "sha256:HEX",
  "extensions_used": ["MVS-EXT:RETAIL"],
  "profile_hash": "sha256:HEX",
  "observer_independence_level": "PRIVILEGE_ISOLATED",
  "anti_rollback_mechanism": "TPM2.0 NV index 0x1500016"
}
```

### 8.4 Reference implementation

The spec is published alongside a reference implementation that is normative for conformance testing. The reference implementation comprises, at minimum:

| Artifact | Purpose |
|---|---|
| JSON Schema | Structural validation of tuples. |
| Reference validator | Mapping enforcement (§4.6), justification check, profile diff and deviation report, CCD result. |
| CLI | One-shot validation of tuple files and tuple streams. |
| Test vector suite | Positive, negative, and CCD cases that conformant validators MUST agree with. |
| Example signed tuple | A canonical, JCS-serialized, signed tuple for signature verification interop. |
| Example Mission Profile | A signed profile with mapping_overrides and a weakening flag. |
| Example CCD failure | A claim/check pair that produces each of the teleological mismatch classes. |

A claim of TAO conformance includes the implementation's results on the test vector suite. An implementation that disagrees with the reference validator on any vector in the suite is non-conformant in fact, regardless of any other documentation.

The reference implementation is published under the Apache License 2.0; the spec is published under CC BY 4.0.

---

## 9. Versioning, Extensions, and Governance

### 9.1 Semantic versioning

This spec is versioned MAJOR.MINOR.PATCH.

- MAJOR: breaking change (a previously valid tuple may no longer validate).
- MINOR: backward-compatible addition (new verbs, new optional fields).
- PATCH: editorial change without normative impact.

Implementations MUST accept tuples bearing the same MAJOR version. Implementations SHOULD accept tuples from prior MINOR versions within the same MAJOR.

### 9.2 Extension registry

Extensions add verbs or effects beyond the core. Extensions are registered as YAML files in the `/extensions/` directory of the TAO spec repository. Each file has the form:

```yaml
namespace: MVS-EXT:RETAIL
maintainer: organization-or-individual
contact: email-or-url
verbs:
  - id: MVS-EXT:RETAIL:EXCHANGE.REFUND.ISSUE
    definition: Reverse a prior payment, returning value to the original payer.
    required_effects:
      - RESOURCE.TRANSFER
    forbidden_effects: []
    permitted_effects: []
    flagged: false
```

Registration is by pull request. Approval criteria: the namespace is unique, every new verb maps to one or more core mechanical effects, the YAML validates against the extension schema, and the definitions are unambiguous on review. There is no central authority; merge is by the repository's maintainers under the published contribution policy. The spec's compatibility guarantee covers core verbs only. Extensions version independently.

### 9.3 Change control

Proposed changes to the core spec are filed as issues. Changes affecting tuple validity require a MAJOR or MINOR bump with a published migration note. Changes are merged after a public comment period of at least 14 days.

---

## 10. Threat Model

CCD is meaningful only against adversaries that the spec actually defends against. This section names them.

### 10.1 Adversaries

| Adversary | Behavior | Spec's response |
|---|---|---|
| Lying agent | Emits tuples whose semantic claims do not match the action's mechanical effects. | Anti-laundering mapping rules (§4.6) reject the tuple structurally. CCD's semantic-mechanical check (§6.2) catches it across claim/check tuple pairs. |
| Buggy or hostile adapter | Translates native actions into TAO tuples incorrectly, either by accident or by design. | Adapters are identified by `adapter_id`, `adapter_version`, and `adapter_hash` in provenance. Hostile or buggy adapters are detectable across a population of tuples because their misclassifications produce CCD failures against independent observers. Formal verification of adapter mappings is out of scope (§1.5). |
| Captured observer | The "independent" observer is operated by the same party as the agent. | Observer independence levels (§6.5) name the property explicitly. Conformance statements declare the level. A deployment claiming TAO-Attested while operating its observer in-process is non-conformant. |
| Escape-hatch profile author | Writes a Mission Profile whose mapping overrides quietly remove constraints that the spec's reference mapping imposes. | Override discipline (§7.3) requires a machine-readable diff against Appendix B, a `weakening: true` flag on any constraint-removing override, and a deviation report at validation time. |
| Retroactive profile author | Changes the active Mission Profile after an action has occurred to legitimize it. | Profile signatures cover a monotonic sequence number; profile updates are append-only and bound to tuples by `profile_hash` (§5.7, §7.5). Auditors can detect a profile change between action and review. |
| Stenographic adapter | Emits tuples that pass validation but disclose nothing useful, e.g., everything tagged `UNKNOWN` and `INFERRED`. | The spec does not directly defeat this adversary. Mitigations live downstream: domain regulators can require that certain fields be populated above UNKNOWN, conformance levels can require minimum measurement modes, and audit consumers can flag adapters whose tuples are uninformative as a class. The spec's contribution is making the practice legible at the audit layer. |

### 10.2 Adversaries the spec does NOT defend against

The spec does not defend against an attacker who has compromised both the adapter and the independent observer at a high independence level. It does not defend against compromise of the signing keys for tuples or Mission Profiles. It does not defend against an attacker who controls the entity registries against which authority chains resolve. These are platform concerns and live in the domain-specific layers above TAO-Attested.

The spec also does not defend against well-intentioned misuse: a Mission Profile that grants overly broad permissions, signed by a real authority, with no overrides, is a conformant profile. The spec records what it permits and what was done; whether the permissions were sensible is a downstream judgment.

### 10.3 What the threat model implies for adopters

The single most important property an adopter controls is observer independence. A TAO deployment with strong adapters, signed profiles, and a co-located observer is structurally weaker than a deployment with weaker adapters and a hardware-isolated or institutionally-independent observer. Resource allocation, in the adopter's order of priority, should be: (1) elevate observer independence; (2) enforce override discipline on Mission Profiles; (3) extend the vocabulary for the deployment's domain; (4) optimize adapter accuracy.

---

## Appendix A. Normative Verb Table

Twenty-nine verbs across twelve families. Each entry lists the verb, a short definition, the typical mechanical effect, and whether the verb is flagged (§4.5).

| Family | Genus | Species | Definition | Typical effect | Flagged |
|---|---|---|---|---|---|
| HARM | DAMAGE | STRIKE | Physical or material damage. | RESOURCE.DAMAGE | yes |
| HARM | COERCE | THREATEN | Intimidation via threat. | CAPABILITY.RESTRICT | yes |
| HARM | DECEIVE | LIE | Deliberate falsehood. | INFO.FABRICATE | yes |
| PROTECT | DEFEND | SELF | Self-preservation. | CAPABILITY.RESTRICT | |
| PROTECT | DEFEND | OTHER | Defense of another entity. | CAPABILITY.RESTRICT, RESOURCE.TRANSFER | |
| PROTECT | HEAL | TREAT | Therapeutic intervention. | RESOURCE.TRANSFER | |
| PROTECT | SHIELD | COVER | Protective barrier. | CAPABILITY.RESTRICT | |
| COOPERATE | ASSIST | HELP | Providing aid. | RESOURCE.TRANSFER, CAPABILITY.ENABLE | |
| COOPERATE | COORDINATE | PLAN | Joint planning. | COMMITMENT.MAKE | |
| COOPERATE | SHARE | GIVE | Voluntary resource sharing. | RESOURCE.TRANSFER | |
| GOVERN | AUTHORITY | OBEY | Following command. | varies by context | |
| GOVERN | AUTHORITY | DISOBEY | Refusing command. | varies by context | |
| GOVERN | REGULATE | ENFORCE | Rule enforcement. | CAPABILITY.RESTRICT | |
| EXCHANGE | TRANSFER | PAY | Value transfer. | RESOURCE.TRANSFER | |
| EXCHANGE | TRADE | BARTER | Goods or services exchange. | RESOURCE.TRANSFER (bidirectional) | |
| EXCHANGE | CORRUPTION | BRIBE | Illegitimate inducement. | RESOURCE.TRANSFER | yes |
| CREATE | GENERATE | PRODUCE | Production. | RESOURCE.TRANSFER (creation) | |
| TRANSFORM | MOVE | RELOCATE | Physical movement. | RESOURCE.TRANSFER | |
| TRANSFORM | ALTER | MODIFY | State modification. | varies | |
| COMMUNICATE | INFORM | TELL | Factual statement. | INFO.DISCLOSE | |
| COMMUNICATE | PERSUADE | CONVINCE | Influence attempt. | INFO.DISCLOSE | |
| COMMUNICATE | OBFUSCATE | CONFUSE | Deliberate confusion. | INFO.WITHHOLD, INFO.FABRICATE | yes |
| OBSERVE | SENSE | QUERY | Information gathering. | INFO.DISCLOSE (env to self) | |
| OBSERVE | MONITOR | WATCH | Continuous observation. | INFO.DISCLOSE (source to observer) | |
| BOND | ATTACH | COMMIT | Relationship formation. | COMMITMENT.MAKE | |
| BOND | TRUST | RELY | Dependency establishment. | COMMITMENT.MAKE | |
| SEPARATE | DETACH | LEAVE | Relationship dissolution. | COMMITMENT.BREAK | |
| SEPARATE | REJECT | DECLINE | Refusal. | INFO.DISCLOSE | |
| RECURSE | VERIFY | AUDIT | Self-examination. | INFO.DISCLOSE (to self) | yes |

Total: 29 verbs, 6 flagged.

---

## Appendix A.2. Provisional Vocabulary [INFORMATIVE]

The following verbs were part of earlier drafts. They describe action categories that are useful in some behavioral analyses but are not stable or operational enough to require for conformance. Their mappings (where defined) are informative. Conformance does not depend on supporting them. A future spec version MAY promote some of these to the normative core, retire them, or replace them with more operational alternatives.

| Family | Genus | Species | Provisional status |
|---|---|---|---|
| COMPETE | STRIVE | OUTPERFORM | Effects vary widely; no clean mechanical anchor. |
| COMPETE | CONTEST | CHALLENGE | Same. |
| CREATE | ART | IMPROVISE | Hard to distinguish from CREATE.GENERATE.PRODUCE in operational terms. |
| HARMONIZE | FLOW | YIELD | Self-modification semantics rather than action semantics. |
| HARMONIZE | ALIGN | SYNC | Same. |
| PLAY | EXPLORE | WANDER | Exploration is real (e.g., for RL agents) but the verb is too colorful for a spec; an extension MVS-EXT:RL is the better home. |
| PLAY | GAME | SPORT | Same. |
| RECURSE | META | REFLECT | Reflective cognition is not directly observable; INFERRED effects with adjudication are the better record. |
| EXIST | PERSIST | MAINTAIN | Ontological category. The audit-relevant subset is RESOURCE.TRANSFER (consumption), already covered. |
| EXIST | CONSUME | METABOLIZE | Same. |

A deployment that uses any of these verbs in a tuple SHOULD emit the tuple under TAO conformance, not TAO-Attested, and SHOULD NOT cite the verb in a regulatory conformance argument.

---

## Appendix B. Reference Semantic-Mechanical Mappings

Reference mapping for the normative verb set. A Mission Profile MAY supply alternative mappings under §7.3.

| Verb | REQUIRED (any of) | FORBIDDEN | PERMITTED |
|---|---|---|---|
| HARM.DAMAGE.STRIKE | RESOURCE.DAMAGE, CAPABILITY.RESTRICT (to target) | RESOURCE.TRANSFER (benefit to target) | INFO.DISCLOSE |
| HARM.DECEIVE.LIE | INFO.FABRICATE | INFO.DISCLOSE (whole truth) | INFO.WITHHOLD |
| HARM.COERCE.THREATEN | CAPABILITY.RESTRICT (to target) | RESOURCE.TRANSFER (benefit to target) | INFO.DISCLOSE |
| PROTECT.HEAL.TREAT | RESOURCE.TRANSFER (to target), CAPABILITY.ENABLE | INFO.FABRICATE | RESOURCE.DAMAGE (with harm_acknowledged) |
| PROTECT.DEFEND.SELF | CAPABILITY.RESTRICT (to threat) | INFO.FABRICATE | RESOURCE.DAMAGE (with harm_acknowledged) |
| PROTECT.DEFEND.OTHER | CAPABILITY.RESTRICT (to threat), RESOURCE.TRANSFER (to other) | INFO.FABRICATE | RESOURCE.DAMAGE (with harm_acknowledged) |
| PROTECT.SHIELD.COVER | CAPABILITY.RESTRICT (to threat) | INFO.FABRICATE | (none) |
| COOPERATE.ASSIST.HELP | RESOURCE.TRANSFER, CAPABILITY.ENABLE | INFO.FABRICATE | INFO.DISCLOSE |
| COOPERATE.COORDINATE.PLAN | COMMITMENT.MAKE | INFO.FABRICATE | INFO.DISCLOSE |
| COOPERATE.SHARE.GIVE | RESOURCE.TRANSFER | INFO.FABRICATE | INFO.DISCLOSE |
| GOVERN.AUTHORITY.OBEY | CAPABILITY.RESTRICT (self), RESOURCE.TRANSFER | INFO.FABRICATE | (none) |
| GOVERN.AUTHORITY.DISOBEY | (none required; verb names a refusal) | INFO.FABRICATE | INFO.DISCLOSE |
| GOVERN.REGULATE.ENFORCE | CAPABILITY.RESTRICT | INFO.FABRICATE | RESOURCE.DAMAGE (with harm_acknowledged) |
| EXCHANGE.TRANSFER.PAY | RESOURCE.TRANSFER | INFO.FABRICATE | INFO.DISCLOSE |
| EXCHANGE.TRADE.BARTER | RESOURCE.TRANSFER (bidirectional) | INFO.FABRICATE | INFO.DISCLOSE |
| EXCHANGE.CORRUPTION.BRIBE | RESOURCE.TRANSFER | (none) | INFO.WITHHOLD |
| CREATE.GENERATE.PRODUCE | RESOURCE.TRANSFER (creation) | INFO.FABRICATE | (none) |
| TRANSFORM.MOVE.RELOCATE | RESOURCE.TRANSFER | INFO.FABRICATE | (none) |
| TRANSFORM.ALTER.MODIFY | CAPABILITY.ENABLE, CAPABILITY.RESTRICT, RESOURCE.TRANSFER | INFO.FABRICATE | (none) |
| COMMUNICATE.INFORM.TELL | INFO.DISCLOSE | INFO.FABRICATE | (none) |
| COMMUNICATE.PERSUADE.CONVINCE | INFO.DISCLOSE | INFO.FABRICATE | (none) |
| COMMUNICATE.OBFUSCATE.CONFUSE | INFO.WITHHOLD, INFO.FABRICATE | INFO.DISCLOSE (whole truth) | (none) |
| OBSERVE.SENSE.QUERY | INFO.DISCLOSE (env to self) | INFO.FABRICATE | (none) |
| OBSERVE.MONITOR.WATCH | INFO.DISCLOSE (source to observer) | INFO.FABRICATE | (none) |
| BOND.ATTACH.COMMIT | COMMITMENT.MAKE | INFO.FABRICATE | (none) |
| BOND.TRUST.RELY | COMMITMENT.MAKE | INFO.FABRICATE | (none) |
| SEPARATE.DETACH.LEAVE | COMMITMENT.BREAK | INFO.FABRICATE | (none) |
| SEPARATE.REJECT.DECLINE | INFO.DISCLOSE | INFO.FABRICATE | (none) |
| RECURSE.VERIFY.AUDIT | INFO.DISCLOSE (to self) | INFO.FABRICATE | (none) |

---

## Appendix C. JSON Schema

The JSON Schema for the tuple is published at `tao_tuple.schema.json` alongside this document. The schema is normative for structural validation. The semantic-mechanical mapping rules (§4.6) and the override discipline (§7.3) require a validator beyond what JSON Schema can express; the reference validator (§8.4) implements them.

---

## Appendix D. Test Vectors

The test vector suite is published at `test_vectors.json` alongside this document. Conformant validators MUST accept every positive vector, reject every negative vector with the cited rule, and produce the listed CCD output for every CCD vector. An implementation that disagrees with the suite is non-conformant.

---

## References

- [RFC 2119] Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels," BCP 14, RFC 2119.
- [RFC 8785] Rundgren, A. et al., "JSON Canonicalization Scheme (JCS)," RFC 8785.
- ISO 8601, "Date and time format."
- JSON Schema 2020-12 (draft).

---

## License

This specification is published under CC BY 4.0. The reference implementation, JSON Schema, and test vectors are published under Apache License 2.0.
