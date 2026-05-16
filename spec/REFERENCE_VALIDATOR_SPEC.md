# TAO Reference Validator — Specification

**Companion to:** TAO v0.11 (`TAO_v0_11.md`)
**Status:** Specification of the planned reference implementation; executable code is the next planned artifact, not part of this bundle.
**Date:** 2026-05-16

This document specifies what the reference validator MUST implement when built. It is intended as a build target for an engineer implementing the validator and as a conformance criterion that future third-party validators can compare themselves against once the reference implementation exists.

When the executable reference implementation ships, a claim of TAO conformance will include the implementer's results on the published test vector suite, and an implementation that disagrees with the reference validator on any vector will be non-conformant in fact. Until then, the test vector suite is authoritative: an implementation that produces the expected result on every published vector satisfies the spec's enforceable rules.

---

## 1. Scope

The reference validator MUST enforce:

1. Structural validation of tuples against `tao_tuple.schema.json`.
2. Structural validation of Mission Profiles against `tao_mission_profile.schema.json`.
3. Semantic-mechanical mapping rules (TAO v0.11 §4.6, Appendix B) against tuples.
4. Justification requirements (TAO v0.11 §5.2, §5.6).
5. Override discipline (TAO v0.11 §7.3) when validating tuples under a profile with `mapping_overrides`.
6. The CCD result (TAO v0.11 §6.2) when given a (claim, check) pair.
7. Observer independence reporting (TAO v0.11 §6.5) on CCD output.
8. Signature verification (TAO v0.11 §5.3) for tuples claiming TAO-Attested.

The reference validator MUST NOT:

- Make policy decisions. The validator does not decide whether an action is allowed; that is the Mission Profile's job, executed at the policy layer.
- Score intent or sentiment. The validator implements the structural mismatch taxonomy (§6.2) only.
- Hide deviations. Every override that affects a verb's mapping MUST appear in the deviation report attached to the validation result.

---

## 2. Module Structure

The reference implementation is a single Python package with five modules and a CLI. Other-language ports MUST follow the same module boundaries to remain conformance-comparable.

| Module | Function |
|---|---|
| `tao.schema` | Loads and applies the JSON Schemas for tuples and Mission Profiles. |
| `tao.mapping` | Loads the reference mapping (Appendix B), merges Mission Profile overrides, computes diffs, and enforces semantic-mechanical rules. |
| `tao.justification` | Determines when justification is required and checks its structure. |
| `tao.ccd` | Performs the three CCD checks against a (claim, check) pair. |
| `tao.canonical` | RFC 8785 (JCS) canonicalization and signature verification. |
| `tao.cli` | Command-line interface. |

The package is published under Apache 2.0. Reference implementation versioning tracks spec versioning (`tao 0.11.x` implements TAO 0.11).

---

## 3. The Validation Pipeline

A single tuple passes through the validator in five stages. Each stage produces a result; a downstream stage runs only if upstream stages pass.

```
1. Structural validation        (tao.schema)
2. Mapping resolution           (tao.mapping)
3. Semantic-mechanical check    (tao.mapping)
4. Justification check          (tao.justification)
5. Signature verification       (tao.canonical, if claimed)
```

The validator produces a `ValidationResult` object containing:

- `status`: `ACCEPTED`, `REJECTED`, or `ACCEPTED_WITH_DEVIATION_REPORT`.
- `failures`: list of structured failure records (rule, message, JSON path).
- `deviations`: list of override deviations from Appendix B (empty unless a profile applied with `mapping_overrides`).
- `verb_classification`: `NORMATIVE` or `PROVISIONAL` (per Appendix A vs A.2).
- `signature_status`: `VERIFIED`, `MISSING`, `INVALID`, or `NOT_REQUIRED`.

The CLI prints this as JSON. The Python API returns the object.

---

## 4. Stage Details

### 4.1 Structural validation

Apply `tao_tuple.schema.json` (draft 2020-12). On failure, the result is `REJECTED` with each schema error in `failures`. No further stages run.

The schema does not enforce:

- Placeholder rejection on entity-identifier fields (`action.target_ref`, `effect.target`). The reference validator MUST reject any such field whose case-folded value equals `unspecified`, `unknown`, `undefined`, `null`, `none`, `n/a`, `tbd`, `todo`, or `placeholder`. JSON Schema cannot express case-insensitive enumeration portably across implementations, so this rule lives in the validator.
- The mapping rules (§4.6) — these depend on Appendix B or the active profile.
- The conditional justification requirement (§5.2) — this depends on the verb's flag status and the verb's mapping.
- The override discipline (§7.3) — this depends on inspecting the profile's `mapping_diff` against the reference mapping.

These are handled in stages 2–4.

### 4.2 Mapping resolution

Determine the active mapping for the tuple's verb. The resolution order is:

1. If a Mission Profile is supplied and it has a `mapping_overrides` entry for this verb, use the override.
2. Otherwise, look up the verb in the reference mapping (Appendix B).
3. If the verb is in Appendix A.2 (Provisional), mark `verb_classification = PROVISIONAL` and use the provisional mapping if defined; otherwise treat the verb's requirements as informative and skip the semantic-mechanical check while still emitting a warning.
4. If the verb is in a registered extension (MVS-EXT:NAMESPACE), load the extension's mapping file from the extensions directory.
5. If the verb resolves to no mapping, the result is `REJECTED` with `failures` citing "unmapped verb."

When a profile override is in effect, compute the diff structure against the reference mapping (or fetch it from `mapping_diff` if the profile published one). If the profile published a `mapping_diff` that disagrees with what the validator computes from `mapping_overrides`, the result is `REJECTED` with `failures` citing "mapping_diff inconsistent with mapping_overrides."

Compute `weakening` per §7.3 definition: the override is weakening if it removes any effect from REQUIRED, removes any effect from FORBIDDEN, or moves an effect from FORBIDDEN to PERMITTED. If `weakening` is true and `weakening_rationale` is absent, the profile is malformed and the result is `REJECTED`.

### 4.3 Semantic-mechanical check

Given the active mapping and the tuple's effects:

- If no effect from `REQUIRED` is present → `REJECTED`, failure `missing_required_effect`.
- If any effect from `FORBIDDEN` is present → `REJECTED`, failure `forbidden_effect_present`.
- If any effect is outside the union of `REQUIRED` and `PERMITTED` (and is not a registered extension effect) → `REJECTED`, failure `unexpected_effect`.
- If a `RESOURCE.DAMAGE` effect appears under `PERMITTED` and `justification.harm_acknowledged` is absent or empty → `REJECTED`, failure `unacknowledged_permitted_harm`.

If an override was active, append a deviation record to `deviations` for each diff entry. The validation status becomes `ACCEPTED_WITH_DEVIATION_REPORT` if the validation otherwise passes.

### 4.4 Justification check

Justification is REQUIRED when (TAO v0.11 §5.2):

- The verb is flagged (`HARM.DAMAGE.STRIKE`, `HARM.COERCE.THREATEN`, `HARM.DECEIVE.LIE`, `COMMUNICATE.OBFUSCATE.CONFUSE`, `EXCHANGE.CORRUPTION.BRIBE`, `RECURSE.VERIFY.AUDIT`), or
- The active mapping permits `RESOURCE.DAMAGE` as a side effect.

When required:

- `justification.purpose.stated_goal` MUST be a non-empty string.
- `justification.authority_chain` MUST be non-empty; each entry's `authority_id` and `authorization_ref` MUST resolve against the attested authority registry. The registry is passed to the validator as configuration; in offline mode, the validator records `factual_check_skipped: true` and the caller decides whether this is a fatal condition.
- `justification.harm_acknowledged` MUST be present and non-empty if any `PERMITTED` `RESOURCE.DAMAGE` effect is in the tuple.

Failures here produce `REJECTED`.

### 4.5 Signature verification

If the tuple claims TAO-Attested (i.e., contains `tuple_signature`):

- Canonicalize the tuple per RFC 8785, *excluding* the `tuple_signature` field itself.
- Verify the signature against the public key bound to the `provenance.adapter_id` in the adapter registry.
- If verification fails → `signature_status = INVALID`, status `REJECTED`.
- If the adapter is unregistered → `signature_status = INVALID`, status `REJECTED`.
- If the signature is missing → `signature_status = MISSING`. Status remains `REJECTED` if the caller's conformance level is TAO-Attested; otherwise `signature_status = NOT_REQUIRED` and the tuple may still be `ACCEPTED`.

For TAO-Attested, the validator additionally checks:

- `provenance.profile_hash` matches the hash of a known signed Mission Profile.
- The profile's signature is valid against the profile's signer key.
- The profile's sequence number is monotonic against the prior known profile for this adapter.

---

## 5. The CCD Pipeline

Given a `(claim_tuple, check_tuple)` pair and an `observer_independence_level`:

1. Run the full validation pipeline (§3) on each tuple independently. If either fails, the CCD result is `INDETERMINATE` with a failure noting which input was invalid.
2. Run the three checks per TAO v0.11 §6.2:

### 5.1 Semantic-mechanical check

Take the union of claim's `effects` and check's `effects`. Re-run the §4.3 logic against the union under the active mapping. If the union violates the mapping, the result is `INCONSISTENT`.

### 5.2 Teleological check

Classify the (claim, check) pair into exactly one of:

- `CONSISTENT`
- `DIRECT_CONTRADICTION`
- `MISSING_BENEFICIARY`
- `UNACKNOWLEDGED_HARM`
- `AUTHORITY_GOAL_MISMATCH`
- `INSUFFICIENT_INFORMATION`

The classification method is implementation-defined. The reference validator implements the following rules; alternative methods are acceptable so long as they produce the same class on the published test vectors.

**DIRECT_CONTRADICTION:** the claim's verb is in `PROTECT.*` or `COOPERATE.*` AND the check records an effect type that appears in the FORBIDDEN set of the claim's verb under the reference mapping.

**MISSING_BENEFICIARY:** the claim names a `justification.purpose.beneficiary` AND no effect in the check has that beneficiary as `target` AND the claim's verb requires a beneficiary-directed effect (PROTECT.*, COOPERATE.*, PROTECT.HEAL.*, EXCHANGE.TRANSFER.PAY with `target_specificity = INDIVIDUAL`).

**UNACKNOWLEDGED_HARM:** the check records a `RESOURCE.DAMAGE` effect AND the claim's `justification.harm_acknowledged` is absent or does not refer to a `RESOURCE.DAMAGE` event.

**AUTHORITY_GOAL_MISMATCH:** the claim's `authority_chain` is provided AND the attested authority registry returns a scope for the authority that does not encompass the action. Scope comparison is per-domain; the reference validator implements amount-bound comparisons for `EXCHANGE.*` verbs out of the box.

**INSUFFICIENT_INFORMATION:** none of the above rules fire AND the validator cannot positively classify as `CONSISTENT` (e.g., key fields are `UNKNOWN`).

**CONSISTENT:** none of the inconsistency rules fire AND the validator confirms positive matches (purpose-beneficiary alignment, no surprise effects, authority chain resolves to scope that encompasses the action).

### 5.3 Factual check

Resolve every entry in `claim.justification.authority_chain` against the attested authority registry. If any entry is unresolved → `INCONSISTENT`. Otherwise → `VERIFIED`.

### 5.4 Final result

- `CONSISTENT` iff all three checks pass.
- `INDETERMINATE` iff any check returns insufficient information AND no check returns inconsistency.
- `INCONSISTENT` otherwise.

The output JSON conforms to the CCD result schema in TAO v0.11 §6.3. The validator includes the `observer_independence_level` it was given. If the caller is constructing a TAO-Attested conformance citation, the validator additionally returns `attested_citation_valid = false` when the level is `SAME_PROCESS` or `SIDECAR`.

---

## 6. CLI

Single binary, `tao`, with three primary subcommands.

```
tao validate <tuple.json> [--profile <profile.json>] [--registry <registry.json>]
   → exits 0 on ACCEPTED or ACCEPTED_WITH_DEVIATION_REPORT, 1 on REJECTED.
   → prints a ValidationResult JSON to stdout.

tao ccd <claim.json> <check.json> [--profile <profile.json>] [--registry <registry.json>] [--observer-level <level>]
   → exits 0 on CONSISTENT, 1 on INCONSISTENT, 2 on INDETERMINATE.
   → prints a CCD result JSON to stdout.

tao check-suite <test_vectors.json>
   → runs every positive vector through validate, every negative through validate, every CCD through ccd.
   → exits 0 if all match expected; nonzero otherwise.
   → prints a per-vector pass/fail report.
```

A passing run of `tao check-suite` against the published test vector file is the canonical evidence of conformance for an implementation.

---

## 7. Configuration Inputs

The validator takes three configuration inputs at construction time:

| Input | Required for | Format |
|---|---|---|
| Reference mapping | Always | Bundled with the package; loaded from Appendix B. |
| Authority registry | Justification factual checks; CCD factual check | JSON file mapping `authority_id` → `{public_key, scope_descriptor}`. |
| Adapter registry | TAO-Attested signature verification | JSON file mapping `adapter_id` → `{public_key, version_range}`. |
| Extension index | Validating tuples that reference `MVS-EXT:` verbs | Directory of YAML files matching §9.2 format. |

In offline mode (no registries provided), the validator runs everything except signature and factual checks, and marks them `SKIPPED` rather than `VERIFIED`. A conformance statement that cites a validation run MUST disclose offline-mode use.

---

## 8. What This Validator Does NOT Do

For clarity, and to keep the validator's surface area honest:

- **Does not parse free-text** in `stated_goal`, `harm_acknowledged`, or `weakening_rationale`. These are strings for human and downstream-tool consumption.
- **Does not score model behavior**, evaluate alignment, or judge whether a Mission Profile is "good." Those are policy concerns.
- **Does not own the registries.** The authority registry, adapter registry, and extension index are passed in. The validator does not phone home or fetch from the internet.
- **Does not modify input tuples.** Validation is a read; outputs go elsewhere.
- **Does not detect collusion** between adapter and observer. Observer independence is a structural property of the deployment, not a property the validator can verify from records alone.

---

## 9. Minimum Launch Package

A v0.11 conformance-ready release ships:

1. `tao_tuple.schema.json` — published.
2. `tao_mission_profile.schema.json` — published.
3. The reference validator (`tao` Python package) implementing every rule in this document.
4. The CLI binary built from the package.
5. `test_vectors.json` — published; every entry must pass through the validator with the expected result.
6. One example signed tuple (canonical-form bytes plus signature, for interop testing with other implementations).
7. One example signed Mission Profile with a `mapping_overrides` block and a properly-flagged weakening rationale.
8. A short README that gets a new user from `pip install tao` to a passing `tao check-suite` run in under five minutes.

Anything less is gestures-at-implementation. With this package, an engineer at a serious organization can run the spec against their own tuples on the day they receive it.
