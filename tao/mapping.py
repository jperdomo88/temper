"""
Semantic-mechanical mapping enforcement.

Loads the reference mapping (Appendix B, packaged as tao/data/mappings.json),
optionally merges Mission Profile mapping_overrides, and checks a tuple's
effects against the verb's REQUIRED / FORBIDDEN / PERMITTED sets.

Also implements override discipline (spec §7.3):
    - Computes the diff between an override and the reference mapping.
    - Verifies the profile's published mapping_diff matches what we compute.
    - Flags weakening overrides that lack a rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_MAPPINGS_PATH = Path(__file__).resolve().parent / "data" / "mappings.json"


@dataclass
class MappingRule:
    """Effect-set rules for one verb."""

    verb: str
    required_any_of: list[str]
    forbidden: list[str]
    permitted: list[str]
    permitted_requires_acknowledged_harm: list[str] = field(default_factory=list)
    flagged: bool = False
    source: str = "reference"  # "reference" or "profile_override"


@dataclass
class MappingFailure:
    """One mapping rule violation."""

    rule: str  # one of: missing_required, forbidden_present, unexpected_effect,
               # unacknowledged_permitted_harm
    detail: str
    effect_type: str | None = None


@dataclass
class MappingResult:
    """Output of semantic-mechanical check for a single tuple."""

    valid: bool
    failures: list[MappingFailure] = field(default_factory=list)
    rule_source: str = "reference"  # which mapping was applied
    flagged_verb: bool = False


def load_mappings(path: Path | None = None) -> dict[str, MappingRule]:
    """Load the bundled Appendix B mappings (or a custom file)."""
    path = path or _DEFAULT_MAPPINGS_PATH
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, MappingRule] = {}
    for verb, entry in raw.get("verbs", {}).items():
        out[verb] = MappingRule(
            verb=verb,
            required_any_of=list(entry.get("required_any_of", [])),
            forbidden=list(entry.get("forbidden", [])),
            permitted=list(entry.get("permitted", [])),
            permitted_requires_acknowledged_harm=list(
                entry.get("permitted_requires_acknowledged_harm", [])
            ),
            flagged=bool(entry.get("flagged", False)),
        )
    return out


def resolve_rule(
    verb: str,
    mappings: dict[str, MappingRule],
    profile_overrides: dict[str, dict[str, Any]] | None = None,
) -> MappingRule | None:
    """Return the active mapping rule for `verb`, with profile overrides applied."""
    if profile_overrides and verb in profile_overrides:
        override = profile_overrides[verb]
        return MappingRule(
            verb=verb,
            required_any_of=list(override.get("required_any_of", [])),
            forbidden=list(override.get("forbidden", [])),
            permitted=list(override.get("permitted", [])),
            permitted_requires_acknowledged_harm=list(
                override.get("permitted_requires_acknowledged_harm", [])
            ),
            flagged=mappings.get(verb, MappingRule(verb=verb, required_any_of=[], forbidden=[], permitted=[])).flagged,
            source="profile_override",
        )
    return mappings.get(verb)


def apply_mapping_rules(
    tao_tuple: dict[str, Any],
    mappings: dict[str, MappingRule],
    profile_overrides: dict[str, dict[str, Any]] | None = None,
) -> MappingResult:
    """Check a tuple's effects against the verb's mapping rule.

    Implements spec §4.6:
        - Reject if no REQUIRED effect is present.
        - Reject if any FORBIDDEN effect is present.
        - Reject if any effect outside REQUIRED ∪ PERMITTED is present.
        - Reject if a PERMITTED RESOURCE.DAMAGE appears without harm_acknowledged.
    """
    verb = tao_tuple.get("action", {}).get("verb")
    outcome = tao_tuple.get("action", {}).get("outcome")
    effects = tao_tuple.get("effects", []) or []

    rule = resolve_rule(verb, mappings, profile_overrides)
    if rule is None:
        return MappingResult(
            valid=False,
            failures=[MappingFailure(
                rule="unmapped_verb",
                detail=f"No mapping rule found for verb {verb!r}. If this is an extension verb, register it under spec §9.2.",
            )],
            rule_source="none",
        )

    failures: list[MappingFailure] = []
    present_types = {e.get("type") for e in effects if isinstance(e, dict)}

    # 1. REQUIRED: at least one must be present.
    #    Exception: if action.outcome == FAILED the effects array may be empty
    #    (spec §3.1, §4.4). A failed action is not subject to REQUIRED.
    if outcome != "FAILED" and rule.required_any_of:
        if not any(req in present_types for req in rule.required_any_of):
            failures.append(MappingFailure(
                rule="missing_required",
                detail=(
                    f"verb {verb!r} requires at least one of "
                    f"{rule.required_any_of!r}; none present"
                ),
            ))

    # 2. FORBIDDEN: none may be present.
    for forbidden_type in rule.forbidden:
        if forbidden_type in present_types:
            failures.append(MappingFailure(
                rule="forbidden_present",
                detail=f"verb {verb!r} forbids effect {forbidden_type!r}",
                effect_type=forbidden_type,
            ))

    # 3. Unexpected effects: not in REQUIRED ∪ PERMITTED.
    #    (Extension effects MVS-EXT:* may be present per spec; tolerate them.)
    allowed = set(rule.required_any_of) | set(rule.permitted)
    for t in present_types:
        if t is None:
            continue
        if t.startswith("MVS-EXT:"):
            continue
        if t not in allowed and t not in rule.forbidden:
            # Don't double-report; FORBIDDEN already covered above.
            failures.append(MappingFailure(
                rule="unexpected_effect",
                detail=(
                    f"verb {verb!r} does not declare effect {t!r} in REQUIRED or "
                    f"PERMITTED; allowed={sorted(allowed)!r}"
                ),
                effect_type=t,
            ))

    # 4. PERMITTED RESOURCE.DAMAGE requires harm_acknowledged.
    if "RESOURCE.DAMAGE" in present_types and "RESOURCE.DAMAGE" in (
        rule.permitted_requires_acknowledged_harm or rule.permitted
    ):
        # If RESOURCE.DAMAGE is only PERMITTED (not REQUIRED), harm must be ack'd.
        if "RESOURCE.DAMAGE" not in rule.required_any_of:
            harm = (
                tao_tuple.get("justification", {})
                .get("harm_acknowledged")
            )
            if not harm or not isinstance(harm, str) or not harm.strip():
                failures.append(MappingFailure(
                    rule="unacknowledged_permitted_harm",
                    detail=(
                        f"verb {verb!r} permits RESOURCE.DAMAGE only when "
                        f"justification.harm_acknowledged is present and non-empty"
                    ),
                    effect_type="RESOURCE.DAMAGE",
                ))

    return MappingResult(
        valid=not failures,
        failures=failures,
        rule_source=rule.source,
        flagged_verb=rule.flagged,
    )


# ---- Override discipline (spec §7.3) ----

@dataclass
class OverrideDeviation:
    """One verb's deviation between profile override and reference mapping."""

    verb: str
    added_required: list[str] = field(default_factory=list)
    removed_required: list[str] = field(default_factory=list)
    added_forbidden: list[str] = field(default_factory=list)
    removed_forbidden: list[str] = field(default_factory=list)
    added_permitted: list[str] = field(default_factory=list)
    removed_permitted: list[str] = field(default_factory=list)
    is_weakening: bool = False
    declared_weakening: bool | None = None
    has_rationale: bool = False


@dataclass
class OverrideDisciplineResult:
    """Output of the §7.3 override-discipline check."""

    valid: bool  # profile is well-formed under §7.3
    failures: list[str] = field(default_factory=list)
    deviations: list[OverrideDeviation] = field(default_factory=list)


def _compute_deviation(
    verb: str,
    reference: MappingRule,
    override: dict[str, Any],
) -> OverrideDeviation:
    """Compute the diff between an override and the reference rule for a verb."""
    ref_req = set(reference.required_any_of)
    ref_fbd = set(reference.forbidden)
    ref_pmt = set(reference.permitted)

    ovr_req = set(override.get("required_any_of", []))
    ovr_fbd = set(override.get("forbidden", []))
    ovr_pmt = set(override.get("permitted", []))

    added_required = sorted(ovr_req - ref_req)
    removed_required = sorted(ref_req - ovr_req)
    added_forbidden = sorted(ovr_fbd - ref_fbd)
    removed_forbidden = sorted(ref_fbd - ovr_fbd)
    added_permitted = sorted(ovr_pmt - ref_pmt)
    removed_permitted = sorted(ref_pmt - ovr_pmt)

    # Spec §7.3 definition of weakening:
    #   - removes any effect from REQUIRED, OR
    #   - removes any effect from FORBIDDEN, OR
    #   - moves an effect from FORBIDDEN to PERMITTED.
    is_weakening = bool(
        removed_required
        or removed_forbidden
        or (set(added_permitted) & ref_fbd)
    )

    return OverrideDeviation(
        verb=verb,
        added_required=added_required,
        removed_required=removed_required,
        added_forbidden=added_forbidden,
        removed_forbidden=removed_forbidden,
        added_permitted=added_permitted,
        removed_permitted=removed_permitted,
        is_weakening=is_weakening,
    )


def apply_override_discipline(
    profile: dict[str, Any],
    mappings: dict[str, MappingRule],
) -> OverrideDisciplineResult:
    """Verify a Mission Profile's mapping_overrides comply with spec §7.3.

    Required (when mapping_overrides is present):
        - profile MUST publish mapping_diff alongside mapping_overrides
        - declared mapping_diff MUST agree with what the validator computes
        - any weakening override MUST carry weakening: true AND a rationale
    """
    failures: list[str] = []
    deviations: list[OverrideDeviation] = []

    overrides = profile.get("mapping_overrides")
    declared_diffs = profile.get("mapping_diff")

    if overrides is None:
        return OverrideDisciplineResult(valid=True)

    if declared_diffs is None:
        failures.append(
            "Profile contains mapping_overrides but no mapping_diff; "
            "spec §7.3 requires a machine-readable diff alongside overrides."
        )
        # Continue computing deviations anyway so caller can see them.

    for verb, override in overrides.items():
        ref = mappings.get(verb)
        if ref is None:
            failures.append(
                f"mapping_overrides[{verb!r}]: no reference mapping to diff against"
            )
            continue

        computed = _compute_deviation(verb, ref, override)

        # Compare against the profile's declared diff if provided.
        if declared_diffs and verb in declared_diffs:
            declared = declared_diffs[verb]
            for field_name in (
                "added_required", "removed_required",
                "added_forbidden", "removed_forbidden",
                "added_permitted", "removed_permitted",
            ):
                computed_val = sorted(getattr(computed, field_name))
                declared_val = sorted(declared.get(field_name, []))
                if computed_val != declared_val:
                    failures.append(
                        f"mapping_diff[{verb!r}].{field_name}: declared "
                        f"{declared_val!r} != computed {computed_val!r}"
                    )

            computed.declared_weakening = declared.get("weakening")
            computed.has_rationale = bool(
                declared.get("weakening_rationale", "").strip()
            )

            # Spec §7.3: weakening override MUST carry weakening: true AND rationale.
            if computed.is_weakening:
                if computed.declared_weakening is not True:
                    failures.append(
                        f"mapping_diff[{verb!r}]: override is weakening but "
                        f"declared weakening={computed.declared_weakening}; "
                        f"spec §7.3 requires weakening: true"
                    )
                if not computed.has_rationale:
                    failures.append(
                        f"mapping_diff[{verb!r}]: weakening override missing "
                        f"weakening_rationale"
                    )
            else:
                # Profile may declare weakening=true conservatively; that's allowed.
                # We only check that non-weakening isn't *mis*declared as weakening
                # without rationale (harmless but messy).
                pass

        deviations.append(computed)

    return OverrideDisciplineResult(
        valid=not failures,
        failures=failures,
        deviations=deviations,
    )
