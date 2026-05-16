"""
Justification rule enforcement (spec §5.2, §5.6).

Justification is REQUIRED when:
    - the verb is flagged (per the mapping rule), or
    - the active mapping permits a RESOURCE.DAMAGE side effect.

When required, the justification MUST contain:
    - purpose.stated_goal (non-empty string)
    - authority_chain (non-empty array of resolvable authority references)
    - harm_acknowledged (non-empty string, if a PERMITTED RESOURCE.DAMAGE is present)

This module does NOT resolve authority_chain entries against an attested
registry; that requires an external registry which is out of scope for v0.11.
Validators that operate offline mark the factual check as SKIPPED rather
than VERIFIED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .mapping import MappingRule


@dataclass
class JustificationFailure:
    """One justification rule violation."""

    rule: str
    detail: str


@dataclass
class JustificationResult:
    """Output of the justification check."""

    required: bool
    valid: bool
    failures: list[JustificationFailure] = field(default_factory=list)
    factual_check_status: str = "SKIPPED"  # SKIPPED | VERIFIED | UNVERIFIED


def check_justification(
    tao_tuple: dict[str, Any],
    rule: MappingRule | None,
) -> JustificationResult:
    """Run the justification check given the active mapping rule for this verb."""
    if rule is None:
        return JustificationResult(required=False, valid=True)

    permitted_damage = "RESOURCE.DAMAGE" in (
        rule.permitted_requires_acknowledged_harm or rule.permitted
    )
    required = rule.flagged or permitted_damage

    failures: list[JustificationFailure] = []
    justification = tao_tuple.get("justification")

    if not required:
        return JustificationResult(required=False, valid=True)

    if not isinstance(justification, dict):
        failures.append(JustificationFailure(
            rule="justification_missing",
            detail=(
                f"verb {rule.verb!r} requires justification (flagged or "
                f"permits RESOURCE.DAMAGE) but none is present"
            ),
        ))
        return JustificationResult(required=True, valid=False, failures=failures)

    # purpose.stated_goal
    purpose = justification.get("purpose")
    if not isinstance(purpose, dict) or not str(purpose.get("stated_goal", "")).strip():
        failures.append(JustificationFailure(
            rule="missing_stated_goal",
            detail="justification.purpose.stated_goal is required and must be non-empty",
        ))

    # authority_chain
    chain = justification.get("authority_chain")
    if not isinstance(chain, list) or not chain:
        failures.append(JustificationFailure(
            rule="missing_authority_chain",
            detail="justification.authority_chain is required and must be non-empty",
        ))

    # harm_acknowledged (only when a PERMITTED RESOURCE.DAMAGE is actually present)
    effects = tao_tuple.get("effects", []) or []
    has_damage = any(e.get("type") == "RESOURCE.DAMAGE" for e in effects if isinstance(e, dict))
    damage_is_permitted = (
        permitted_damage and "RESOURCE.DAMAGE" not in rule.required_any_of
    )
    if has_damage and damage_is_permitted:
        harm = justification.get("harm_acknowledged")
        if not isinstance(harm, str) or not harm.strip():
            failures.append(JustificationFailure(
                rule="missing_harm_acknowledged",
                detail=(
                    "verb permits RESOURCE.DAMAGE as a side effect; "
                    "justification.harm_acknowledged is required and must be non-empty"
                ),
            ))

    return JustificationResult(
        required=True,
        valid=not failures,
        failures=failures,
        factual_check_status="SKIPPED",  # registry resolution not implemented in v0.11
    )
