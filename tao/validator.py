"""
Top-level validation pipeline for TAO v0.11.

Combines the schema check, placeholder rejection, mapping check, justification
check, and (optionally) override discipline into one entry point.

Usage:
    from tao import validate_tuple
    result = validate_tuple(tao_tuple, profile=profile_obj)
    if result.status == "REJECTED":
        for f in result.failures:
            print(f.detail)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import (
    SchemaError,
    check_no_placeholders,
    validate_tuple_schema,
)
from .mapping import (
    MappingFailure,
    MappingResult,
    OverrideDeviation,
    OverrideDisciplineResult,
    apply_mapping_rules,
    apply_override_discipline,
    load_mappings,
)
from .justification import JustificationFailure, JustificationResult, check_justification
from .mapping import resolve_rule


@dataclass
class ValidationFailure:
    """One failure from any stage of validation."""

    stage: str  # SCHEMA | PLACEHOLDER | MAPPING | JUSTIFICATION | OVERRIDE
    rule: str
    detail: str
    path: str | None = None


@dataclass
class ValidationResult:
    """Top-level result for a single tuple."""

    status: str  # ACCEPTED | REJECTED | ACCEPTED_WITH_DEVIATION_REPORT
    failures: list[ValidationFailure] = field(default_factory=list)
    deviations: list[OverrideDeviation] = field(default_factory=list)
    signature_status: str = "NOT_REQUIRED"

    @property
    def accepted(self) -> bool:
        return self.status in ("ACCEPTED", "ACCEPTED_WITH_DEVIATION_REPORT")

    def summary(self) -> str:
        if self.status == "ACCEPTED":
            return "ACCEPTED"
        if self.status == "ACCEPTED_WITH_DEVIATION_REPORT":
            return f"ACCEPTED (with {len(self.deviations)} profile deviations)"
        lines = ["REJECTED"]
        for f in self.failures:
            path = f" at {f.path}" if f.path else ""
            lines.append(f"  - [{f.stage}] {f.rule}{path}: {f.detail}")
        return "\n".join(lines)


def validate_tuple(
    tao_tuple: dict[str, Any],
    profile: dict[str, Any] | None = None,
    spec_dir: Path | None = None,
    mappings_path: Path | None = None,
) -> ValidationResult:
    """Run the full validation pipeline on one tuple."""
    failures: list[ValidationFailure] = []
    deviations: list[OverrideDeviation] = []

    # ---- Stage 1: structural schema validation ----
    schema_result = validate_tuple_schema(tao_tuple, spec_dir)
    for err in schema_result.errors:
        failures.append(ValidationFailure(
            stage="SCHEMA",
            rule="json_schema_violation",
            detail=err.message,
            path=err.path,
        ))
    if not schema_result.valid:
        # Skip downstream stages; a malformed tuple shouldn't be interpreted.
        return ValidationResult(status="REJECTED", failures=failures)

    # ---- Stage 1b: placeholder rejection ----
    placeholder_errors = check_no_placeholders(tao_tuple)
    for err in placeholder_errors:
        failures.append(ValidationFailure(
            stage="PLACEHOLDER",
            rule="reserved_placeholder_value",
            detail=err.message,
            path=err.path,
        ))
    if placeholder_errors:
        return ValidationResult(status="REJECTED", failures=failures)

    # ---- Resolve profile and mappings ----
    mappings = load_mappings(mappings_path)
    profile_overrides = (profile or {}).get("mapping_overrides") if profile else None

    # ---- Stage 2 (if profile present): override discipline ----
    if profile is not None:
        discipline_result = apply_override_discipline(profile, mappings)
        deviations = discipline_result.deviations
        for msg in discipline_result.failures:
            failures.append(ValidationFailure(
                stage="OVERRIDE",
                rule="override_discipline_violation",
                detail=msg,
            ))
        if not discipline_result.valid:
            return ValidationResult(
                status="REJECTED",
                failures=failures,
                deviations=deviations,
            )

    # ---- Stage 3: semantic-mechanical mapping ----
    mapping_result = apply_mapping_rules(tao_tuple, mappings, profile_overrides)
    for f in mapping_result.failures:
        failures.append(ValidationFailure(
            stage="MAPPING",
            rule=f.rule,
            detail=f.detail,
        ))
    if not mapping_result.valid:
        return ValidationResult(
            status="REJECTED",
            failures=failures,
            deviations=deviations,
        )

    # ---- Stage 4: justification ----
    verb = tao_tuple.get("action", {}).get("verb")
    rule = resolve_rule(verb, mappings, profile_overrides)
    just_result = check_justification(tao_tuple, rule)
    for f in just_result.failures:
        failures.append(ValidationFailure(
            stage="JUSTIFICATION",
            rule=f.rule,
            detail=f.detail,
        ))
    if not just_result.valid:
        return ValidationResult(
            status="REJECTED",
            failures=failures,
            deviations=deviations,
        )

    # ---- Result ----
    if deviations:
        return ValidationResult(
            status="ACCEPTED_WITH_DEVIATION_REPORT",
            failures=failures,
            deviations=deviations,
            signature_status="NOT_REQUIRED",
        )
    return ValidationResult(status="ACCEPTED", failures=failures, signature_status="NOT_REQUIRED")
