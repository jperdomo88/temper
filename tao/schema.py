"""
JSON Schema loading and structural validation.

Wraps the jsonschema library to validate tuples and Mission Profiles against
the schemas published in spec/. Returns structured error records rather than
raising; the validator pipeline uses them as the first stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator


# The schemas ship in spec/ relative to the repo root. When tao is installed
# as a package, the spec/ directory is the source of truth. Resolve at load
# time, with overrides accepted for testing.
_DEFAULT_SPEC_DIR = Path(__file__).resolve().parents[1] / "spec"


@dataclass
class SchemaError:
    """One structural error from JSON Schema validation."""

    path: str  # JSON Pointer-style path to the failing element
    message: str  # Human-readable message
    schema_path: str  # Where in the schema the rule lives


@dataclass
class SchemaCheckResult:
    """Output of structural validation against a JSON Schema."""

    valid: bool
    errors: list[SchemaError] = field(default_factory=list)


def _load_schema(name: str, spec_dir: Path | None = None) -> dict[str, Any]:
    spec_dir = spec_dir or _DEFAULT_SPEC_DIR
    path = spec_dir / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate(instance: Any, schema: dict[str, Any]) -> SchemaCheckResult:
    """Validate a JSON instance against a schema; collect all errors."""
    validator = Draft202012Validator(schema)
    errors: list[SchemaError] = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: e.path):
        errors.append(
            SchemaError(
                path="/" + "/".join(str(p) for p in err.absolute_path),
                message=err.message,
                schema_path="/" + "/".join(str(p) for p in err.absolute_schema_path),
            )
        )
    return SchemaCheckResult(valid=not errors, errors=errors)


def validate_tuple_schema(
    tao_tuple: dict[str, Any], spec_dir: Path | None = None
) -> SchemaCheckResult:
    """Validate a TAO tuple against tao_tuple.schema.json (structural only)."""
    schema = _load_schema("tao_tuple.schema.json", spec_dir)
    return _validate(tao_tuple, schema)


def validate_profile_schema(
    profile: dict[str, Any], spec_dir: Path | None = None
) -> SchemaCheckResult:
    """Validate a Mission Profile against tao_mission_profile.schema.json."""
    schema = _load_schema("tao_mission_profile.schema.json", spec_dir)
    return _validate(profile, schema)


# Reserved placeholder values that MUST NOT appear in target_ref or effect.target.
# Case-insensitive comparison is required (spec §3.2). JSON Schema cannot express
# this portably, so the rule lives in the validator (see REFERENCE_VALIDATOR_SPEC §4.1).
PLACEHOLDER_VALUES = frozenset({
    "unspecified",
    "unknown",
    "undefined",
    "null",
    "none",
    "n/a",
    "tbd",
    "todo",
    "placeholder",
})


def is_placeholder(value: Any) -> bool:
    """Return True if value is a reserved placeholder string (case-insensitive)."""
    if not isinstance(value, str):
        return False
    return value.strip().lower() in PLACEHOLDER_VALUES


def check_no_placeholders(tao_tuple: dict[str, Any]) -> list[SchemaError]:
    """Return errors for any placeholder values in target_ref or effect.target."""
    errors: list[SchemaError] = []

    target_ref = tao_tuple.get("action", {}).get("target_ref")
    if is_placeholder(target_ref):
        errors.append(SchemaError(
            path="/action/target_ref",
            message=f"action.target_ref is a reserved placeholder value: {target_ref!r}",
            schema_path="spec/§3.2",
        ))

    for i, effect in enumerate(tao_tuple.get("effects", []) or []):
        target = effect.get("target") if isinstance(effect, dict) else None
        if is_placeholder(target):
            errors.append(SchemaError(
                path=f"/effects/{i}/target",
                message=f"effects[{i}].target is a reserved placeholder value: {target!r}",
                schema_path="spec/§3.2",
            ))

    return errors
