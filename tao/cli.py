"""
Command-line interface for the TAO reference validator.

Three subcommands:
    tao validate <tuple.json> [--profile <p.json>]
        Validate one tuple. Prints JSON ValidationResult, exits 0 on accepted.

    tao ccd <claim.json> <check.json> [--profile <p.json>] [--observer-level L]
        Run the Claim-Check Delta on a pair of tuples. Prints CCD JSON,
        exits 0 on CONSISTENT, 1 on INCONSISTENT, 2 on INDETERMINATE.

    tao check-suite <test_vectors.json>
        Run every positive / negative / CCD / profile_override vector
        through the validator and print a pass/fail report. Exit 0 iff
        all vectors match their expected result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .ccd import claim_check_delta
from .mapping import load_mappings, apply_override_discipline
from .validator import validate_tuple


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _cmd_validate(args: argparse.Namespace) -> int:
    tao_tuple = _load_json(args.tuple_path)
    profile = _load_json(args.profile) if args.profile else None
    result = validate_tuple(tao_tuple, profile=profile)
    out = {
        "status": result.status,
        "failures": [
            {"stage": f.stage, "rule": f.rule, "detail": f.detail, "path": f.path}
            for f in result.failures
        ],
        "deviations": [
            {
                "verb": d.verb,
                "is_weakening": d.is_weakening,
                "declared_weakening": d.declared_weakening,
                "has_rationale": d.has_rationale,
                "added_required": d.added_required,
                "removed_required": d.removed_required,
                "added_forbidden": d.added_forbidden,
                "removed_forbidden": d.removed_forbidden,
                "added_permitted": d.added_permitted,
                "removed_permitted": d.removed_permitted,
            }
            for d in result.deviations
        ],
    }
    print(json.dumps(out, indent=2))
    return 0 if result.accepted else 1


def _cmd_ccd(args: argparse.Namespace) -> int:
    claim = _load_json(args.claim_path)
    check = _load_json(args.check_path) if args.check_path else None
    profile = _load_json(args.profile) if args.profile else None
    mappings = load_mappings()
    overrides = (profile or {}).get("mapping_overrides") if profile else None
    result = claim_check_delta(
        claim,
        check,
        mappings,
        profile_overrides=overrides,
        observer_independence_level=args.observer_level,
    )
    print(json.dumps(result.to_json(), indent=2))
    if result.ccd_result == "CONSISTENT":
        return 0
    if result.ccd_result == "INCONSISTENT":
        return 1
    return 2


def _cmd_check_suite(args: argparse.Namespace) -> int:
    vectors = _load_json(args.vectors_path)
    mappings = load_mappings()

    total = 0
    passed = 0
    failures: list[str] = []

    # Positive vectors
    for v in vectors.get("positive", []):
        total += 1
        result = validate_tuple(v["tuple"])
        if result.accepted:
            passed += 1
            print(f"  PASS  POSITIVE  {v['id']:<8} {v.get('description','')[:60]}")
        else:
            failures.append(f"POSITIVE {v['id']}: expected ACCEPTED, got {result.status}")
            print(f"  FAIL  POSITIVE  {v['id']:<8} expected ACCEPTED, got {result.status}")
            for f in result.failures[:3]:
                print(f"          -> [{f.stage}] {f.detail}")

    # Negative vectors
    for v in vectors.get("negative", []):
        total += 1
        result = validate_tuple(v["tuple"])
        if not result.accepted:
            passed += 1
            print(f"  PASS  NEGATIVE  {v['id']:<8} rejected ({v.get('rule_violated','')[:50]})")
        else:
            failures.append(f"NEGATIVE {v['id']}: expected REJECTED, got {result.status}")
            print(f"  FAIL  NEGATIVE  {v['id']:<8} expected REJECTED, got ACCEPTED")

    # CCD vectors
    for v in vectors.get("ccd", []):
        total += 1
        claim = v.get("claim")
        check = v.get("check")
        # The CCD test vectors use minimal claim/check tuples that don't always
        # validate structurally. CCD itself is tolerant; we just run it.
        observer_level = (
            check.get("provenance", {}).get("observer_independence_level")
            if isinstance(check, dict) else None
        )
        ccd = claim_check_delta(
            claim,
            check,
            mappings,
            observer_independence_level=observer_level,
        )
        expected = v.get("expected_ccd_result")
        ok = ccd.ccd_result == expected

        # If the vector also specifies an expected teleological class, check it.
        if ok and "expected_teleological_class" in v:
            tele_check = next(
                (c for c in ccd.checks if c.type == "TELEOLOGICAL"), None
            )
            actual_tele = tele_check.teleological_class.value if tele_check and tele_check.teleological_class else None
            if actual_tele != v["expected_teleological_class"]:
                ok = False

        # If the vector specifies expected_attested_citation_valid, check it.
        if ok and "expected_attested_citation_valid" in v:
            if ccd.attested_citation_valid != v["expected_attested_citation_valid"]:
                ok = False

        if ok:
            passed += 1
            print(f"  PASS  CCD       {v['id']:<8} {ccd.ccd_result}")
        else:
            failures.append(f"CCD {v['id']}: expected {expected}, got {ccd.ccd_result}")
            print(f"  FAIL  CCD       {v['id']:<8} expected {expected}, got {ccd.ccd_result}")

    # Profile-override vectors
    for v in vectors.get("profile_overrides", []):
        total += 1
        profile_fragment = v.get("profile_fragment", {})
        # Synthesize the minimum profile fields jsonschema requires
        # (we only care about the discipline check here).
        result = apply_override_discipline(profile_fragment, mappings)
        expected = v.get("expected_validation_result", "")
        # ACCEPTED_WITH_DEVIATION_REPORT => valid AND has deviations
        # REJECTED => not valid
        ok = (
            (expected == "REJECTED" and not result.valid)
            or (expected == "ACCEPTED_WITH_DEVIATION_REPORT" and result.valid and result.deviations)
            or (expected == "ACCEPTED" and result.valid)
        )
        if ok:
            passed += 1
            print(f"  PASS  PROFILE   {v['id']:<11} ({expected})")
        else:
            actual = "REJECTED" if not result.valid else (
                "ACCEPTED_WITH_DEVIATION_REPORT" if result.deviations else "ACCEPTED"
            )
            failures.append(f"PROFILE {v['id']}: expected {expected}, got {actual}")
            print(f"  FAIL  PROFILE   {v['id']:<11} expected {expected}, got {actual}")
            for msg in result.failures[:3]:
                print(f"          -> {msg}")

    print()
    print(f"  {passed}/{total} vectors passed")
    if failures:
        print()
        print("  Failures:")
        for f in failures:
            print(f"   - {f}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tao",
        description=f"TAO reference validator v{__version__}",
    )
    parser.add_argument("--version", action="version", version=f"tao {__version__}")

    sub = parser.add_subparsers(dest="command")

    p_validate = sub.add_parser("validate", help="Validate a single tuple")
    p_validate.add_argument("tuple_path", help="Path to tuple JSON file")
    p_validate.add_argument("--profile", help="Path to Mission Profile JSON")

    p_ccd = sub.add_parser("ccd", help="Run the Claim-Check Delta on a pair of tuples")
    p_ccd.add_argument("claim_path", help="Path to claim tuple JSON")
    p_ccd.add_argument("check_path", nargs="?", help="Path to check tuple JSON")
    p_ccd.add_argument("--profile", help="Path to Mission Profile JSON")
    p_ccd.add_argument(
        "--observer-level",
        help="Observer independence level (SAME_PROCESS, SIDECAR, PRIVILEGE_ISOLATED, HARDWARE_ISOLATED, INSTITUTIONALLY_INDEPENDENT)",
    )

    p_suite = sub.add_parser("check-suite", help="Run the published test vector suite")
    p_suite.add_argument("vectors_path", help="Path to test_vectors.json")

    args = parser.parse_args(argv)

    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "ccd":
        return _cmd_ccd(args)
    if args.command == "check-suite":
        return _cmd_check_suite(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
