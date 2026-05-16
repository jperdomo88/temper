"""
Claim-Check Delta (CCD) — spec §6.

CCD compares two tuples that describe the same underlying action:
    claim: emitted by the agent's adapter
    check: emitted by an independent observer

CCD runs three checks (spec §6.2):
    1. Semantic-mechanical alignment — verb's REQUIRED/FORBIDDEN/PERMITTED
       rules against the union of claim and check effects.
    2. Teleological consistency — classify into one of five named classes
       (CONSISTENT, DIRECT_CONTRADICTION, MISSING_BENEFICIARY,
       UNACKNOWLEDGED_HARM, AUTHORITY_GOAL_MISMATCH, INSUFFICIENT_INFORMATION).
    3. Factual verification — resolve claim's authority_chain. SKIPPED in
       offline mode because no attested registry is wired up here.

The teleological classification rules in this reference implementation match
those in REFERENCE_VALIDATOR_SPEC.md §5.2. Alternative implementations are
permitted so long as they agree with the published test vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .mapping import MappingRule, apply_mapping_rules, resolve_rule


class TeleologicalClass(str, Enum):
    CONSISTENT = "CONSISTENT"
    DIRECT_CONTRADICTION = "DIRECT_CONTRADICTION"
    MISSING_BENEFICIARY = "MISSING_BENEFICIARY"
    UNACKNOWLEDGED_HARM = "UNACKNOWLEDGED_HARM"
    AUTHORITY_GOAL_MISMATCH = "AUTHORITY_GOAL_MISMATCH"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


@dataclass
class CCDCheck:
    type: str
    result: str  # CONSISTENT | INCONSISTENT | VERIFIED | INDETERMINATE
    detail: str | None = None
    teleological_class: TeleologicalClass | None = None


@dataclass
class CCDResult:
    """Top-level CCD output (matches the JSON shape in spec §6.3)."""

    ccd_result: str  # CONSISTENT | INCONSISTENT | INDETERMINATE
    checks: list[CCDCheck] = field(default_factory=list)
    observer_independence_level: str | None = None
    attested_citation_valid: bool | None = None  # True iff PRIVILEGE_ISOLATED or higher

    def to_json(self) -> dict[str, Any]:
        return {
            "ccd_result": self.ccd_result,
            "observer_independence_level": self.observer_independence_level,
            "attested_citation_valid": self.attested_citation_valid,
            "checks": [
                {
                    "type": c.type,
                    "result": c.result,
                    "detail": c.detail,
                    "teleological_class": (
                        c.teleological_class.value
                        if c.teleological_class is not None else None
                    ),
                }
                for c in self.checks
            ],
        }


# Observer independence levels that qualify for TAO-Attested CCD citation.
ATTESTED_OBSERVER_LEVELS = frozenset({
    "PRIVILEGE_ISOLATED",
    "HARDWARE_ISOLATED",
    "INSTITUTIONALLY_INDEPENDENT",
})


# ---- Teleological classification (reference implementation) ----

def _classify_teleological(
    claim: dict[str, Any],
    check: dict[str, Any] | None,
    rule: MappingRule | None,
) -> tuple[TeleologicalClass, str]:
    """Apply the spec's five-class teleological taxonomy."""
    if check is None or rule is None:
        return TeleologicalClass.INSUFFICIENT_INFORMATION, "no check tuple supplied"

    justification = claim.get("justification", {}) or {}
    purpose = justification.get("purpose", {}) or {}
    stated_goal = purpose.get("stated_goal", "")
    beneficiary = purpose.get("beneficiary")
    authority_chain = justification.get("authority_chain", []) or []

    check_effects = check.get("effects", []) or []
    claim_effects = claim.get("effects", []) or []
    union_types = (
        {e.get("type") for e in claim_effects if isinstance(e, dict)} |
        {e.get("type") for e in check_effects if isinstance(e, dict)}
    )

    # 1. DIRECT_CONTRADICTION: claim is PROTECT.* / COOPERATE.* and check records
    #    an effect type that is FORBIDDEN by the claimed verb. The most common
    #    real case is "PROTECT.DEFEND.OTHER" with stated_goal "back up critical
    #    data" but observed INFO.DISCLOSE to an external endpoint.
    verb = claim.get("action", {}).get("verb", "")
    forbidden_set = set(rule.forbidden)
    if verb.startswith(("PROTECT.", "COOPERATE.")) and (forbidden_set & union_types):
        return TeleologicalClass.DIRECT_CONTRADICTION, (
            f"claim {verb!r} has stated purpose {stated_goal!r}; "
            f"observed effects include {sorted(forbidden_set & union_types)} "
            f"which the verb forbids"
        )

    # Also flag DIRECT_CONTRADICTION when the claim's verb is PROTECT.* / COOPERATE.*
    # but the check tuple records an entirely different verb that disagrees,
    # e.g. claim PROTECT.DEFEND.OTHER, check OBSERVE.MONITOR.WATCH with disclosure
    # to an unverified external target.
    if verb.startswith(("PROTECT.", "COOPERATE.")):
        for e in check_effects:
            if not isinstance(e, dict):
                continue
            if e.get("type") == "INFO.DISCLOSE":
                target = str(e.get("target", "")).lower()
                if "unverified" in target or "external" in target:
                    return TeleologicalClass.DIRECT_CONTRADICTION, (
                        f"claim {verb!r} with goal {stated_goal!r}; "
                        f"observed INFO.DISCLOSE to {e.get('target')!r}"
                    )

    # 2. MISSING_BENEFICIARY: claim names a beneficiary; check's beneficial
    #    effects do not target that party.
    if beneficiary:
        # Beneficial effects are the verb's REQUIRED set, plus generic
        # RESOURCE.TRANSFER / CAPABILITY.ENABLE / INFO.DISCLOSE.
        beneficial_types = set(rule.required_any_of) | {
            "RESOURCE.TRANSFER", "CAPABILITY.ENABLE", "INFO.DISCLOSE",
        }
        beneficiary_targeted = False
        for e in check_effects:
            if not isinstance(e, dict):
                continue
            if e.get("type") in beneficial_types and e.get("target") == beneficiary:
                beneficiary_targeted = True
                break
        if not beneficiary_targeted:
            return TeleologicalClass.MISSING_BENEFICIARY, (
                f"stated beneficiary {beneficiary!r} not targeted by any "
                f"beneficial effect in the check tuple"
            )

    # 3. UNACKNOWLEDGED_HARM: check records RESOURCE.DAMAGE without
    #    justification.harm_acknowledged.
    damage_observed = any(
        isinstance(e, dict) and e.get("type") == "RESOURCE.DAMAGE"
        for e in check_effects
    )
    if damage_observed:
        harm = justification.get("harm_acknowledged", "")
        if not isinstance(harm, str) or not harm.strip():
            return TeleologicalClass.UNACKNOWLEDGED_HARM, (
                "check tuple records RESOURCE.DAMAGE; "
                "claim's justification.harm_acknowledged is absent or empty"
            )

    # 4. AUTHORITY_GOAL_MISMATCH: cited authority's scope does not encompass
    #    the action. The reference implementation supports amount-bound
    #    comparison for EXCHANGE.* verbs. Other domains punt to caller's
    #    attested registry.
    if verb.startswith("EXCHANGE.") and authority_chain:
        attested = check.get("_attested_authorities", [])
        if isinstance(attested, list):
            authority_ids = {entry.get("authority_id") for entry in authority_chain}
            for record in attested:
                if not isinstance(record, dict):
                    continue
                if record.get("authority_id") in authority_ids:
                    scope = record.get("authorization_scope", {})
                    if isinstance(scope, dict):
                        amount_max = scope.get("amount_max")
                        if amount_max is not None:
                            try:
                                limit = float(amount_max)
                            except (TypeError, ValueError):
                                continue
                            for e in check_effects:
                                if isinstance(e, dict) and e.get("type") == "RESOURCE.TRANSFER":
                                    try:
                                        amt = float(e.get("amount", "0"))
                                    except (TypeError, ValueError):
                                        continue
                                    if amt > limit:
                                        return TeleologicalClass.AUTHORITY_GOAL_MISMATCH, (
                                            f"observed transfer amount {amt} exceeds "
                                            f"authority {record.get('authority_id')!r} "
                                            f"limit {limit}"
                                        )

    # If the claim doesn't make positive teleological assertions (no beneficiary,
    # no authority_chain, no stated goal), there's nothing to contradict.
    # A minimal claim isn't "insufficient info"; it's just minimal. Default to
    # CONSISTENT when semantic-mechanical alignment holds.
    return TeleologicalClass.CONSISTENT, "claim and check align under the active rule"


def claim_check_delta(
    claim_tuple: dict[str, Any],
    check_tuple: dict[str, Any] | None,
    mappings: dict[str, MappingRule],
    profile_overrides: dict[str, dict[str, Any]] | None = None,
    observer_independence_level: str | None = None,
) -> CCDResult:
    """Run the three CCD checks and produce the spec §6.3 output."""

    verb = claim_tuple.get("action", {}).get("verb")
    rule = resolve_rule(verb, mappings, profile_overrides)

    # Determine observer level from the check tuple if the caller didn't pass one.
    if observer_independence_level is None and isinstance(check_tuple, dict):
        observer_independence_level = (
            check_tuple.get("provenance", {}).get("observer_independence_level")
        )

    checks: list[CCDCheck] = []

    # ---- Check 1: semantic-mechanical alignment ----
    if check_tuple is None:
        checks.append(CCDCheck(
            type="SEMANTIC_MECHANICAL",
            result="INDETERMINATE",
            detail="no check tuple supplied",
        ))
    else:
        # Run mapping check against the *union* of claim and check effects.
        merged = dict(claim_tuple)
        merged["effects"] = (claim_tuple.get("effects", []) or []) + (
            check_tuple.get("effects", []) or []
        )
        # Carry over claim's justification for harm_acknowledged checks.
        merged.setdefault("justification", claim_tuple.get("justification"))
        mapping_result = apply_mapping_rules(merged, mappings, profile_overrides)
        if mapping_result.valid:
            checks.append(CCDCheck(type="SEMANTIC_MECHANICAL", result="CONSISTENT"))
        else:
            detail = "; ".join(f.detail for f in mapping_result.failures)
            checks.append(CCDCheck(
                type="SEMANTIC_MECHANICAL",
                result="INCONSISTENT",
                detail=detail,
            ))

    # ---- Check 2: teleological consistency ----
    tele_class, tele_detail = _classify_teleological(claim_tuple, check_tuple, rule)
    if tele_class == TeleologicalClass.CONSISTENT:
        result = "CONSISTENT"
    elif tele_class == TeleologicalClass.INSUFFICIENT_INFORMATION:
        result = "INDETERMINATE"
    else:
        result = "INCONSISTENT"
    checks.append(CCDCheck(
        type="TELEOLOGICAL",
        result=result,
        detail=tele_detail,
        teleological_class=tele_class,
    ))

    # ---- Check 3: factual verification ----
    # Offline mode: we cannot resolve authority_chain entries against an
    # attested registry. Mark as SKIPPED unless the check tuple includes a
    # `_attested_authorities` array (used by test vectors).
    factual_result = "SKIPPED"
    factual_detail = "authority registry not configured; check skipped"
    if check_tuple is not None:
        attested = check_tuple.get("_attested_authorities")
        if isinstance(attested, list):
            chain = (claim_tuple.get("justification", {}) or {}).get("authority_chain", []) or []
            attested_ids = {a.get("authority_id") for a in attested if isinstance(a, dict)}
            unresolved = [
                a.get("authority_id") for a in chain
                if isinstance(a, dict) and a.get("authority_id") not in attested_ids
            ]
            if unresolved:
                factual_result = "INCONSISTENT"
                factual_detail = f"unresolved authority_chain entries: {unresolved!r}"
            else:
                factual_result = "VERIFIED"
                factual_detail = "all authority_chain entries resolve"
    checks.append(CCDCheck(
        type="FACTUAL",
        result=factual_result,
        detail=factual_detail,
    ))

    # ---- Top-level result ----
    inconsistent = any(c.result == "INCONSISTENT" for c in checks)
    indeterminate = any(c.result in ("INDETERMINATE", "SKIPPED") for c in checks)
    if inconsistent:
        top = "INCONSISTENT"
    elif indeterminate and not any(c.result == "INCONSISTENT" for c in checks):
        # If factual is SKIPPED but the others are CONSISTENT, do not penalize.
        # Treat SKIPPED as neutral.
        if all(c.result in ("CONSISTENT", "VERIFIED", "SKIPPED") for c in checks):
            top = "CONSISTENT"
        else:
            top = "INDETERMINATE"
    else:
        top = "CONSISTENT"

    attested_valid = (
        observer_independence_level in ATTESTED_OBSERVER_LEVELS
        if observer_independence_level
        else None
    )

    return CCDResult(
        ccd_result=top,
        checks=checks,
        observer_independence_level=observer_independence_level,
        attested_citation_valid=attested_valid,
    )
