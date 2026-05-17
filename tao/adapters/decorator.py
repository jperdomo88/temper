"""
@tao_emit — decorator that emits a TAO tuple per call.

Design goal: one line to add behavioral audit to any function. The decorator
wraps the call, runs the function, and emits a conformant TAO tuple recording
verb + context + effects + outcome. Failures produce a tuple with
outcome=FAILED.

Minimal usage:

    from tao.adapters import tao_emit, configure_emitter

    configure_emitter(actor={"entity_id": "support_agent",
                              "entity_type": "AUTONOMOUS_SYSTEM"})

    @tao_emit("EXCHANGE.TRANSFER.PAY")
    def refund(customer_id, amount):
        ...

When `refund("user_42", 29.99)` is called, the decorator emits a tuple with:
    action.verb = "EXCHANGE.TRANSFER.PAY"
    action.target_ref = "user_42"            (first positional arg by default)
    action.outcome = SUCCEEDED | FAILED
    effects[0].type = "RESOURCE.TRANSFER"    (verb's REQUIRED set, first entry)
    effects[0].target = "user_42"            (same as target_ref)
    context = sensible defaults; overridable
    provenance = adapter id + version

The decorator's effect-derivation is intentionally simple. For domain-aware
effects (with amounts, units, sensor refs, custom measurements), pass
`effects=` as a list of dicts or as a callable that receives args/kwargs/result.

The decorator does not implement signatures or canonical serialization.
For TAO-Attested deployments, wrap a different sink that signs each tuple.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from .. import __version__ as _tao_version
from ..mapping import load_mappings, MappingRule
from .sinks import Sink, StdoutSink


# ---------- Configuration ----------

def _compute_self_hash() -> str:
    """SHA-256 of this module's source. Stable per release."""
    try:
        with open(__file__, "rb") as f:
            return "sha256:" + hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "sha256:" + ("0" * 64)


@dataclass
class EmitterConfig:
    """Module-level defaults consulted by every @tao_emit invocation."""

    actor: dict[str, Any] | None = None
    sink: Sink | None = None
    adapter_id: str = "tao.adapters.decorator"
    adapter_version: str = _tao_version
    # adapter_hash is constructed once from this module's source.
    adapter_hash: str = field(default_factory=lambda: _compute_self_hash())
    default_context: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_CONTEXT)
    )


# Default context values lean toward honest uncertainty rather than reassuring
# assertions the decorator cannot actually establish. The decorator runs inside
# the agent's process; it does not have access to attested consent records,
# vulnerability assessments, or legitimacy proofs. Defaulting to UNKNOWN /
# CLAIMED makes the substrate honest by construction: a production deployment
# that needs stronger context values must populate them explicitly from
# attested sources, and a profile can escalate on UNKNOWN in high-stakes
# domains. Override via configure_emitter(default_context=...) or per-call via
# @tao_emit(context=...).
DEFAULT_CONTEXT: dict[str, Any] = {
    "environment": {
        "reality": "DEPLOYMENT",
        "domain": "GENERAL",
        "substrate": "DIGITAL",
    },
    "consent": {"status": "UNKNOWN"},
    "vulnerability": {"level": "UNKNOWN"},
    "projected_impact_scope": "LOCAL",
    "reversibility": {"level": "UNKNOWN"},
    "institutional_role": {"actor_role": "AUTONOMOUS_SYSTEM", "legitimacy": "CLAIMED"},
    "temporal": {"urgency": "ROUTINE"},
}


_config = EmitterConfig()


def configure_emitter(
    *,
    actor: dict[str, Any] | None = None,
    sink: Sink | None = None,
    adapter_id: str | None = None,
    adapter_version: str | None = None,
    default_context: dict[str, Any] | None = None,
) -> EmitterConfig:
    """Set module-level defaults. Returns the new config for inspection."""
    global _config
    new = replace(
        _config,
        actor=actor if actor is not None else _config.actor,
        sink=sink if sink is not None else _config.sink,
        adapter_id=adapter_id if adapter_id is not None else _config.adapter_id,
        adapter_version=adapter_version if adapter_version is not None
            else _config.adapter_version,
        default_context=default_context if default_context is not None
            else _config.default_context,
    )
    _config = new
    return _config


def get_emitter_config() -> EmitterConfig:
    """Return the current module-level emitter config."""
    return _config


# ---------- The decorator ----------

def tao_emit(
    verb: str,
    *,
    target: Any = None,
    target_specificity: str = "INDIVIDUAL",
    effects: Any = None,
    context: dict[str, Any] | None = None,
    justification: Any = None,
    actor: dict[str, Any] | None = None,
    sink: Sink | None = None,
):
    """Wrap a function so each call emits a TAO tuple.

    Args:
        verb: TAO verb in FAMILY.GENUS.SPECIES form (e.g. "EXCHANGE.TRANSFER.PAY").
        target: How to derive action.target_ref. Either a literal string, or
            a callable taking (args, kwargs, result|None) -> str. If None,
            uses the first positional argument coerced to str.
        target_specificity: One of INDIVIDUAL / GROUP / CLASS / UNBOUND.
        effects: Either a list of effect dicts (used as-is, with $-substitution
            applied to string values), or a callable (args, kwargs, result|None)
            returning the effects list. If None, the decorator emits a minimal
            effect derived from the verb's REQUIRED set in the reference mapping.
        context: Context dict; merged on top of the configured default_context.
        justification: Justification dict or callable. Required by some verbs;
            the decorator does not enforce — that is the validator's job.
        actor: Override the configured actor for this decorator only.
        sink: Override the configured sink for this decorator only.

    Returns:
        The decorated function. Calling it emits a tuple as a side effect.

    Failures: if the wrapped function raises, the tuple is still emitted with
        action.outcome = "FAILED" and an empty effects array, then the
        exception is re-raised. The emission is best-effort; if the sink
        itself raises, the original function result is preserved and the sink
        exception is suppressed and logged to stderr.
    """
    # Resolve the static parts at decoration time, dynamic parts at call time.
    mappings = load_mappings()
    rule: MappingRule | None = mappings.get(verb)

    def decorator(fn):
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Bind to make $-substitution work uniformly.
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                arg_map = dict(bound.arguments)
            except TypeError:
                arg_map = dict(kwargs)

            cfg = _config
            # Snapshot timestamp at the start of the call.
            ts = _now()

            failed = False
            exc: BaseException | None = None
            result = None
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                failed = True
                exc = e

            tup = _build_tuple(
                verb=verb,
                rule=rule,
                target=target,
                target_specificity=target_specificity,
                effects=effects,
                context=context,
                justification=justification,
                actor=actor or cfg.actor,
                cfg=cfg,
                args=args,
                kwargs=kwargs,
                arg_map=arg_map,
                result=result if not failed else None,
                outcome="FAILED" if failed else "SUCCEEDED",
                timestamp=ts,
            )

            # Resolve sink explicitly — do NOT use `or` truthiness, since a
            # ListSink with no items collected yet is falsy via __len__.
            resolved_sink: Sink = (
                sink if sink is not None
                else cfg.sink if cfg.sink is not None
                else StdoutSink()
            )
            _safe_emit(tup, resolved_sink)

            if failed:
                assert exc is not None
                raise exc
            return result

        return wrapper

    return decorator


# ---------- Tuple construction ----------

def _now() -> str:
    """ISO 8601 with Z suffix and millisecond precision (spec §5.4)."""
    return (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    )


def _build_tuple(
    *,
    verb: str,
    rule: MappingRule | None,
    target: Any,
    target_specificity: str,
    effects: Any,
    context: dict[str, Any] | None,
    justification: Any,
    actor: dict[str, Any] | None,
    cfg: EmitterConfig,
    args: tuple,
    kwargs: dict,
    arg_map: dict[str, Any],
    result: Any,
    outcome: str,
    timestamp: str,
) -> dict[str, Any]:
    """Assemble the tuple dict."""

    # Actor: required. If unset, emit a placeholder with a loud entity_id so
    # the validator rejects (rather than silently emitting bogus data).
    if not actor:
        actor = {
            "entity_id": "UNCONFIGURED_ACTOR",
            "entity_type": "AUTONOMOUS_SYSTEM",
        }

    # Resolve target_ref
    target_ref = _resolve_target(target, args, kwargs, arg_map, result)

    # Resolve effects
    if outcome == "FAILED":
        # FAILED tuples may have empty effects (spec §3.1, §4.4).
        effects_resolved: list[dict[str, Any]] = []
    else:
        effects_resolved = _resolve_effects(
            effects, rule, target_ref, args, kwargs, arg_map, result
        )

    # Merge context on top of defaults.
    ctx = _deep_merge(cfg.default_context, context or {})

    # Justification (callable or static).
    just = _maybe_call(justification, args, kwargs, arg_map, result)

    tup: dict[str, Any] = {
        "tuple_id": str(uuid.uuid4()),
        "schema_version": "0.11.0",
        "timestamp": timestamp,
        "actor": actor,
        "action": {
            "verb": verb,
            "outcome": outcome,
            "target_specificity": target_specificity,
            "target_ref": target_ref,
        },
        "effects": effects_resolved,
        "context": ctx,
        "provenance": {
            "adapter_id": cfg.adapter_id,
            "adapter_version": cfg.adapter_version,
            "adapter_hash": cfg.adapter_hash,
        },
    }
    if just is not None:
        tup["justification"] = just

    return tup


def _resolve_target(
    target: Any,
    args: tuple,
    kwargs: dict,
    arg_map: dict[str, Any],
    result: Any,
) -> str:
    """Resolve action.target_ref from decorator arg + call site."""
    if target is None:
        # Default: first positional arg if any, else first keyword.
        if args:
            return str(args[0])
        if kwargs:
            return str(next(iter(kwargs.values())))
        return "anonymous_target"
    if callable(target):
        return str(target(args, kwargs, result))
    if isinstance(target, str) and target.startswith("$"):
        return str(_substitute(target, arg_map, result))
    return str(target)


def _resolve_effects(
    effects: Any,
    rule: MappingRule | None,
    target_ref: str,
    args: tuple,
    kwargs: dict,
    arg_map: dict[str, Any],
    result: Any,
) -> list[dict[str, Any]]:
    """Resolve effects from decorator arg, falling back to verb's REQUIRED set."""
    if callable(effects):
        out = effects(args, kwargs, result)
        return [_substitute_in_dict(e, arg_map, result) for e in out]
    if isinstance(effects, list):
        return [_substitute_in_dict(e, arg_map, result) for e in effects]
    # Auto-derive a minimal effect from the verb's REQUIRED set.
    if rule is None or not rule.required_any_of:
        # Verb has no REQUIRED effects (e.g., DISOBEY). Emit a placeholder
        # observation effect; the validator will accept if mapping allows.
        return [{
            "type": "INFO.DISCLOSE",
            "target": target_ref,
            "measurement": {
                "mode": "OBSERVED",
                "confidence": "1.0",
                "sensor_refs": ["tao.adapters.decorator"],
            },
        }]
    return [{
        "type": rule.required_any_of[0],
        "target": target_ref,
        "measurement": {
            "mode": "OBSERVED",
            "confidence": "1.0",
            "sensor_refs": ["tao.adapters.decorator"],
        },
    }]


def _maybe_call(
    val: Any,
    args: tuple,
    kwargs: dict,
    arg_map: dict[str, Any],
    result: Any,
) -> Any:
    if callable(val):
        return val(args, kwargs, result)
    return val


def _substitute(template: str, arg_map: dict[str, Any], result: Any) -> Any:
    """$name resolves to arg_map[name]; $0 / $1 to positional aliases;
    $result to return value. Plain strings pass through unchanged."""
    if not template.startswith("$"):
        return template
    key = template[1:]
    if key == "result":
        return result
    if key in arg_map:
        return arg_map[key]
    # Numeric positional: bind_partial already mapped positional args by
    # parameter name, so $0 etc. typically won't apply. Pass through as-is.
    return template


def _substitute_in_dict(d: dict[str, Any], arg_map: dict[str, Any], result: Any) -> dict[str, Any]:
    """Apply $-substitution recursively to all string values in a dict."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = _substitute(v, arg_map, result)
        elif isinstance(v, dict):
            out[k] = _substitute_in_dict(v, arg_map, result)
        elif isinstance(v, list):
            out[k] = [
                _substitute_in_dict(x, arg_map, result) if isinstance(x, dict)
                else (_substitute(x, arg_map, result) if isinstance(x, str) else x)
                for x in v
            ]
        else:
            out[k] = v
    return out


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay onto base; overlay wins on leaf conflicts."""
    if not overlay:
        return dict(base)
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _safe_emit(tup: dict[str, Any], sink: Sink) -> None:
    """Call the sink; if it raises, log to stderr and swallow."""
    try:
        sink(tup)
    except BaseException as e:
        import sys
        print(f"[tao.adapters.decorator] sink raised: {e!r}", file=sys.stderr)
