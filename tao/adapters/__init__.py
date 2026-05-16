"""
TAO adapters — drop-in integrations that produce TAO tuples from existing
agent telemetry surfaces.

The decorator adapter (`tao_emit`) is the entry point most developers reach
for first: wrap any function with a verb and the decorator emits a conformant
TAO tuple every time the function runs, recording the action's outcome and
effects.

Configure once at module load:

    from tao.adapters import tao_emit, configure_emitter, JsonlSink

    configure_emitter(
        actor={"entity_id": "support_agent_v3", "entity_type": "AUTONOMOUS_SYSTEM"},
        sink=JsonlSink("audit.jsonl"),
    )

Then decorate individual actions:

    @tao_emit("EXCHANGE.TRANSFER.PAY", target="$customer_id")
    def refund(customer_id, amount): ...

The decorator is intentionally thin. It does not implement TAO-Attested
(no canonical serialization, no signatures). Production deployments compose
this adapter with a separate signer.
"""

from .decorator import (
    tao_emit,
    configure_emitter,
    get_emitter_config,
    EmitterConfig,
)
from .sinks import (
    Sink,
    StdoutSink,
    ListSink,
    JsonlSink,
    CallableSink,
)

__all__ = [
    "tao_emit",
    "configure_emitter",
    "get_emitter_config",
    "EmitterConfig",
    "Sink",
    "StdoutSink",
    "ListSink",
    "JsonlSink",
    "CallableSink",
]
