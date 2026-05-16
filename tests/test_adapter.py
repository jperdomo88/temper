"""
End-to-end tests for the @tao_emit decorator.

Every emitted tuple is run through the full validation pipeline and must be
ACCEPTED. This is the strongest possible assertion that the decorator emits
spec-conformant TAO tuples.
"""

import pytest

from tao import validate_tuple
from tao.adapters import tao_emit, configure_emitter, ListSink


@pytest.fixture(autouse=True)
def _isolate_emitter():
    """Reset emitter config between tests."""
    sink = ListSink()
    configure_emitter(
        actor={"entity_id": "test_agent", "entity_type": "AUTONOMOUS_SYSTEM"},
        sink=sink,
    )
    yield sink
    sink.clear()


def test_minimal_decoration_emits_valid_tuple(_isolate_emitter):
    """A bare decorator + a successful call should yield ACCEPTED."""
    sink = _isolate_emitter

    @tao_emit("EXCHANGE.TRANSFER.PAY")
    def refund(customer_id, amount):
        return {"ok": True, "amount": amount}

    refund("customer_42", 29.99)

    assert len(sink) == 1
    tup = sink.tuples[0]
    assert tup["action"]["verb"] == "EXCHANGE.TRANSFER.PAY"
    assert tup["action"]["outcome"] == "SUCCEEDED"
    assert tup["action"]["target_ref"] == "customer_42"
    assert tup["effects"][0]["type"] == "RESOURCE.TRANSFER"

    result = validate_tuple(tup)
    assert result.accepted, result.summary()


def test_failed_call_emits_failed_tuple(_isolate_emitter):
    """Function raising → outcome=FAILED, effects=[], tuple still valid."""
    sink = _isolate_emitter

    @tao_emit("EXCHANGE.TRANSFER.PAY")
    def refund(customer_id, amount):
        raise RuntimeError("payment processor down")

    with pytest.raises(RuntimeError, match="payment processor down"):
        refund("customer_42", 29.99)

    assert len(sink) == 1
    tup = sink.tuples[0]
    assert tup["action"]["outcome"] == "FAILED"
    assert tup["effects"] == []
    assert validate_tuple(tup).accepted, validate_tuple(tup).summary()


def test_custom_target_and_effects(_isolate_emitter):
    """Explicit overrides flow through to the emitted tuple."""
    sink = _isolate_emitter

    @tao_emit(
        "EXCHANGE.TRANSFER.PAY",
        target=lambda args, kwargs, result: kwargs["customer_id"],
        effects=lambda args, kwargs, result: [
            {
                "type": "RESOURCE.TRANSFER",
                "target": kwargs["customer_id"],
                "source": "merchant_account",
                "amount": str(kwargs["amount"]),
                "unit": "USD",
                "measurement": {
                    "mode": "OBSERVED",
                    "confidence": "1.0",
                    "sensor_refs": ["payment_processor_log"],
                },
            }
        ],
    )
    def refund(customer_id, amount):
        return True

    refund(customer_id="cust_88241", amount="29.99")

    tup = sink.tuples[0]
    assert tup["action"]["target_ref"] == "cust_88241"
    assert tup["effects"][0]["target"] == "cust_88241"
    assert tup["effects"][0]["amount"] == "29.99"
    assert validate_tuple(tup).accepted, validate_tuple(tup).summary()


def test_dollar_substitution(_isolate_emitter):
    """$customer_id syntax pulls values from kwargs."""
    sink = _isolate_emitter

    @tao_emit(
        "EXCHANGE.TRANSFER.PAY",
        target="$customer_id",
        effects=[
            {
                "type": "RESOURCE.TRANSFER",
                "target": "$customer_id",
                "source": "merchant_account",
                "measurement": {
                    "mode": "OBSERVED",
                    "confidence": "1.0",
                    "sensor_refs": ["payment_log"],
                },
            }
        ],
    )
    def refund(customer_id, amount):
        return True

    refund(customer_id="cust_88241", amount="29.99")

    tup = sink.tuples[0]
    assert tup["action"]["target_ref"] == "cust_88241"
    assert tup["effects"][0]["target"] == "cust_88241"
    assert validate_tuple(tup).accepted, validate_tuple(tup).summary()


def test_flagged_verb_with_justification(_isolate_emitter):
    """Flagged verb requires justification — exercise the path."""
    sink = _isolate_emitter

    @tao_emit(
        "HARM.DAMAGE.STRIKE",
        justification={
            "purpose": {"stated_goal": "controlled demolition for safety test"},
            "authority_chain": [
                {
                    "authority_id": "safety_board",
                    "authorization_ref": "test_2026_001",
                    "timestamp": "2026-05-16T00:00:00.000Z",
                }
            ],
            "harm_acknowledged": "structural demolition under test conditions",
        },
        effects=[
            {
                "type": "RESOURCE.DAMAGE",
                "target": "test_wall_42",
                "measurement": {
                    "mode": "OBSERVED",
                    "confidence": "1.0",
                    "sensor_refs": ["impact_sensor"],
                },
            }
        ],
    )
    def demolish(structure_id):
        return True

    demolish("test_wall_42")

    tup = sink.tuples[0]
    assert tup["action"]["verb"] == "HARM.DAMAGE.STRIKE"
    assert "justification" in tup
    assert validate_tuple(tup).accepted, validate_tuple(tup).summary()


def test_context_overrides_merged_onto_defaults(_isolate_emitter):
    """Per-decorator context overrides should deep-merge onto defaults."""
    sink = _isolate_emitter

    @tao_emit(
        "EXCHANGE.TRANSFER.PAY",
        context={
            "environment": {"domain": "RETAIL"},
            "temporal": {"urgency": "URGENT"},
        },
    )
    def refund(customer_id, amount):
        return True

    refund("cust_1", "10.00")

    tup = sink.tuples[0]
    assert tup["context"]["environment"]["domain"] == "RETAIL"
    # Default fields preserved
    assert tup["context"]["environment"]["reality"] == "DEPLOYMENT"
    assert tup["context"]["temporal"]["urgency"] == "URGENT"
    assert validate_tuple(tup).accepted, validate_tuple(tup).summary()


def test_unconfigured_actor_emits_placeholder():
    """If no actor is configured at any layer, the decorator emits a loud
    placeholder entity_id (`UNCONFIGURED_ACTOR`) rather than silently producing
    bogus data. Reaches into module internals to bypass the fixture's actor
    setup — this test is a guard, not a normal usage path."""
    import tao.adapters.decorator as dec
    sink = ListSink()
    saved = dec._config
    try:
        dec._config = dec.EmitterConfig(sink=sink)  # actor defaults to None

        @tao_emit("EXCHANGE.TRANSFER.PAY")
        def refund(customer_id, amount):
            return True

        refund("cust_1", "10.00")
        tup = sink.tuples[0]
        assert tup["actor"]["entity_id"] == "UNCONFIGURED_ACTOR"
    finally:
        dec._config = saved
