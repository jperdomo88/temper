# For a frontier-lab product manager

**Reader's question:** Will this slow my agent's release cadence?

## The short answer

No — and it makes a class of post-launch incidents that *would* slow you down considerably less likely. TAO is an audit substrate, not a guardrail framework. It sits next to the agent. It doesn't rewrite the agent's behavior; it records what the agent did in a structure that survives the question "what actually happened here?" three months later.

## What it costs you to integrate

The minimum integration is a decorator on the tool calls your agent already makes. To be clear about what this gives you: it produces structured claim records of what the agent says it did. That's TAO-Log (see the adoption playbook's tier table). It is not yet audit-grade, because the adapter runs inside the agent. Audit-grade requires an independent observer (TAO-Check), which is a larger but well-defined piece of work — see `spec/ADOPTION_PLAYBOOK.md`.

The decorator itself:

```python
from tao.adapters import tao_emit

@tao_emit(verb="COMMUNICATE.INFORM.TELL")
def send_message(to, body):
    ...
```

The decorator emits one JSON object per tool call to a sink you configure (stdout, a file, an HTTP endpoint, your existing log pipeline). No new dependencies on the inference path. The hot-path cost is a single JSON serialization per emit. Your agent's behavior is unchanged.

The richer integrations — adding an independent observer (TAO-Check), wiring Mission Profile decisions back into the agent's effect surface (TAO-Governed) — are incremental and well-defined. The right sequencing is: ship TAO-Log, run it in shadow mode for two sprints to characterize your effect surface, then add observer coverage on the highest-cost effect family (network egress for a code agent, payments for a browser agent, message sends for a customer-support agent). Don't wait for an incident — incidents are how you discover that the observer surface you skipped was the one that mattered.

## What it gives you that you don't have today

Three things, ordered by how soon you'll want them:

1. **A behavioral record you can show a regulator.** When the EU AI Act risk classification asks "how do you log agentic behavior?", the TAO emit stream is an answer with structure. Today most teams ship trace JSON that's expensive to query and impossible to compare across deployments.

2. **A common vocabulary across agents.** If your portfolio has a code agent, a customer-support agent, and a browser agent, the same audit log format lets you write one analyzer instead of three. The 29-verb taxonomy is small enough to learn in an afternoon.

3. **Claim-vs-check as an early-warning signal.** The agent's own description of its action *and* an independent observer's record of what actually happened, both in the same schema. When they diverge, you have a behavioral signal independent of model introspection. This is the only piece that requires non-trivial work, but it's the piece that catches the failures introspection misses.

## What it doesn't claim to do

TAO doesn't certify model safety. It doesn't replace evals. It doesn't tell you whether your agent's *decisions* are good — it tells you whether what your agent *says it did* matches what it *actually did*. A clean CCD log on a misbehaving agent is just a clean log of misbehavior. The substrate is necessary; safety work is not optional.

## Decision shape

If you ship agentic features in 2026, you will be asked for some version of this audit substrate by a regulator, an enterprise customer, or your own incident retrospective. The cost of building it after a public incident is high. The cost of dropping the decorator into your existing tool calls is low.

The full spec is ~18 pages: [`TAO_v0_11.md`](../TAO_v0_11.md). The decorator is ~150 lines of Python: [`tao/adapters/decorator.py`](../../tao/adapters/decorator.py). A worked failure mode you may already worry about: [`scenarios/browser_agent_subscription.md`](../scenarios/browser_agent_subscription.md).
