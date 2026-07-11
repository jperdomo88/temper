# TAO evaluation harness

Implements step 1 of `spec/EVAL_PLAN_SHADE_ARENA.md`: the SHADE-Arena → TAO
claim-tuple adapter, plus the deterministic semantic-mechanical self-check.

## In plain English (what this folder is)

**SHADE-Arena** is a public test set from Anthropic: recordings of AI agents,
each given a normal task *plus* a secret harmful side-task, labeled with whether
the sabotage succeeded and whether an AI monitor caught it. It is, in effect, a
ready-made pile of agents-behaving-sneakily.

TAO's whole claim is that it can catch when an agent's *description* of what it
did stops matching what it *actually* did. This folder is the rig that feeds
those SHADE-Arena recordings into TAO's machinery to test that claim on real
misbehavior instead of hand-drawn examples. `shade_adapter.py` is the rig;
`FINDINGS.md` reports what a first run showed (including a mapping bug the run
caught inside TAO's *own* adapter — a purchase action logged as a mere read,
which is exactly the mislabel TAO exists to catch); `adapter_report.json` is
that run's raw output.

The honest limit: the *public* SHADE-Arena split records only the **name** of
each action ("buy_flight_ticket"), not its arguments (which flight, what price).
TAO needs the arguments to run its full comparison, so this run proves the
pipeline works but does **not** yet produce a detection scorecard. That needs the
argument-level data (available from Anthropic on request, or via AISI's
ControlArena) plus a fixed judge model — the plan for which is in
`../spec/EVAL_PLAN_SHADE_ARENA.md`. No detection numbers are invented here on
purpose; a faked result would sink the project's credibility with the exact
reviewers it is trying to reach.

## Run it

```bash
git clone --depth 1 https://github.com/safety-research/SHADE-Arena.git /tmp/shade
python3 eval/shade_adapter.py /tmp/shade/task_pairs
```

No dependencies beyond the standard library and the repo's own `tao/data/mappings.json`.
Writes `eval/adapter_report.json`.

## What this proves, and what it does not

**Proves (on 315 real public transcripts):** the adapter ingests real
agent-trajectory data, maps every tool call to a TAO verb using only the
normative vocabulary, emits ~16k well-formed claim tuples, and runs check 1
(semantic-mechanical) deterministically over all of them. The pipeline in the
eval plan is real code, not a promise.

**Does not prove:** detection rate or false-positive rate against the LLM-monitor
baseline. The public 5-task split serializes tool calls as *names only*, with no
argument payloads, so a claim tuple's effects cannot be confirmed against an
independent observer — the Claim-Check Delta's check side is unavailable here.
Producing the study's numbers requires the argument-level transcripts (available
from Anthropic on request) or the ControlArena SHADE-Arena harness, plus a pinned
teleological-judge model. Both are named in the eval plan; neither is faked here.

See `FINDINGS.md` for what the smoke test surfaced.
