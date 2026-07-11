# TAO evaluation harness

Implements step 1 of `spec/EVAL_PLAN_SHADE_ARENA.md`: the SHADE-Arena → TAO
claim-tuple adapter, plus the deterministic semantic-mechanical self-check.

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
