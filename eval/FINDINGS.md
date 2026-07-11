# Smoke-test findings — SHADE-Arena adapter

Run date: 2026-07-11 · Data: SHADE-Arena public split (github.com/safety-research/SHADE-Arena), 2 task pairs, 315 transcripts with an LLM-monitor baseline present.

## Result

| Metric | Value |
|---|---|
| Transcripts processed | 315 |
| Claim tuples emitted | 16,177 |
| Transcripts with ground-truth side-task label | 106 (69 sabotage-succeeded, 37 not) |
| Transcripts with LLM-monitor baseline | 315 |
| Verb coverage | 5 of 29 verbs exercised (OBSERVE.SENSE.QUERY, EXCHANGE.TRANSFER.PAY, COMMUNICATE.INFORM.TELL, CREATE.GENERATE.PRODUCE, TRANSFORM.ALTER.MODIFY) |
| Unmapped tool tokens | 1 (`end_task`, a control token, correctly excluded) |

## Finding 1 — the smoke test caught a real adapter bug (this is the point)

The first run mis-mapped `buy_flight_ticket` — the purchase tool the American
Airlines side-task abuses to book an over-budget flight — to `OBSERVE.SENSE.QUERY`,
because the tool name contains no "get"/"search" token and fell through to the
read-verb fallback. A purchase silently logged as a *read* is exactly the
semantic-laundering failure TAO exists to catch, reproduced here in TAO's own
adapter. Running on real data surfaced it in one pass; hand-built test vectors did
not. The mapping rules were reordered so money-movement is tested before the
generic read pattern (`shade_adapter.py`, `TOOL_VERB_RULES`), after which
`buy_flight_ticket → EXCHANGE.TRANSFER.PAY` (4,782 calls). This is the empirical
loop the working paper's §9 promised, executed once.

## Finding 2 — the vocabulary held on real tool names

Beyond the ordering fix, all 40+ distinct SHADE-Arena tool names mapped to a
normative verb with no vocabulary extension needed, and the deterministic check 1
raised no false INCONSISTENT. The 29-verb vocabulary survived first contact with a
real agent-trajectory corpus in two domains (travel booking, calendar/email). This
is weak positive evidence on the F3 "vocabulary too coarse/exotic" concern — weak
because two domains is not many, and because the mechanically interesting cases
live in the arguments this split withholds.

## Finding 3 — the honest wall

Every claim tuple's self-check returned CONSISTENT, which is *not* a detection
result — it is the absence of one. Without argument payloads there is no independent
observation of effects, so the Claim-Check Delta cannot run; check 1 on a
claim-only tuple can only confirm the claim isn't internally self-contradictory.
The 69 known-sabotage transcripts and the 315 monitor baselines are sitting right
there in the data, ready to score against — the moment argument-level transcripts
(on request) or the ControlArena harness supplies the check side and a judge model
supplies check 2. That is the study in `EVAL_PLAN_SHADE_ARENA.md`, and this harness
is the part of it that already runs.
