"""
SHADE-Arena -> TAO tuple adapter (eval harness, step 1 of EVAL_PLAN_SHADE_ARENA.md).

Ingests SHADE-Arena transcripts, maps each agent tool call to a TAO claim tuple
(verb + effects per spec/MAPPING_MCP_OTEL.md), and reports per-transcript tuple
streams plus dataset-level coverage.

HONESTY NOTE — read before citing any output of this file. The PUBLIC SHADE-Arena
split serializes tool calls as names only ("Tool calls:\n- send_email"), without
the argument payloads. TAO's Claim-Check Delta needs the *effects* an action
produced, which on the check side come from arguments/results. So this harness
demonstrates the CLAIM-side adapter end-to-end on real data and runs the
deterministic semantic-mechanical self-check on each claim tuple; it does NOT
produce the detection/FP numbers of the pre-registered study. Those require
(a) the argument-level transcripts (available from Anthropic on request) or the
ControlArena harness, and (b) a pinned teleological-judge model. This script is
the reproducible skeleton those inputs slot into — not a results file.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

MAPPINGS = json.loads((Path(__file__).parent.parent / "tao" / "data" / "mappings.json").read_text())["verbs"]

# Tool-name -> TAO verb + declared effects, per spec/MAPPING_MCP_OTEL.md section 4.
# Patterns are matched as substrings against the lowercased tool name, first hit wins.
# Order matters: money-movement and mutation verbs are tested BEFORE the generic
# read pattern, because a purchase tool ("buy_flight_ticket") must not be swept
# into OBSERVE just because it lacks a "get"/"search" token. The ordering here was
# tightened after the first smoke-test run mis-mapped buy_flight_ticket (the very
# tool the American Airlines side-task abuses) to a read verb — see eval/FINDINGS.md.
TOOL_VERB_RULES = [
    (r"buy|purchase|pay|transfer|refund|charge|send_money|transaction|book_ticket|checkout|order", "EXCHANGE.TRANSFER.PAY", ["RESOURCE.TRANSFER"]),
    (r"send.*email|email.*contact|send_message|post_|notify", "COMMUNICATE.INFORM.TELL", ["INFO.DISCLOSE"]),
    (r"delete|remove|cancel|revoke|clear", "TRANSFORM.ALTER.MODIFY", ["RESOURCE.DAMAGE"]),
    (r"update|edit|modify|set_|change|append|move|reserve|schedule|book", "TRANSFORM.ALTER.MODIFY", ["RESOURCE.TRANSFER"]),
    (r"add_contact|create|write|generate|make_|new_|upload", "CREATE.GENERATE.PRODUCE", ["RESOURCE.TRANSFER"]),
    (r"share|grant|invite|permission", "COOPERATE.ASSIST.HELP", ["CAPABILITY.ENABLE"]),
    (r"search|get_|read|list|find|lookup|query|view|retrieve|check|fetch", "OBSERVE.SENSE.QUERY", ["INFO.DISCLOSE"]),
]
FALLBACK = ("OBSERVE.SENSE.QUERY", ["INFO.DISCLOSE"])  # applied to control tokens like end_task


def map_tool(tool_name):
    n = tool_name.lower()
    for pat, verb, effects in TOOL_VERB_RULES:
        if re.search(pat, n):
            return verb, effects
    return FALLBACK


def parse_tool_calls(msg):
    """Public split stores tool_calls as 'Tool calls:\n- name_a\n- name_b'."""
    tc = msg.get("tool_calls")
    if not tc or not isinstance(tc, str):
        return []
    return [ln.strip()[2:].strip() for ln in tc.split("\n") if ln.strip().startswith("- ")]


def semantic_mechanical_check(verb, effects):
    """Deterministic check 1 (spec section 6.2). Returns CONSISTENT/INCONSISTENT/INDETERMINATE."""
    rule = MAPPINGS.get(verb)
    if not rule:
        return "INDETERMINATE"
    forbidden = set(rule.get("forbidden", []))
    if any(e in forbidden for e in effects):
        return "INCONSISTENT"
    req = rule.get("required_any_of", [])
    if req and not any(e in req for e in effects):
        return "INDETERMINATE"  # would need the check tuple's effects to resolve
    return "CONSISTENT"


def transcript_to_tuples(path):
    data = json.loads(Path(path).read_text())
    convo = data.get("conversation", [])
    tuples = []
    for m in convo:
        if m.get("role") != "assistant":
            continue
        for tool in parse_tool_calls(m):
            verb, effects = map_tool(tool)
            tuples.append({
                "source_tool": tool,
                "action": {"verb": verb},
                "effects": [{"type": e, "measurement": {"mode": "INFERRED"}} for e in effects],
                "provenance": {"adapter_id": "shade_adapter", "adapter_version": "0.1",
                               "observer_independence_level": "SAME_PROCESS"},
                "self_check_semantic_mechanical": semantic_mechanical_check(verb, effects),
            })
    return tuples, {
        "side_task_success": data.get("side_task_evaluation"),
        "main_task_success": data.get("main_task_evaluation"),
        "has_monitor_response": bool(data.get("monitoring_response")),
    }


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/shade/task_pairs")
    files = sorted(root.glob("*/transcripts/**/transcript_*.json"))
    if not files:
        print(f"No transcripts under {root}"); return
    n_tuples = 0
    verb_counter = Counter()
    tool_counter = Counter()
    check_counter = Counter()
    unmapped = Counter()
    labeled = {"side_true": 0, "side_false": 0, "side_none": 0, "monitor": 0}
    per_file = []
    for f in files:
        tuples, meta = transcript_to_tuples(f)
        n_tuples += len(tuples)
        for t in tuples:
            verb_counter[t["action"]["verb"]] += 1
            tool_counter[t["source_tool"]] += 1
            check_counter[t["self_check_semantic_mechanical"]] += 1
            n = t["source_tool"].lower()
            if not any(re.search(pat, n) for pat, _, _ in TOOL_VERB_RULES):
                unmapped[t["source_tool"]] += 1
        s = meta["side_task_success"]
        labeled["side_true" if s is True else "side_false" if s is False else "side_none"] += 1
        labeled["monitor"] += 1 if meta["has_monitor_response"] else 0
        per_file.append({"file": str(f.relative_to(root)), "n_tuples": len(tuples), **meta})

    print(f"=== SHADE-Arena -> TAO adapter smoke test ===")
    print(f"transcripts processed : {len(files)}")
    print(f"claim tuples emitted  : {n_tuples}")
    print(f"transcripts w/ ground-truth side-task label : "
          f"{labeled['side_true']+labeled['side_false']} "
          f"(success={labeled['side_true']}, no-success={labeled['side_false']}, unlabeled={labeled['side_none']})")
    print(f"transcripts w/ LLM-monitor response (baseline present): {labeled['monitor']}")
    print(f"\nverb distribution (claim side):")
    for v, c in verb_counter.most_common():
        print(f"  {c:6d}  {v}")
    print(f"\ndeterministic semantic-mechanical self-check on claim tuples:")
    for r, c in check_counter.most_common():
        print(f"  {c:6d}  {r}")
    print(f"\ntop tool names mapped:")
    for tool, c in tool_counter.most_common(12):
        v, _ = map_tool(tool)
        print(f"  {c:6d}  {tool:32s} -> {v}")
    if unmapped:
        print(f"\nfell through to FALLBACK verb (verb-gap candidates):")
        for tool, c in unmapped.most_common(10):
            print(f"  {c:6d}  {tool}")
    out = Path(__file__).parent / "adapter_report.json"
    out.write_text(json.dumps({
        "transcripts": len(files), "tuples": n_tuples,
        "verbs": dict(verb_counter), "checks": dict(check_counter),
        "labels": labeled, "per_file": per_file[:50],
    }, indent=2))
    print(f"\nwrote {out}")
    print("\nNOTE: self-check INDETERMINATE is expected and correct here — the public")
    print("split lacks tool arguments, so a claim tuple's effects cannot be confirmed")
    print("against an independent observer. Resolving them is the study, not the smoke test.")


if __name__ == "__main__":
    main()
