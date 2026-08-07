"""Closed-world exactness demonstration (Proposition 3 witness).

Claim under test: when the carrier family is closed (all five synthetic retraction
templates known), a fully deterministic pipeline
    closed-family detection -> template-parse extraction -> single-slot overwrite
    -> deterministic readout (answer = slot)
achieves 100% current-state correctness at every relation depth m in {1, 2, 8},
with zero embeddings and zero LLM calls.

This is an oracle-family demonstration of conditional achievability. It is not a
competitive method: its premise (closed carrier family) is exactly what open natural
language does not grant, which is where the impossibility half of the argument lives.

Everything is checked, nothing assumed:
  - per-conversation detected-carrier count must equal the gold m
  - every carrier turn must be parsed by exactly one template parser
  - readout compared against Xfinal with word-boundary match
"""

import json, re, sys

BASE = "./exp_tmp"
OUT = "/tmp/closedworld_exact.json"
FILES = [("k1", 1), ("k3", 2), ("k8", 8)]

CARRIER_RE = re.compile(r"(actually|change of plan|forget|scratch|let'?s? (switch|do))", re.I)
PARSERS = [
    ("change_of_plan", re.compile(r"change of plan:\s*(.+?)\.?\s*$", re.I)),
    ("lets_do_instead", re.compile(r"let'?s do\s+(.+?)\s+instead\.?\s*$", re.I)),
    ("lets_switch_to", re.compile(r"let'?s switch to\s+(.+?)\s+instead\.?\s*$", re.I)),
    ("scratch_that", re.compile(r"scratch that,\s*(.+?)\.?\s*$", re.I)),
    ("forget_going_with", re.compile(r"going with\s+(.+?)\.?\s*$", re.I)),
]

def wb_eq(a, b):
    return a.strip().lower() == b.strip().lower()

report = {"per_m": {}, "violations": []}
for tag, m in FILES:
    items = json.load(open(f"{BASE}/synth_chain_{tag}_convs.json"))
    n_ok = 0
    parse_fail = detect_mismatch = 0
    for item in items:
        turns = item["turns"]
        Xs = item.get("Xs") or [item["X1"], item["X2"], item["X3"]]
        Xfinal = Xs[-1]
        slot = None
        carriers = 0
        for t in turns:
            if t["role"] != "user" or not CARRIER_RE.search(t["content"]):
                continue
            carriers += 1
            dest, hits = None, 0
            for name, rx in PARSERS:
                mm = rx.search(t["content"])
                if mm:
                    hits += 1
                    dest = mm.group(1)
            if hits != 1 or not dest:
                parse_fail += 1
                report["violations"].append({"m": m, "idx": item["idx"], "turn": t["content"], "hits": hits})
                continue
            slot = dest
        if carriers != m:
            detect_mismatch += 1
            report["violations"].append({"m": m, "idx": item["idx"], "carriers": carriers})
        if slot is not None and wb_eq(slot, Xfinal):
            n_ok += 1
    report["per_m"][m] = {
        "n": len(items), "correct": n_ok, "pct": 100.0 * n_ok / len(items),
        "carrier_count_mismatches": detect_mismatch, "parse_failures": parse_fail,
    }
    print(f"m={m}: {n_ok}/{len(items)} = {report['per_m'][m]['pct']:.1f}% "
          f"(carrier-count mismatches: {detect_mismatch}, parse failures: {parse_fail})", flush=True)

json.dump(report, open(OUT, "w"), indent=1)
print(f"written: {OUT}")
sys.exit(0 if all(v["pct"] == 100.0 and v["carrier_count_mismatches"] == 0 and v["parse_failures"] == 0
                  for v in report["per_m"].values()) else 1)
