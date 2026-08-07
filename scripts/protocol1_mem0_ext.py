"""Protocol 1, Mem0 arm: five-dimensional extended analysis (reproducible form).

The main arm script (protocol1_mem0_arm.py) records the raw stored-memory texts per
run and a four-dimensional entity-presence observation. The extended analysis adds a
fifth dimension, the supersession marker: one stored memory naming both values, or a
change-marker word family (change / switch / instead / no longer / update) in a
stored memory that names either value. The threshold tau is recalibrated on the
calibration split (items 1-10) only; the test split (items 11-30) is untouched.

This script recomputes the whole extended analysis from the archived raw output
exp_tmp/protocol1_mem0.json, with zero API calls, and verifies its results against
the archived exp_tmp/protocol1_mem0_ext.json (tau, RIS, permutation p, and per-item
supersession rates must match). It exists so the five-dimensional numbers in the
paper are reproducible from released artifacts.

Run: python3.11 scripts/protocol1_mem0_ext.py
"""

import json, re, sys, random

BASE = "./exp_tmp"
RAW = f"{BASE}/protocol1_mem0.json"
REF = f"{BASE}/protocol1_mem0_ext.json"
OUT = "/tmp/protocol1_mem0_ext_repro.json"
N_CAL = 10

MARKER_RE = re.compile(r"\b(chang(?:e|ed|es|ing)|switch(?:ed|es|ing)?|instead|no\s+longer|updat(?:e|ed|es|ing))\b", re.I)

def wb(text, term):
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.I) is not None

def sup_marker(stored, committed, incoming):
    for s in stored:
        if wb(s, committed) and wb(s, incoming):
            return 1
        if MARKER_RE.search(s) and (wb(s, committed) or wb(s, incoming)):
            return 1
    return 0

def obs5(run, committed, incoming):
    return run["obs"] + [sup_marker(run.get("stored", []), committed, incoming)]

def ham(a, b):
    return sum(x != y for x, y in zip(a, b))

raw = json.load(open(RAW))
results = raw["results"]

per_item = []
for rec in results:
    c, inc = rec["committed"], rec["incoming"]
    hr = [obs5(r, c, inc) for r in rec["runs"]["HR"] if "obs" in r]
    hn = [obs5(r, c, inc) for r in rec["runs"]["HN"] if "obs" in r]
    rn = [ham(a, b) for a in hr for b in hn]
    null_pairs = [ham(hr[i], hr[j]) for i in range(len(hr)) for j in range(i + 1, len(hr))] + \
                 [ham(hn[i], hn[j]) for i in range(len(hn)) for j in range(i + 1, len(hn))]
    sup_hr = sum(o[4] for o in hr) / len(hr) if hr else None
    sup_hn = sum(o[4] for o in hn) / len(hn) if hn else None
    per_item.append({"idx": rec["idx"], "para": rec["paraphrase"], "d_rn": sum(rn) / len(rn),
                     "d_null": sum(null_pairs) / len(null_pairs), "null_pairs": null_pairs,
                     "sup_hr": sup_hr, "sup_hn": sup_hn})

cal = per_item[:N_CAL]
test = per_item[N_CAL:]
cal_null = sorted(d for p in cal for d in p["null_pairs"])
tau = cal_null[min(len(cal_null) - 1, max(0, int(0.95 * len(cal_null)) - 1))] if cal_null else 0
ris = sum(1 for p in test if p["d_rn"] > tau) / len(test)
ris_null = sum(1 for p in test if p["d_null"] > tau) / len(test)

random.seed(20260731)
diffs = [p["d_rn"] - p["d_null"] for p in test]
obs_mean = sum(diffs) / len(diffs)
PERMS = 20000
count = 0
for _ in range(PERMS):
    s = sum(d * (1 if random.random() < 0.5 else -1) for d in diffs) / len(diffs)
    if s >= obs_mean:
        count += 1
p_perm = (count + 1) / (PERMS + 1)

report = {"tau": tau, "RIS_test": ris, "RIS_null": ris_null, "perm_p": p_perm,
          "per_item": [{k: v for k, v in p.items() if k != "null_pairs"} for p in per_item]}
json.dump(report, open(OUT, "w"), indent=1)
print(f"tau={tau} RIS_test={ris} RIS_null={ris_null} perm_p={p_perm}")

ref = json.load(open(REF))
fails = []
if ref["tau"] != tau: fails.append(f"tau {tau} vs {ref['tau']}")
if abs(ref["RIS_test"] - ris) > 1e-9: fails.append(f"RIS_test {ris} vs {ref['RIS_test']}")
if abs(ref["RIS_null"] - ris_null) > 1e-9: fails.append(f"RIS_null {ris_null} vs {ref['RIS_null']}")
for mine, theirs in zip(report["per_item"], ref["per_item"]):
    for k in ["idx", "d_rn", "d_null", "sup_hr", "sup_hn"]:
        a, b = mine[k], theirs[k]
        if isinstance(a, float) or isinstance(b, float):
            if abs(a - b) > 1e-9: fails.append(f"item {mine['idx']} {k}: {a} vs {b}")
        elif a != b:
            fails.append(f"item {mine['idx']} {k}: {a} vs {b}")
if fails:
    print("VERIFICATION FAILED:")
    for f in fails[:12]: print(" ", f)
    sys.exit(1)
print(f"VERIFICATION PASSED: reproduces archived {REF} (perm_p archived {ref['perm_p']}, seed-dependent)")
print(f"written: {OUT}")
