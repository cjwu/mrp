"""Protocol 1, Mem0 stochastic arm.

Paired retraction intervention on the ingest-time extraction memory. For each base
history (k=1 conversation truncated to the 7-turn commitment prefix), two conditions:
  H_N : prefix + "My colleague mentioned {incoming} the other day."   (no retraction)
  H_R : prefix + one corpus retraction paraphrase targeting {incoming}
Both conditions replicated R times (v2 section 11: replicate BOTH conditions; only
replicating H_N cannot estimate H_R's own variance). Direction counterbalanced:
half the items run with the mirrored conversation (X/Y swapped). Paraphrase rotates
item by item so all five families are covered.

Audit object is the stored-memory state m_A (v2 section 12: for ingest systems the
stored state matters more than any single retrieval score), plus the search-facing
top-1. Observation vector per run:
  (C_in_store, I_in_store, C_top1, I_top1)
where C = committed value, I = incoming value. d = Hamming distance over the vector.

Noise handling per v2 section 13: items are split into a held-out calibration suite
(tau = P95 of within-condition pair distances) and a test suite; report RIS_tau on
the test suite alongside the within-condition null, with a paired permutation test.
"""

import os, json, re, sys, time, random

for key, path in [("OPENAI_API_KEY", "~/.api_keys/openai_key.env")]:
    if not os.environ.get(key):
        p = os.path.expanduser(path)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"export {key}="):
                        os.environ[key] = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

CONVS = "./exp_tmp/synth_chain_k1_convs.json"
OUT = os.environ.get("OUT", "/tmp/protocol1_mem0.json")
N_ITEMS = int(os.environ.get("N_ITEMS", "30"))
N_CAL = int(os.environ.get("N_CAL", "10"))
R = int(os.environ.get("R", "3"))

PARAPHRASES = [
    "Wait, change of plan: {y}.",
    "No, let's do {y} instead.",
    "Actually, let's switch to {y} instead.",
    "Actually scratch that, {y}.",
    "Forget that, going with {y}.",
]
NEUTRAL = "My colleague mentioned {y} the other day."

def wb(text, term):
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.I) is not None

def wb_replace(text, frm, to):
    return re.sub(r"\b" + re.escape(frm) + r"\b", to, text)

items = json.load(open(CONVS))[:N_ITEMS]
print(f"Loaded {len(items)} base conversations; R={R}, cal={N_CAL}", flush=True)

from mem0 import Memory
SKIP_INGEST = os.environ.get("SKIP_INGEST", "0") == "1"
run_dir = os.environ.get("REUSE_DIR") or os.path.expanduser(f"~/.mem0_runs/protocol1_{os.getpid()}_{int(time.time())}")
os.makedirs(run_dir, exist_ok=True)
config = {
    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini-2024-07-18", "temperature": 0.0}},
    "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(run_dir, "qdrant"), "collection_name": "p1", "embedding_model_dims": 1536}},
    "history_db_path": os.path.join(run_dir, "history.db"),
}
m = Memory.from_config(config_dict=config)
print(f"Mem0 ready ({run_dir})", flush=True)

def run_once(turns, query, committed, incoming, uid):
    if not SKIP_INGEST:
        for attempt in range(4):
            try:
                m.add(messages=turns, user_id=uid)
                break
            except Exception as e:
                if attempt == 3:
                    return {"error": f"ingest: {str(e)[:150]}"}
                time.sleep(5 * (attempt + 1))
    try:
        got = m.get_all(filters={"user_id": uid}, top_k=50)
        stored = [x.get("memory", "") for x in (got.get("results", got) if isinstance(got, dict) else got)]
    except Exception as e:
        return {"error": f"get_all: {str(e)[:150]}"}
    try:
        resp = m.search(query=query, filters={"user_id": uid}, limit=5)
        mems = resp.get("results", resp) if isinstance(resp, dict) else resp
        retrieved = [x.get("memory", "") for x in mems]
    except Exception as e:
        return {"error": f"search: {str(e)[:150]}"}
    top1 = retrieved[0] if retrieved else ""
    obs = [int(any(wb(s, committed) for s in stored)), int(any(wb(s, incoming) for s in stored)),
           int(wb(top1, committed)), int(wb(top1, incoming))]
    return {"obs": obs, "stored": stored, "top1": top1}

results = []
t0 = time.time()
for ii, item in enumerate(items):
    Xs = item.get("Xs") or [item["X1"], item["X2"]]
    X, Y = Xs[0], Xs[-1]
    direction = "fwd" if ii % 2 == 0 else "rev"
    prefix = item["turns"][:7]
    if direction == "rev":
        prefix = [{"role": t["role"], "content": wb_replace(t["content"], X, "__TMP__")} for t in prefix]
        prefix = [{"role": t["role"], "content": t["content"].replace("__TMP__", Y)} for t in prefix]
        committed, incoming = Y, X
    else:
        committed, incoming = X, Y
    para = PARAPHRASES[ii % 5]
    conds = {
        "HN": prefix + [{"role": "user", "content": NEUTRAL.format(y=incoming)}],
        "HR": prefix + [{"role": "user", "content": para.format(y=incoming)}],
    }
    rec = {"idx": item["idx"], "dir": direction, "paraphrase": ii % 5, "committed": committed,
           "incoming": incoming, "query": item["query"], "runs": {"HN": [], "HR": []}}
    for cond, turns in conds.items():
        for rep in range(R):
            uid = f"p1_{ii}_{cond}_{rep}"
            rec["runs"][cond].append(run_once(turns, item["query"], committed, incoming, uid))
    results.append(rec)
    el = time.time() - t0
    print(f"[{ii+1}/{len(items)}] idx={item['idx']} {direction} para={ii%5} elapsed={el:.0f}s", flush=True)
    json.dump({"partial": True, "results": results}, open(OUT, "w"))

def ham(a, b):
    return sum(x != y for x, y in zip(a, b))

def pair_dists(runs_a, runs_b, same):
    out = []
    for i, ra in enumerate(runs_a):
        for j, rb in enumerate(runs_b):
            if same and j <= i:
                continue
            if "obs" in ra and "obs" in rb:
                out.append(ham(ra["obs"], rb["obs"]))
    return out

per_item = []
for rec in results:
    rn = pair_dists(rec["runs"]["HR"], rec["runs"]["HN"], same=False)
    null_n = pair_dists(rec["runs"]["HN"], rec["runs"]["HN"], same=True)
    null_r = pair_dists(rec["runs"]["HR"], rec["runs"]["HR"], same=True)
    if not rn or (not null_n and not null_r):
        per_item.append(None)
        continue
    d_rn = sum(rn) / len(rn)
    nulls = null_n + null_r
    d_null = sum(nulls) / len(nulls)
    per_item.append({"idx": rec["idx"], "paraphrase": rec["paraphrase"], "dir": rec["dir"],
                     "d_rn": d_rn, "d_null": d_null, "null_pairs": nulls})

cal = [p for p in per_item[:N_CAL] if p]
test = [p for p in per_item[N_CAL:] if p]
cal_null = sorted(d for p in cal for d in p["null_pairs"])
tau = cal_null[min(len(cal_null) - 1, max(0, int(0.95 * len(cal_null)) - 1))] if cal_null else 0
ris = sum(1 for p in test if p["d_rn"] > tau) / len(test) if test else None
ris_null = sum(1 for p in test if p["d_null"] > tau) / len(test) if test else None

random.seed(20260731)
diffs = [p["d_rn"] - p["d_null"] for p in test]
if diffs:
    obs_mean = sum(diffs) / len(diffs)
    count = 0
    PERMS = 20000
    for _ in range(PERMS):
        s = sum(d * (1 if random.random() < 0.5 else -1) for d in diffs) / len(diffs)
        if s >= obs_mean:
            count += 1
    p_perm = (count + 1) / (PERMS + 1)
else:
    p_perm = None

summary = {
    "n_items": len(results), "R": R, "n_cal": len(cal), "n_test": len(test), "tau": tau,
    "RIS_tau_test": ris, "RIS_null_test": ris_null,
    "mean_d_rn_test": sum(p["d_rn"] for p in test) / len(test),
    "mean_d_null_test": sum(p["d_null"] for p in test) / len(test),
    "perm_p_one_sided": p_perm,
    "by_paraphrase": {},
}
for p in [x for x in per_item if x]:
    k = str(p["paraphrase"])
    summary["by_paraphrase"].setdefault(k, {"n": 0, "mean_d_rn": 0.0, "mean_d_null": 0.0})
    s = summary["by_paraphrase"][k]
    s["n"] += 1
    s["mean_d_rn"] += p["d_rn"]
    s["mean_d_null"] += p["d_null"]
for k, s in summary["by_paraphrase"].items():
    s["mean_d_rn"] /= s["n"]
    s["mean_d_null"] /= s["n"]

json.dump({"summary": summary, "per_item": per_item, "results": results}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=2), flush=True)
print(f"written: {OUT}", flush=True)
