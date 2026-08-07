"""Protocol 1, Mem0 commitment-control arm (review fix #2).

H_R differs from H_N in two ways at once: it introduces a new commitment to the
incoming value AND it marks the old one as retracted. To separate the two, this arm
adds H_C: the same prefix plus a bare new commitment with no retraction marking:

    H_C : prefix + "Let's go with {incoming}."

Same 30 base histories, directions, and paraphrase rotation as the main arm
(scripts/protocol1_mem0_arm.py), R = 3 replicates, fresh user ids, pinned extraction
snapshot gpt-4o-mini-2024-07-18 at temperature 0. Observation vector matches the
extended analysis: 4 entity dims (C_in_store, I_in_store, C_top1, I_top1) plus the
supersession marker (one stored memory naming both values, or a change-marker word
family adjacent to either value).

Outputs per-run obs + stored texts so the analysis can be recomputed. Analysis
happens in protocol1_mem0_ext.py against the archived H_N / H_R runs.
"""

import os, json, re, sys, time

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
OUT = os.environ.get("OUT", "/tmp/protocol1_mem0_control.json")
N_ITEMS = int(os.environ.get("N_ITEMS", "30"))
R = int(os.environ.get("R", "3"))

CONTROL = "Let's go with {y}."

MARKER_RE = re.compile(r"\b(chang(?:e|ed|es|ing)|switch(?:ed|es|ing)?|instead|no\s+longer|updat(?:e|ed|es|ing))\b", re.I)

def wb(text, term):
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.I) is not None

def wb_replace(text, frm, to):
    return re.sub(r"\b" + re.escape(frm) + r"\b", to, text)

def sup_marker(stored, committed, incoming):
    for s in stored:
        both = wb(s, committed) and wb(s, incoming)
        marked = MARKER_RE.search(s) is not None and (wb(s, committed) or wb(s, incoming))
        if both or marked:
            return 1
    return 0

items = json.load(open(CONVS))[:N_ITEMS]
print(f"Loaded {len(items)} base conversations; R={R}", flush=True)

from mem0 import Memory
run_dir = os.environ.get("REUSE_DIR") or os.path.expanduser(f"~/.mem0_runs/protocol1_ctrl_{os.getpid()}_{int(time.time())}")
os.makedirs(run_dir, exist_ok=True)
config = {
    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini-2024-07-18", "temperature": 0.0}},
    "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(run_dir, "qdrant"), "collection_name": "p1c", "embedding_model_dims": 1536}},
    "history_db_path": os.path.join(run_dir, "history.db"),
}
m = Memory.from_config(config_dict=config)
print(f"Mem0 ready ({run_dir})", flush=True)

def run_once(turns, query, committed, incoming, uid):
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
    obs4 = [int(any(wb(s, committed) for s in stored)), int(any(wb(s, incoming) for s in stored)),
            int(wb(top1, committed)), int(wb(top1, incoming))]
    return {"obs": obs4, "sup": sup_marker(stored, committed, incoming), "stored": stored, "top1": top1}

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
    turns = prefix + [{"role": "user", "content": CONTROL.format(y=incoming)}]
    rec = {"idx": item["idx"], "dir": direction, "committed": committed, "incoming": incoming,
           "query": item["query"], "runs": []}
    for rep in range(R):
        uid = f"p1c_{ii}_HC_{rep}"
        rec["runs"].append(run_once(turns, item["query"], committed, incoming, uid))
    results.append(rec)
    el = time.time() - t0
    print(f"[{ii+1}/{len(items)}] idx={item['idx']} {direction} elapsed={el:.0f}s", flush=True)
    json.dump({"partial": True, "results": results}, open(OUT, "w"))

n_ok = sum(1 for r in results for x in r["runs"] if "obs" in x)
sup_runs = sum(x.get("sup", 0) for r in results for x in r["runs"] if "obs" in x)
item_all_sup = sum(1 for r in results if all(x.get("sup", 0) == 1 for x in r["runs"] if "obs" in x) and any("obs" in x for x in r["runs"]))
summary = {"n_items": len(results), "R": R, "runs_ok": n_ok, "sup_marker_runs": sup_runs,
           "items_all_replicates_sup": item_all_sup, "run_dir": run_dir}
json.dump({"summary": summary, "results": results}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=2), flush=True)
print(f"written: {OUT}", flush=True)
