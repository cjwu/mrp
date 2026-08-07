"""Per-revision snapshot instrumentation of Mem0 on the m=8 chains.

Purpose: the chain grid shows single-event update success does not imply
chain-level validity, but the pinned runs observe only the final state. This
instrumented arm ingests each conversation segment by segment and dumps the
full memory store after every revision, separating five layers per step:
  L1 store change      : the store's text set changed after the segment
  L2 incoming present  : the incoming value appears somewhere in the store
  L3 step marker       : one memory names both the outgoing and incoming value,
                         or names the incoming value beside a change-marker
                         word (change, switch, instead, no longer, update);
                         mirrors the protocol-1 supersession-marker definition
  L4 retrieval inclusion: final-state search top-5 contains the current value
  L5 generator resolution: final answer category under the pinned prompt
Also recorded per snapshot: wrong-final assertion (a memory matching the
finality family that names a superseded value and not the expected current
one) and correct-final assertion.

Composition test: a conversation counts as a pure composition failure only if
every step passes L2 and L3 and the final answer is still stale; conversations
where some step already fails locally are local-update failures under
accumulation, and the output separates the two.

Comparability caveat (documented in the paper if used): ingestion here adds
one segment per m.add call (9 calls per conversation), while the pinned run
added 50-turn batches (2 calls); extraction input grouping therefore differs,
so this arm stands alone and is not a re-measurement of the pinned 46%.

Segmentation: boundary b_r = first user turn whose text word-boundary-matches
chain value Xs[r]; verified strictly increasing for all 50 conversations.
Segment r = turns[b_r .. b_{r+1}-1]; snapshot S_r after ingesting segment r
has expected current value Xs[r].

Pins: mem0ai 2.0.1, extraction gpt-4o-mini-2024-07-18 at temperature 0,
per-run qdrant store, answering model claude-haiku-4-5-20251001 at
temperature 0 with the pinned prompt template and word-boundary grading.

Run:    python3 scripts/mem0_chain_snapshot.py
Env:    K=8 (default), NLIM=0 (limit conversations for smoke test), OUT=path
Writes: /tmp/mem0_chain_snapshot_k8.json
"""

import os
import json
import time
import re
import sys
import urllib.request

for key, path in [("OPENAI_API_KEY", "~/.api_keys/openai_key.env"),
                  ("ANTHROPIC_API_KEY", "~/.api_keys/anthropic_key.env")]:
    if not os.environ.get(key):
        p = os.path.expanduser(path)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"export {key}="):
                        os.environ[key] = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

CHANGE_RE = re.compile(r"(chang|switch|instead|no longer|update)", re.IGNORECASE)
FINAL_RE = re.compile(r"(final|ultimately|locking in|locked in|settled on|decided on|final choice|final decision)", re.IGNORECASE)


def wb(text, term):
    return bool(re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE))


def dump_store(m, user_id):
    resp = m.get_all(filters={"user_id": user_id})
    mems = resp.get("results", []) if isinstance(resp, dict) else resp
    return sorted((mem.get("memory") or mem.get("text") or str(mem)) for mem in mems)


def snapshot_metrics(store_texts, prev_texts, incoming, outgoing, expected_current, all_values):
    joined = " \n ".join(store_texts)
    step_marker = any(
        (wb(t, outgoing) and wb(t, incoming)) or (wb(t, incoming) and CHANGE_RE.search(t))
        for t in store_texts
    ) if outgoing else True
    wrong_final = any(
        FINAL_RE.search(t) and not wb(t, expected_current) and any(wb(t, v) for v in all_values if v != expected_current)
        for t in store_texts
    )
    correct_final = any(FINAL_RE.search(t) and wb(t, expected_current) for t in store_texts)
    return {
        "store_changed": store_texts != prev_texts,
        "incoming_present": wb(joined, incoming),
        "step_marker": bool(step_marker),
        "wrong_final": wrong_final,
        "correct_final": correct_final,
        "n_memories": len(store_texts),
    }


def ask_answer(ctx_items, query):
    ctx = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(ctx_items))
    prompt = f"Below is retrieved context from a conversation. Answer the question using ONLY this context. Respond in one short word/phrase, just the answer with no explanation.\n\nContext:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": 50, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(5):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                     headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                              "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())["content"][0]["text"].strip()
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("answer API retries exhausted")


def main():
    K = int(os.environ.get("K", "8"))
    convs_path = os.environ.get("CONVS_IN", f"./exp_tmp/synth_chain_k{K}_convs.json")
    items = json.load(open(convs_path))
    nlim = int(os.environ.get("NLIM", "0"))
    if nlim > 0:
        items = items[:nlim]
    print(f"Loaded {len(items)} chain k={K} conversations (segment-wise instrumented arm)", flush=True)

    from mem0 import Memory
    run_dir = os.path.join(os.path.expanduser("~/.mem0_runs"), f"snapshot_k{K}_{os.getpid()}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    config = {
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini-2024-07-18", "temperature": 0.0}},
        "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(run_dir, "qdrant"), "collection_name": "chain", "embedding_model_dims": 1536}},
        "history_db_path": os.path.join(run_dir, "history.db"),
    }
    m = Memory.from_config(config_dict=config)
    print(f"Mem0 initialized (run dir {run_dir})", flush=True)

    per_sample = []
    for i, item in enumerate(items):
        user_id = f"snap_user_{i}"
        Xs = item["Xs"]
        turns = item["turns"]
        bounds = []
        prev = -1
        for v in Xs:
            first = next(j for j, t in enumerate(turns) if t["role"] == "user" and wb(t["content"], v))
            assert first > prev, f"non-monotone boundary conv {i}"
            bounds.append(first)
            prev = first
        bounds.append(len(turns))

        sample = {"idx": item["idx"], "Xs": Xs, "query": item["query"], "steps": []}
        prev_texts = []
        try:
            for r in range(len(Xs)):
                seg = turns[bounds[r]:bounds[r + 1]]
                m.add(messages=seg, user_id=user_id)
                store = dump_store(m, user_id)
                met = snapshot_metrics(store, prev_texts, Xs[r], Xs[r - 1] if r > 0 else "", Xs[r], Xs)
                met["r"] = r
                met["store"] = store
                sample["steps"].append(met)
                prev_texts = store
        except Exception as e:
            sample["error"] = f"ingest: {str(e)[:200]}"
            per_sample.append(sample)
            print(f"  [conv {i}] INGEST ERROR: {str(e)[:120]}", flush=True)
            continue

        try:
            resp = m.search(query=item["query"], filters={"user_id": user_id}, limit=10)
            mems = resp.get("results", []) if isinstance(resp, dict) else resp
            retrieved = [(mem.get("memory") or mem.get("text") or str(mem)) for mem in mems]
        except Exception as e:
            sample["error"] = f"search: {str(e)[:200]}"
            per_sample.append(sample)
            continue
        cur = Xs[-1]
        prior = Xs[:-1]
        sample["retrieved_top5"] = retrieved[:5]
        sample["current_in_top5"] = any(wb(t, cur) for t in retrieved[:5])
        try:
            ans = ask_answer(retrieved[:5], item["query"])
        except Exception as e:
            sample["error"] = f"answer: {str(e)[:200]}"
            per_sample.append(sample)
            continue
        y = wb(ans, cur)
        x = any(wb(ans, p) for p in prior)
        sample["e2e_answer"] = ans
        sample["e2e_category"] = "y_correct" if y and not x else ("x_distractor" if x and not y else "neither")

        steps = sample["steps"][1:]
        sample["all_steps_incoming"] = all(s["incoming_present"] for s in steps)
        sample["all_steps_marker"] = all(s["step_marker"] for s in steps)
        sample["first_missing_step"] = next((s["r"] for s in steps if not s["incoming_present"]), None)
        per_sample.append(sample)
        print(f"  [{i+1}/{len(items)}] cat={sample['e2e_category']} allL2={sample['all_steps_incoming']} allL3={sample['all_steps_marker']} cur@5={sample['current_in_top5']}", flush=True)

    ok = [s for s in per_sample if "error" not in s]
    n = len(ok)
    depth_rates = {}
    for r in range(1, len(items[0]["Xs"])):
        seen = [s["steps"][r] for s in ok if len(s["steps"]) > r]
        depth_rates[r] = {
            "incoming_present": sum(1 for x in seen if x["incoming_present"]),
            "step_marker": sum(1 for x in seen if x["step_marker"]),
            "wrong_final": sum(1 for x in seen if x["wrong_final"]),
            "n": len(seen),
        }
    cats = {"y_correct": 0, "x_distractor": 0, "neither": 0}
    comp = {"stale_all_local_ok": 0, "stale_local_fail": 0, "stale_local_fail_retrieved_ok": 0}
    for s in ok:
        cats[s["e2e_category"]] += 1
        if s["e2e_category"] == "x_distractor":
            if s["all_steps_incoming"] and s["all_steps_marker"]:
                comp["stale_all_local_ok"] += 1
            else:
                comp["stale_local_fail"] += 1
                if s["current_in_top5"]:
                    comp["stale_local_fail_retrieved_ok"] += 1
    out = {
        "config": {"n": len(items), "n_processed": n, "k": K, "arm": "segmentwise_snapshot",
                   "ingest_granularity": "one m.add per revision segment (9 calls per conversation)",
                   "extraction": "gpt-4o-mini-2024-07-18@t0", "answering": "claude-haiku-4-5-20251001@t0"},
        "e2e": cats,
        "final_current_in_top5": sum(1 for s in ok if s["current_in_top5"]),
        "depth_rates": depth_rates,
        "composition_split": comp,
        "per_sample": per_sample,
    }
    out_path = os.environ.get("OUT", f"/tmp/mem0_chain_snapshot_k{K}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"e2e: {cats}  cur@5: {out['final_current_in_top5']}/{n}", flush=True)
    print(f"depth incoming_present: {[ (r, v['incoming_present'], v['n']) for r, v in depth_rates.items() ]}", flush=True)
    print(f"composition split: {comp}", flush=True)
    print(f"Report: {out_path}", flush=True)


if __name__ == "__main__":
    main()
