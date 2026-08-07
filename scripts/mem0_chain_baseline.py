"""Mem0 baseline on chain-retraction k=3 synth.

Reads /tmp/synth_chain_k3_convs.json (built by synth_chain_retract.mjs).
Ingests each conversation through Mem0, queries, scores Hit@K (X3 mentioned)
+ Distractor@K (X1 or X2 mentioned).

Then E2E with Claude haiku-4-5 for paired comparison with cosine/BAF.
"""

import os
import json
import time
import sys

# Auto-load API keys
for key, path in [("OPENAI_API_KEY", "~/.api_keys/openai_key.env"),
                  ("ANTHROPIC_API_KEY", "~/.api_keys/anthropic_key.env")]:
    if not os.environ.get(key):
        p = os.path.expanduser(path)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"export {key}="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ[key] = v
                        break


def main():
    K = int(os.environ.get("K", "3"))
    convs_path = os.environ.get("CONVS_IN", f"/tmp/synth_chain_k{K}_convs.json")
    if not os.path.exists(convs_path):
        print(f"Run synth_chain_retract.mjs first to produce {convs_path}", flush=True)
        sys.exit(1)
    items = json.load(open(convs_path))
    print(f"Loaded {len(items)} chain-k3 conversations", flush=True)

    from mem0 import Memory
    base_dir = os.path.expanduser("~/.mem0_runs")
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, f"chain_k{K}_run_{os.getpid()}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    config = {
        # Snapshot pinned 2026-07-27. "gpt-4o-mini" is a moving alias: the extraction
        # behaviour behind it changed between the April run and this one, which shifted
        # Mem0's retrieval ranking on long chains. A dated snapshot keeps the ingest
        # side comparable across future re-runs.
        "llm": { "provider": "openai", "config": { "model": "gpt-4o-mini-2024-07-18", "temperature": 0.0 } },
        "vector_store": { "provider": "qdrant", "config": { "path": os.path.join(run_dir, "qdrant"), "collection_name": "chain", "embedding_model_dims": 1536 } },
        "history_db_path": os.path.join(run_dir, "history.db"),
    }
    print(f"Initializing Mem0 (chain k={K})...", flush=True)
    m = Memory.from_config(config_dict=config)

    # Ingest + query
    agg = {1:0, 3:0, 5:0, 10:0}
    dist_agg = {1:0, 3:0, 5:0, 10:0}
    per_sample = []
    for i, item in enumerate(items):
        user_id = f"chain_user_{i}"
        try:
            INGEST_BATCH = int(os.environ.get("INGEST_BATCH", "50"))
            MAX_TURN_CHARS = int(os.environ.get("MAX_TURN_CHARS", "0"))  # 0 = no truncation
            turns = item["turns"]
            if MAX_TURN_CHARS > 0:
                # Pre-truncate each turn's content to MAX_TURN_CHARS
                turns = [
                    {**t, "content": (t.get("content") or "")[:MAX_TURN_CHARS]} if isinstance(t, dict) else t
                    for t in turns
                ]
            for bi in range(0, len(turns), INGEST_BATCH):
                m.add(messages=turns[bi:bi+INGEST_BATCH], user_id=user_id)
        except Exception as e:
            print(f"  [ingest error i={i}]: {str(e)[:200]}", flush=True)
            per_sample.append({"idx":i, "error":f"ingest: {e}"})
            continue
        try:
            resp = m.search(query=item["query"], filters={"user_id": user_id}, limit=10)
            mems = resp.get("results", []) if isinstance(resp, dict) else resp
        except Exception as e:
            print(f"  [search error i={i}]: {str(e)[:200]}", flush=True)
            per_sample.append({"idx":i, "error":f"search: {e}"})
            continue
        retrieved = [(mem.get("memory") or mem.get("text") or str(mem)) for mem in mems]
        Xs = item.get("Xs") or [item["X1"], item.get("X2",""), item["X3"]]
        Xfinal = Xs[-1]
        Xprior = [x for x in Xs[:-1] if x]
        for k in [1,3,5,10]:
            top_k = retrieved[:k]
            if any(Xfinal.lower() in t.lower() for t in top_k):
                agg[k] += 1
            if any(any(x.lower() in t.lower() for x in Xprior) for t in top_k):
                dist_agg[k] += 1
        per_sample.append({"idx":i, "Xs":Xs, "X1":Xs[0], "X2":Xs[1] if len(Xs)>1 else "", "X3":Xfinal, "retrieved_top5": retrieved[:5]})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] hit@5: {agg[5]}/{i+1}, dist@5: {dist_agg[5]}/{i+1}", flush=True)

    n = sum(1 for s in per_sample if "error" not in s)

    # E2E LLM accuracy on Mem0 retrievals
    import urllib.request
    e2e_agg = {"y_correct":0, "x_distractor":0, "neither":0}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    for i, s in enumerate(per_sample):
        if "error" in s: continue
        ctx = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(s["retrieved_top5"]))
        # rebuild query from item record
        item = items[s["idx"]]
        prompt = f"Below is retrieved context from a conversation. Answer the question using ONLY this context. Respond in one short word/phrase, just the answer with no explanation.\n\nContext:\n{ctx}\n\nQuestion: {item['query']}\n\nAnswer:"
        # temperature pinned 2026-07-27; see the note in synth_chain_retract.mjs.
        body = json.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": 50, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
        ans = None
        last_err = None
        for attempt in range(5):
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    data = json.loads(r.read())
                ans = data["content"][0]["text"].strip()
                break
            except Exception as e:
                last_err = str(e)
                if "429" in last_err or "503" in last_err:
                    time.sleep(5 * (attempt + 1))
                    continue
                break
        if ans is None:
            s["e2e_error"] = last_err
            continue
        import re as _re
        Xs_local = s.get("Xs") or [s["X1"], s.get("X2",""), s["X3"]]
        Xfinal = Xs_local[-1]
        Xprior = [x for x in Xs_local[:-1] if x]
        def _wb(text, term):
            return bool(_re.search(r"\b" + _re.escape(term) + r"\b", text, _re.IGNORECASE))
        y = _wb(ans, Xfinal)
        x = any(_wb(ans, xp) for xp in Xprior)
        if y and not x: cat = "y_correct"
        elif x and not y: cat = "x_distractor"
        else: cat = "neither"
        e2e_agg[cat] += 1
        s["e2e_answer"] = ans
        s["e2e_category"] = cat
        if (i+1) % 10 == 0:
            print(f"  E2E [{i+1}/{n}] y={e2e_agg['y_correct']} x={e2e_agg['x_distractor']}", flush=True)

    safe_n = max(n, 1)
    out = {
        "config": {"n": len(items), "n_processed": n, "k_retractions": K},
        "retrieval_hit_pct": {k: round(agg[k]/safe_n*100, 2) for k in [1,3,5,10]},
        "retrieval_distractor_pct": {k: round(dist_agg[k]/safe_n*100, 2) for k in [1,3,5,10]},
        "e2e_llm": {k: round(v/safe_n*100, 2) for k,v in e2e_agg.items()},
        "per_sample": per_sample,
    }
    out_path = os.environ.get("OUT", f"/tmp/mem0_chain_k{K}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== Mem0 chain-retract k=3 n={n} ===", flush=True)
    print(f"Retrieval Hit@K: {out['retrieval_hit_pct']}", flush=True)
    print(f"Retrieval Distractor@K: {out['retrieval_distractor_pct']}", flush=True)
    print(f"E2E LLM: {out['e2e_llm']}", flush=True)
    print(f"\nReport: {out_path}", flush=True)


if __name__ == "__main__":
    main()
