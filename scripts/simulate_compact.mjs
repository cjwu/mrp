// Simulate Claude Code compact on D7 long-sparse convs.
// Splits each conv into [head_to_compact, recent_keep], summarizes head via Haiku 4.5,
// outputs new convs with compact summary as system turn + recent turns intact.
// Then run mem0_chain_baseline.py on this output to measure Mem0-on-compact degradation.

import { writeFileSync, readFileSync } from "fs";

const KEY_FILE = `${process.env.HOME}/.api_keys/anthropic_key.env`;
let ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
if (!ANTHROPIC_API_KEY) {
  const txt = readFileSync(KEY_FILE, "utf8");
  for (const line of txt.split("\n")) {
    const m = line.match(/^export\s+ANTHROPIC_API_KEY=["']?(.+?)["']?$/);
    if (m) { ANTHROPIC_API_KEY = m[1]; break; }
  }
}

const CONVS_IN  = process.env.CONVS_IN  ?? "/tmp/design_D7_n50_convs.json";
const OUT_PATH  = process.env.OUT       ?? "/tmp/design_D7_n50_compacted_convs.json";
const RECENT_K  = Number(process.env.RECENT_K ?? 20);   // keep last N turns intact
const MODEL     = "claude-haiku-4-5-20251001";
const MAX_PARAL = Number(process.env.MAX_PARALLEL ?? 2);
// Hierarchical compaction: chunk turn cap. ~88 tokens/turn observed on HB; leave headroom under 200k limit.
const CHUNK_TURN_CAP = Number(process.env.CHUNK_TURN_CAP ?? 1500);

const COMPACT_PROMPT = (transcript) => `You are summarizing an earlier portion of a user-assistant conversation that has scrolled out of context (a "compact" operation in Claude Code style).

Your goal: write a concise narrative summary (3-6 sentences) preserving:
1. The user's stated commitments, decisions, and plans
2. Any changes / retractions / replacements the user has made along the way
3. Key topics discussed (briefly)

DO NOT include verbatim quotes. DO NOT speculate beyond what is stated. Output only the summary text, no preamble.

Transcript:
${transcript}

Summary:`;

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function compactCall(transcript) {
  let lastStatus = null, lastBody = null;
  for (let attempt = 0; attempt < 6; attempt++) {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({ model: MODEL, max_tokens: 600, messages: [{ role: "user", content: COMPACT_PROMPT(transcript) }] }),
    });
    if (resp.ok) return (await resp.json()).content[0].text.trim();
    lastStatus = resp.status;
    lastBody = (await resp.text()).slice(0, 300);
    console.error(`  [compactCall] attempt ${attempt + 1}/6 failed: ${lastStatus} ${lastBody}`);
    if (lastStatus === 429 || lastStatus === 503 || lastStatus === 529) { await sleep(8000 * (attempt + 1)); continue; }
    throw new Error(`API ${lastStatus}: ${lastBody}`);
  }
  throw new Error(`compact API exhausted retries — last: ${lastStatus} ${lastBody}`);
}

const items = JSON.parse(readFileSync(CONVS_IN, "utf8"));
console.log(`Compacting ${items.length} convs from ${CONVS_IN}, keeping last ${RECENT_K} turns raw`);

const out = new Array(items.length);

async function summarizeHierarchical(head) {
  if (head.length <= CHUNK_TURN_CAP) {
    const transcript = head.map(t => `${t.role}: ${t.content}`).join("\n");
    return { summary: await compactCall(transcript), levels: 1, chunks: 1 };
  }
  const nChunks = Math.ceil(head.length / CHUNK_TURN_CAP);
  const chunkSize = Math.ceil(head.length / nChunks);
  const chunkSummaries = [];
  for (let i = 0; i < nChunks; i++) {
    const chunk = head.slice(i * chunkSize, (i + 1) * chunkSize);
    const transcript = chunk.map(t => `${t.role}: ${t.content}`).join("\n");
    chunkSummaries.push(await compactCall(transcript));
  }
  const meta = chunkSummaries.map((s, i) => `Segment ${i + 1}/${nChunks}: ${s}`).join("\n\n");
  const finalSummary = await compactCall(meta);
  return { summary: finalSummary, levels: 2, chunks: nChunks };
}

async function processOne(idx) {
  const item = items[idx];
  const turns = item.turns;
  if (turns.length <= RECENT_K) {
    out[idx] = { ...item, turns, _compact_skipped: true };
    return;
  }
  const head = turns.slice(0, turns.length - RECENT_K);
  const tail = turns.slice(turns.length - RECENT_K);
  const { summary, levels, chunks } = await summarizeHierarchical(head);
  const compactedTurns = [
    { role: "user", content: `[Previous conversation summary, compacted from ${head.length} turns]: ${summary}` },
    ...tail,
  ];
  out[idx] = { ...item, turns: compactedTurns, _compact_orig_len: turns.length, _compact_kept: tail.length, _compact_summary_chars: summary.length, _compact_levels: levels, _compact_chunks: chunks };
}

let inFlight = 0, next = 0, done = 0;
await new Promise((resolve, reject) => {
  const tick = () => {
    while (inFlight < MAX_PARAL && next < items.length) {
      const idx = next++;
      inFlight++;
      processOne(idx).then(() => {
        inFlight--; done++;
        if (done % 10 === 0 || done === items.length) console.log(`  [${done}/${items.length}] compacted`);
        if (done === items.length) resolve();
        else tick();
      }).catch(err => reject(err));
    }
  };
  tick();
});

writeFileSync(OUT_PATH, JSON.stringify(out, null, 2));
console.log(`\nSaved compacted convs to ${OUT_PATH}`);
const sample = out[0];
console.log(`\nSample[0]:`);
console.log(`  orig turns: ${sample._compact_orig_len}, kept tail: ${sample._compact_kept}`);
console.log(`  new turns count: ${sample.turns.length}`);
console.log(`  compact summary preview: ${sample.turns[0].content.slice(0, 240)}...`);
