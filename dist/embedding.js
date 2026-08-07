import { Worker } from "worker_threads";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
// Inference runs in a worker thread. ONNX is a synchronous native call, so on
// the main thread a single forward pass froze the event loop: the hook's health
// probe went unanswered, the rescue path read that as a dead server, and a
// perfectly healthy process was killed mid-request. Off-thread inference keeps
// the socket answering while the model works.
const WORKER_PATH = join(dirname(fileURLToPath(import.meta.url)), "embedding_worker.js");
const MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1";
// The tokenizer truncates at the model's 512-token limit, so text beyond a few
// thousand characters is discarded anyway. Cutting it here saves tokenizer work
// on large pastes without changing the resulting vector.
const MAX_INPUT_CHARS = 3000;
// One job at a time keeps ordering predictable. Bulk indexing is split into
// small batches so a query arriving mid-index waits for one sub-batch instead
// of the whole corpus.
const SUB_BATCH = 16;
const JOB_TIMEOUT_MS = 30000;
const MAX_RESPAWNS = 3;
let worker = null;
let ready = false;
let starting = null;
let startResolve = null;
let respawns = 0;
let nextId = 1;
let busy = false;
const hot = [];
const cold = [];
const inflight = new Map();
// The worker holds the process open only while it owes someone an answer. Without
// this the server is unaffected (its socket keeps the loop alive) but every
// standalone script under scripts/ hangs after finishing its work, because an
// idle worker's message port stays referenced.
function updateRef() {
    if (!worker)
        return;
    const pending = hot.length + cold.length + inflight.size + (ready ? 0 : 1);
    if (pending > 0)
        worker.ref();
    else
        worker.unref();
}
function settle(id, vecs) {
    const entry = inflight.get(id);
    if (!entry)
        return;
    clearTimeout(entry.timer);
    inflight.delete(id);
    entry.job.resolve(vecs ?? entry.job.texts.map(() => null));
    busy = false;
    pump();
    updateRef();
}
function pump() {
    if (busy || !ready || !worker)
        return;
    const job = hot.shift() ?? cold.shift();
    if (!job)
        return;
    busy = true;
    const id = nextId++;
    const timer = setTimeout(() => {
        console.error(`[this repository] Embedding job ${id} timed out after ${JOB_TIMEOUT_MS}ms.`);
        settle(id, null);
    }, JOB_TIMEOUT_MS);
    inflight.set(id, { job, timer });
    worker.postMessage({ id, texts: job.texts });
}
function drainOnDeath() {
    for (const [id, entry] of inflight) {
        clearTimeout(entry.timer);
        entry.job.resolve(entry.job.texts.map(() => null));
        inflight.delete(id);
    }
    while (hot.length)
        hot.shift().resolve([]);
    while (cold.length)
        cold.shift().resolve([]);
    busy = false;
}
function handleDeath(reason) {
    if (!worker)
        return;
    console.error(`[this repository] Embedding worker down (${reason}).`);
    ready = false;
    worker = null;
    drainOnDeath();
    if (respawns < MAX_RESPAWNS) {
        respawns++;
        console.error(`[this repository] Respawning embedding worker (attempt ${respawns}).`);
        spawn();
    }
    else {
        console.error(`[this repository] Embedding worker respawn limit reached; retrieval continues without vectors.`);
        startResolve?.();
        startResolve = null;
    }
}
function spawn() {
    const w = new Worker(WORKER_PATH);
    worker = w;
    w.on("message", (msg) => {
        if (msg?.type === "ready") {
            ready = true;
            respawns = 0;
            console.error("[this repository] Embedding model loaded (worker thread).");
            startResolve?.();
            startResolve = null;
            pump();
            updateRef();
            return;
        }
        if (msg?.type === "failed") {
            console.error(`[this repository] Failed to load embedding model:`, msg.error);
            startResolve?.();
            startResolve = null;
            return;
        }
        settle(msg.id, msg.ok ? msg.vecs : null);
    });
    w.on("error", (err) => handleDeath(String(err)));
    w.on("exit", (code) => {
        if (worker === w)
            handleDeath(`exit code ${code}`);
    });
}
function run(texts, priority) {
    if (!ready || texts.length === 0)
        return Promise.resolve(texts.map(() => null));
    const clipped = texts.map((t) => (t.length > MAX_INPUT_CHARS ? t.slice(0, MAX_INPUT_CHARS) : t));
    return new Promise((resolve) => {
        (priority === 0 ? hot : cold).push({ texts: clipped, resolve });
        updateRef();
        pump();
    });
}
function toBuffer(v) {
    return v ? Buffer.from(v.buffer, v.byteOffset, v.byteLength) : null;
}
export function initEmbedding() {
    if (starting)
        return starting;
    starting = new Promise((resolve) => {
        startResolve = resolve;
        console.error(`[this repository] Loading embedding model: ${MODEL_NAME}...`);
        spawn();
    });
    return starting;
}
export function isEmbeddingReady() {
    return ready;
}
export async function embed(text) {
    const [v] = await run([text], 0);
    return toBuffer(v ?? null);
}
export async function embedBatch(texts) {
    if (!ready || texts.length === 0)
        return texts.map(() => null);
    const out = [];
    for (let i = 0; i < texts.length; i += SUB_BATCH) {
        const vecs = await run(texts.slice(i, i + SUB_BATCH), 1);
        for (let j = 0; j < Math.min(SUB_BATCH, texts.length - i); j++) {
            out.push(toBuffer(vecs[j] ?? null));
        }
    }
    return out;
}
export function embeddingDimension() {
    return 1024; // mxbai-embed-large-v1 outputs 1024-dim vectors
}
export function bufferToFloat32Array(buf) {
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
//# sourceMappingURL=embedding.js.map