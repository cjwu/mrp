import { parentPort } from "worker_threads";
import { pipeline } from "@huggingface/transformers";
const MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1";
let extractor = null;
async function load() {
    // Single-threaded ONNX is kept deliberately. The intra-op pool's pthread
    // mutex throws an uncatchable `mutex lock failed: Invalid argument` when the
    // server runs as a nohup-detached self-rescue process. Moving inference into
    // this worker fixes event-loop blocking; it does not make the native thread
    // pool safe, so the pool stays disabled.
    // @ts-ignore - Transformers.js pipeline types are too complex for TS
    extractor = await pipeline("feature-extraction", MODEL_NAME, {
        dtype: "fp32",
        session_options: {
            intraOpNumThreads: 1,
            interOpNumThreads: 1,
        },
    });
}
function splitVectors(output, n) {
    const data = output.data;
    const dim = data.length / n;
    const vecs = [];
    for (let i = 0; i < n; i++) {
        // slice() copies into a standalone buffer, which is what the transfer list
        // below needs; a subarray view would share the model output buffer.
        vecs.push(new Float32Array(data.slice(i * dim, (i + 1) * dim)));
    }
    return vecs;
}
parentPort.on("message", async (msg) => {
    const id = msg?.id;
    const texts = msg?.texts ?? [];
    if (!extractor || texts.length === 0) {
        parentPort.postMessage({ id, ok: false, error: "extractor not ready" });
        return;
    }
    try {
        const output = await extractor(texts, { pooling: "cls", normalize: true });
        const vecs = splitVectors(output, texts.length);
        parentPort.postMessage({ id, ok: true, vecs }, vecs.map((v) => v.buffer));
    }
    catch (err) {
        parentPort.postMessage({ id, ok: false, error: String(err) });
    }
});
load().then(() => parentPort.postMessage({ type: "ready" }), (err) => parentPort.postMessage({ type: "failed", error: String(err) }));
//# sourceMappingURL=embedding_worker.js.map