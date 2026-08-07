// MRP (Monotonic Retrieval Paradox) mitigation — MVP per D-MRP-1..12
//
// Pipeline:
//   Ingestion: detectRetraction(chunk) → tag is_retraction/target_keywords/weight
//   Query:     buildCarrierPool(query) + buildLazyEdges(candidates) + filterRetracted(results)
//
// Invariants (see paper/mrp.tex §4 and related_work/mrp/strategic_framing.md):
//   I1. Carriers are always role="user" (assistant utterances never carry retraction force).
//   I2. Time-order: an edge (carrier → target) requires target.ts < carrier.ts.
//   I3. D-MRP-11: carriers do not attack carriers (guarantees acyclicity).
//   I4. Timestamps are lexicographically comparable (ISO-8601 recommended).
//   I5. Edge weight follows paper Eq.1: w = force(carrier) · m(carrier,target) · f(target).
//       MVP uses m ∈ {0,1} via keyword-substring; Phase 2 adds m = max(kwHit, cosine).
//   I6. Grounded extension collapses to preferred=stable in the acyclic case
//       (Dung 1995 Theorem 30); membership reduces to single-pass "any-attacker-wins".
// ─── Numerical constants ──────────────────────────────────────────────────────
// All three compose via Eq.1. Changing any breaks ablation semantics reported
// in paper §5.3, so treat as fixed load-bearing constants.
const EDGE_THRESHOLD = 0.5; // θ in paper Eq.1
const SEARLE_FORCE_DECLARATION = 1.0; // Table 1, Declaration
const SEARLE_FORCE_DIRECTIVE = 0.9; // Table 1, Directive-negative
const SEARLE_FORCE_COMMISSIVE = 0.7; // Table 1, Commissive-reversal
const COMMIT_FACTOR_HIGH = 1.0; // target has commitment verb
const COMMIT_FACTOR_LOW = 0.4; // target is mere mention / chatter
const MAX_TOPIC_KEYWORDS = 5; // prior-context keyword cap
const SLOT_EXTRACT_WINDOW = 40; // chars after marker to scan for slot
// ─── Detection regex (D-MRP-8 Phase 1, zero-LLM) ──────────────────────────────
// Matched in priority order Declaration > Commissive-reversal > Directive.
// Declaration markers: force 1.0.
// English "cancel" forms cover US "canceled" / UK "cancelled" / noun forms.
const DECLARATION_RE = /(?:取消|撤回|作廢|廢除|hereby\s+(?:cancel|terminate|void)|\bcancell?ed\b|\bcancelling\b|\bcancel\b|\bcancellations?\b|\bterminate[ds]?\b|\bvoid(?:ed)?\b)/i;
// Commissive-reversal markers: force 0.7.
// Chinese 改 must be followed by an explicit reversal particle [成為去用做到]
// to distinguish reversal from neutral reference (e.g. "改目的地" / "改程式碼").
// English requires explicit "to" phrasing; "let's go with" / "pivot to" captured.
const COMMISSIVE_REVERSAL_RE = /(?:改成|改為|改去|改用|改做|改到|\bswitch(?:ing)?\s+to\b|\bswitched?\s+to\b|\bactually\s+let'?s\b|\bchange\s+to\b|\bpivot\s+to\b|\blet'?s\s+go\s+with\b)/i;
// Directive-negative: force 0.9.
// Covers "stop using X" and "don't use X" (both missed by earlier "do"-only
// variant), plus "forget about X" / "nevermind".
const DIRECTIVE_NEG_RE = /(?:別(?:做|用|要|去|管)|不要(?:做|用)|\bstop\s+(?:doing|using)\b|\bdon'?t\s+(?:do|use)\b|\bscrap\s+the\b|\bforget\s+about\b|\bnever\s?mind\b)/i;
// Negation prefix guard: chunk start with these markers → NOT a retraction.
// Chinese 先別 / 不是要 / 還沒要 are tentativity / meta-statement markers.
// NOTE (Phase 1 conservatism): this is a global gate, so a chunk like
// "先別做，改去 B" (legitimate Commissive-reversal following tentativity) is
// blocked. Phase 2 should replace with per-class negation detection.
const NEGATION_GUARD_RE = /^(?:.{0,8})(?:不是要|先別|還沒要|還沒有|暫時(?:不|別)|not\s+to\s+(?:cancel|terminate))/i;
// Conditional clause guard: hypothetical framing → not a live retraction.
const CONDITIONAL_RE = /(?:的話|的時候|假如|如果要|\bif\s+we\b|\bwhenever\b|\bsuppose\s+we\b)/i;
// Commitment verbs: D-MRP-9 D7 target-side commitment detector.
// Expanded "will/would/shall … use/do/go/book/take" variants to catch
// future-tense commitments ("we will use Paris"), not only past-tense ("定了").
const COMMITMENT_VERB_RE = /(?:決定|定了|訂了?|採用|確認|鎖定|採行|就用|就做|就是|\bbook(?:ed|ing)?\b|\bconfirm(?:ed)?\b|\bdecide[ds]?\b|\b(?:will|would|shall)\s+(?:use|do|go|book|take)\b|\bgoing\s+to\b|\bplan(?:ning)?\s+to\b|\blocked?\s+in\b)/i;
const EMPTY_TAG = Object.freeze({
    is_retraction: false,
    speech_act_class: "none",
    retraction_weight: 0,
    target_keywords: [],
    destination_slot: null,
});
/**
 * Classify a chunk as retraction carrier, non-carrier, or guarded.
 *
 * @param chunkText   The chunk's own text.
 * @param priorContext  Concatenated text of the preceding 1-3 chunks in the
 *                      same session, used to extract the RETRACTED target
 *                      when the carrier is a Commissive-reversal (whose verb-
 *                      following NP is the NEW destination, not the retracted
 *                      target). For Declaration and Directive the target is
 *                      extracted from the chunk itself.
 */
export function detectRetraction(chunkText, priorContext = "") {
    const text = chunkText.trim();
    if (text.length < 2)
        return { ...EMPTY_TAG };
    // Guards applied globally (Phase 1; see NEGATION_GUARD_RE note).
    if (CONDITIONAL_RE.test(text))
        return { ...EMPTY_TAG };
    if (/[?？]\s*$/.test(text))
        return { ...EMPTY_TAG }; // trailing question mark
    if (NEGATION_GUARD_RE.test(text))
        return { ...EMPTY_TAG };
    // Priority order: strongest class first.
    const declMatch = text.match(DECLARATION_RE);
    if (declMatch) {
        const kw = extractSlotAfter(text, declMatch.index + declMatch[0].length);
        if (kw.length === 0)
            return { ...EMPTY_TAG }; // no target → low confidence
        return {
            is_retraction: true,
            speech_act_class: "Declaration",
            retraction_weight: SEARLE_FORCE_DECLARATION,
            target_keywords: kw,
            destination_slot: null,
        };
    }
    const reversalMatch = text.match(COMMISSIVE_REVERSAL_RE);
    if (reversalMatch) {
        // Target is in PRIOR context (what is being retracted), not in the carrier
        // (which names the NEW destination). This asymmetry is D-MRP-8.
        const destSlot = extractSlotAfter(text, reversalMatch.index + reversalMatch[0].length);
        const priorKw = extractTopicKeywords(priorContext);
        if (priorKw.length === 0) {
            // No prior context → can't identify what's being retracted; Phase 1
            // conservative: skip rather than guess.
            return { ...EMPTY_TAG };
        }
        return {
            is_retraction: true,
            speech_act_class: "Commissive",
            retraction_weight: SEARLE_FORCE_COMMISSIVE,
            target_keywords: priorKw,
            destination_slot: destSlot.join(" ") || null,
        };
    }
    const dirMatch = text.match(DIRECTIVE_NEG_RE);
    if (dirMatch) {
        const kw = extractSlotAfter(text, dirMatch.index + dirMatch[0].length);
        if (kw.length === 0)
            return { ...EMPTY_TAG };
        return {
            is_retraction: true,
            speech_act_class: "Directive",
            retraction_weight: SEARLE_FORCE_DIRECTIVE,
            target_keywords: kw,
            destination_slot: null,
        };
    }
    return { ...EMPTY_TAG };
}
// Extract up to 3 slot tokens immediately following a marker at `offset`.
// Stop at sentence-level particle (的/了/吧/啊/嗎/呢), punctuation, or
// conjunction. Tokens must be length 2-20 chars after trim.
function extractSlotAfter(text, offset) {
    const slice = text.slice(offset, offset + SLOT_EXTRACT_WINDOW);
    const cleaned = slice.replace(/^[\s,，、：:]+/, "");
    const stopped = cleaned.split(/[的了吧啊嗎呢，。.!！？?;；:\s]|(?:\s(?:and|or|but|then)\s)/i)[0] ||
        "";
    const tokens = stopped
        .split(/[\s,，、/]/)
        .map((t) => t.trim())
        .filter((t) => t.length >= 2 && t.length <= 20);
    return tokens.slice(0, 3);
}
// Extract topical content tokens from joined prior context.
// Heuristic: length ≥ 2 chars, non-stopword, dedup order-preserving, cap at
// MAX_TOPIC_KEYWORDS. Phase 2 should replace with contextual-embedding recall.
function extractTopicKeywords(ctx) {
    if (!ctx)
        return [];
    const stopwords = new Set([
        "的", "了", "在", "是", "有", "和", "與", "也", "就", "都", "要", "去", "來",
        "the", "a", "an", "is", "are", "was", "were", "will", "be", "of", "to", "in",
        "for", "and", "or", "but", "we", "i", "it", "this", "that",
    ]);
    const tokens = ctx
        .split(/[\s,，、。.!！？?:;；()（）「」【】\[\]]+/)
        .map((t) => t.trim())
        .filter((t) => t.length >= 2 && t.length <= 20)
        .filter((t) => !stopwords.has(t.toLowerCase()));
    const seen = new Set();
    const out = [];
    for (const t of tokens) {
        const k = t.toLowerCase();
        if (seen.has(k))
            continue;
        seen.add(k);
        out.push(t);
        if (out.length >= MAX_TOPIC_KEYWORDS)
            break;
    }
    return out;
}
/**
 * Build attack edges per paper Eq.1: w = force · m · f.
 *
 * MVP uses m ∈ {0, 1} via case-insensitive keyword substring match.
 * Phase 2: m = max(kwHit, cos(carrier.phraseEmb, target.ctxEmb)).
 *
 * Complexity O(|carriers| · |candidates| · avg|text|), typical 6-8 ms at
 * |candidates| = 40 per paper §5.4.
 */
export function buildLazyEdges(candidates) {
    // Invariant I1: carriers are always user-role utterances.
    const carriers = candidates.filter((c) => c.is_retraction && c.role === "user");
    const edges = [];
    for (const carrier of carriers) {
        if (carrier.target_keywords.length === 0)
            continue;
        for (const target of candidates) {
            if (target.id === carrier.id)
                continue;
            // Invariant I3 (D-MRP-11): carriers don't attack carriers.
            if (target.is_retraction)
                continue;
            // Invariant I2: time-order.
            if (target.timestamp >= carrier.timestamp)
                continue;
            // Eq.1 term m: case-insensitive substring hit over target chunk text.
            const targetLower = target.chunk_text.toLowerCase();
            const kwHit = carrier.target_keywords.some((kw) => targetLower.includes(kw.toLowerCase()));
            const matchScore = kwHit ? 1.0 : 0.0;
            if (matchScore === 0.0)
                continue;
            // Eq.1 term f: commitment-aware multiplier (D-MRP-9 D7).
            const hasCommit = COMMITMENT_VERB_RE.test(target.chunk_text);
            const commitFactor = hasCommit ? COMMIT_FACTOR_HIGH : COMMIT_FACTOR_LOW;
            // Eq.1: w = force · m · f.
            const weight = carrier.retraction_weight * matchScore * commitFactor;
            if (weight < EDGE_THRESHOLD)
                continue;
            edges.push({
                attacker_id: carrier.id,
                target_id: target.id,
                weight,
            });
        }
    }
    return edges;
}
// ─── Grounded extension filter (paper §4.4) ───────────────────────────────────
/**
 * Filter candidates by grounded-extension membership.
 *
 * Under invariants I2 (time-order) + I3 (carriers-don't-attack-carriers) the
 * attack graph is acyclic, so by Dung 1995 Theorem 30 the grounded, preferred,
 * and stable extensions coincide. Membership is then: a node is OUT iff any
 * active (weight ≥ θ) attack edge targets it. No fixed-point iteration needed.
 *
 * The `e.weight >= EDGE_THRESHOLD` check is redundant (edges emitted by
 * buildLazyEdges already satisfy this) but kept as defence-in-depth against
 * callers that construct edges directly.
 */
export function filterByGroundedExtension(candidates, edges) {
    const attacked = new Set();
    for (const e of edges) {
        if (e.weight >= EDGE_THRESHOLD) {
            attacked.add(e.target_id);
        }
    }
    const survivors = [];
    const filtered = [];
    for (const c of candidates) {
        if (attacked.has(c.id))
            filtered.push(c);
        else
            survivors.push(c);
    }
    return { survivors, filtered };
}
//# sourceMappingURL=mrp.js.map