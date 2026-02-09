/*
 * gpu-substitute.cl -- Class substitution on GPU-resident atom pools
 *
 * After clustering merges words into classes, this kernel scans
 * PairPool and SectionPool, replacing word indices with class IDs.
 * Duplicate pairs (same class pair after substitution) are merged
 * by rebuilding the pair hash table.
 *
 * Pipeline:
 *   1. assign_classes          — batch set word_class_id for merged words
 *   2. substitute_pairs        — replace word indices with class IDs in pairs
 *   3. rebuild_pair_index      — rebuild HT, merge duplicate pairs
 *   4. substitute_section_words — replace section head words with class IDs
 *
 * Appended after gpu-hashtable.cl, gpu-atomspace.cl at load time.
 */

/* ═══════════════════════════════════════════════════════════════
 *  STEP 1: ASSIGN CLASSES
 *
 *  Batch-set word_class_id for a list of words.
 *  Called after each clustering round with the new merges.
 *
 *  class_id = 0 means "no class" (unclassified).
 * ═══════════════════════════════════════════════════════════════ */

__kernel void assign_classes(
    __global uint*       word_class_id,
    __global const uint* in_word_indices,
    __global const uint* in_class_ids,
    const uint           num_assignments)
{
    uint tid = get_global_id(0);
    if (tid >= num_assignments) return;

    uint word_idx = in_word_indices[tid];
    uint class_id = in_class_ids[tid];
    if (word_idx < WORD_CAPACITY)
        word_class_id[word_idx] = class_id;
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 2: SUBSTITUTE PAIRS
 *
 *  One thread per pair. For each pair:
 *    - Look up class_id for word_a and word_b
 *    - If classified (class_id != 0): replace with class_id
 *    - Maintain canonical order (lo < hi)
 *    - Self-pairs (both words → same class) are eliminated
 *    - Mark changed pairs as dirty for MI recompute
 *
 *  After this kernel, the pair hash table is INVALID (keys changed).
 *  Must run rebuild_pair_index to restore it.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void substitute_pairs(
    __global uint*           pair_word_a,
    __global uint*           pair_word_b,
    __global volatile double* pair_count,
    __global double*         pair_mi,
    __global uint*           pair_flags,
    __global const uint*     word_class_id,
    __global volatile uint*  num_changed,
    __global volatile uint*  num_eliminated,
    const uint               num_pairs)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    double count = pair_count[tid];
    if (count < 0.5) return;  /* skip empty pairs */

    uint wa = pair_word_a[tid];
    uint wb = pair_word_b[tid];

    uint ca = word_class_id[wa];
    uint cb = word_class_id[wb];

    uint new_a = (ca != 0) ? ca : wa;
    uint new_b = (cb != 0) ? cb : wb;

    /* Self-pair after substitution: both words map to same class → drop */
    if (new_a == new_b) {
        pair_count[tid] = 0.0;
        pair_mi[tid] = 0.0;
        pair_flags[tid] = 0;
        atomic_add(num_eliminated, 1U);
        return;
    }

    /* Canonical order */
    uint lo = min(new_a, new_b);
    uint hi = max(new_a, new_b);

    if (lo != wa || hi != wb) {
        pair_word_a[tid] = lo;
        pair_word_b[tid] = hi;
        pair_flags[tid] = 1;  /* dirty: needs MI recompute */
        atomic_add(num_changed, 1U);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 3: REBUILD PAIR INDEX
 *
 *  After substitute_pairs changed pair keys, the pair hash table
 *  is stale. This kernel rebuilds it AND merges duplicate pairs.
 *
 *  Algorithm: one thread per pair.
 *    - Compute key from (word_a, word_b)
 *    - CAS into hash table
 *    - If we claim the slot: we're the "primary" copy
 *    - If slot already has our key: duplicate!
 *      → atomically add our count to the primary's count
 *      → zero ourselves (mark as merged)
 *
 *  IMPORTANT: The pair hash table must be cleared (filled with
 *  0xFF) before running this kernel.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void rebuild_pair_index(
    __global const uint*      pair_word_a,
    __global const uint*      pair_word_b,
    __global volatile double* pair_count,
    __global double*          pair_mi,
    __global uint*            pair_flags,
    __global volatile ulong*  pht_keys,
    __global volatile uint*   pht_values,
    __global volatile uint*   num_merged,
    const uint                num_pairs)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    double count = pair_count[tid];
    if (count < 0.5) return;  /* skip empty/eliminated pairs */

    uint wa = pair_word_a[tid];
    uint wb = pair_word_b[tid];
    uint lo = min(wa, wb);
    uint hi = max(wa, wb);
    ulong key = ((ulong)lo << 32) | (ulong)hi;

    ulong cap = PAIR_HT_CAPACITY;
    ulong mask = cap - 1;
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(&pht_keys[slot], HT_EMPTY_KEY, key);

        if (prev == HT_EMPTY_KEY)
        {
            /* We claim this slot — we're the primary copy */
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            pht_values[slot] = tid;
            return;
        }
        if (prev == key)
        {
            /* Duplicate! Merge into the primary. */
            uint primary = pht_values[slot];
            while (primary == HT_EMPTY_VALUE) {
                primary = pht_values[slot];
            }

            /* Add our count to primary */
            atomic_add_double(&pair_count[primary], count);

            /* Mark primary as dirty */
            pair_flags[primary] = 1;

            /* Zero ourselves */
            pair_count[tid] = 0.0;
            pair_mi[tid] = 0.0;
            pair_flags[tid] = 0;

            atomic_add(num_merged, 1U);
            return;
        }

        slot = (slot + 1) & mask;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 4: SUBSTITUTE SECTION WORDS
 *
 *  Replace section head word with its class ID.
 *
 *  NOTE: This only replaces the section's word field. The
 *  disjunct_hash (which encodes connector words) is NOT updated
 *  here. Full connector rewriting happens naturally on the next
 *  call to extract_sections (Phase 4), which uses the updated
 *  word pool indices. This matches the iterative pipeline:
 *  each round substitutes, then re-extracts fresh sections.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void substitute_section_words(
    __global uint*          sec_word,
    __global const double*  sec_count,
    __global const uint*    word_class_id,
    __global volatile uint* num_changed,
    const uint              num_sections)
{
    uint tid = get_global_id(0);
    if (tid >= num_sections) return;

    if (sec_count[tid] < 0.5) return;

    uint word = sec_word[tid];
    uint cls = word_class_id[word];

    if (cls != 0 && cls != word) {
        sec_word[tid] = cls;
        atomic_add(num_changed, 1U);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  READBACK: Dump class assignments and pair data
 * ═══════════════════════════════════════════════════════════════ */

__kernel void read_class_assignments(
    __global const uint* word_class_id,
    __global uint*       out_classes,
    const uint           num_words)
{
    uint tid = get_global_id(0);
    if (tid >= num_words) return;
    out_classes[tid] = word_class_id[tid];
}
