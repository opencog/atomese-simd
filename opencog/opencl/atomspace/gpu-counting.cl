/*
 * gpu-counting.cl -- Sentence processing and pair counting on GPU
 *
 * Takes batches of sentences (as word pool indices), generates all
 * word pairs within a sliding window, finds-or-creates each pair
 * in the pair pool, and atomically increments counts.
 *
 * This replaces cl-l0-count-pairs! from corpus-learning.scm.
 *
 * Appended after gpu-hashtable.cl and gpu-atomspace.cl at load time.
 */

/* ═══════════════════════════════════════════════════════════════
 *  SENTENCE PAIR COUNTING
 *
 *  One thread per word position. Each thread generates up to
 *  WINDOW_SIZE pairs with subsequent words in the same sentence.
 *  For each pair: find-or-create + atomic count increment.
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Process a batch of sentences: count all word pairs within window.
 *
 * Work assignment: one thread per word position across all sentences.
 * Thread i handles the word at flat_words[i], generating pairs with
 * the next WINDOW_SIZE words in the same sentence.
 *
 * Args:
 *   flat_words       - all sentences concatenated [total_words]
 *   sent_offsets      - start index of each sentence [num_sentences]
 *   sent_lengths      - length of each sentence [num_sentences]
 *   num_sentences     - number of sentences in batch
 *   total_words       - total words across all sentences
 *   window_size       - pair window (typically 6)
 *
 *   -- Pair pool + hash table --
 *   pht_keys, pht_values  - pair hash table
 *   pair_word_a, pair_word_b, pair_count, pair_mi, pair_flags
 *   pair_next_free
 *
 *   -- Word pool (marginals) --
 *   word_count
 *
 *   -- Global counter --
 *   total_pair_count  - total pairs counted (atomic)
 */
__kernel void count_sentence_pairs(
    __global const uint*     flat_words,
    __global const uint*     sent_offsets,
    __global const uint*     sent_lengths,
    const uint               num_sentences,
    const uint               total_words,
    const uint               window_size,
    /* pair hash table */
    __global volatile ulong* pht_keys,
    __global volatile uint*  pht_values,
    /* pair pool SoA */
    __global uint*           pair_word_a,
    __global uint*           pair_word_b,
    __global volatile double* pair_count,
    __global double*         pair_mi,
    __global volatile uint*  pair_flags,
    __global volatile uint*  pair_next_free,
    /* word pool */
    __global volatile double* word_count,
    /* global counter */
    __global volatile uint*  total_pair_count)
{
    uint tid = get_global_id(0);
    if (tid >= total_words) return;

    uint word_i = flat_words[tid];

    /* Find which sentence this word belongs to (linear scan — sentences
     * are small, typically <100 per batch, so this is fast) */
    uint sent_idx = 0;
    for (uint s = 0; s < num_sentences; s++) {
        if (tid >= sent_offsets[s] &&
            tid < sent_offsets[s] + sent_lengths[s]) {
            sent_idx = s;
            break;
        }
    }

    uint sent_start = sent_offsets[sent_idx];
    uint sent_len   = sent_lengths[sent_idx];
    uint pos_in_sent = tid - sent_start;

    /* Generate pairs with subsequent words within window */
    uint max_j = min(pos_in_sent + window_size, sent_len - 1);

    for (uint j = pos_in_sent + 1; j <= max_j; j++)
    {
        uint word_j = flat_words[sent_start + j];

        /* Skip self-pairs */
        if (word_i == word_j) continue;

        /* ── Find or create pair ── */
        uint lo = min(word_i, word_j);
        uint hi = max(word_i, word_j);
        ulong key = ((ulong)lo << 32) | (ulong)hi;

        ulong ht_cap = PAIR_HT_CAPACITY;
        ulong mask = ht_cap - 1;
        ulong slot = ht_hash(key) & mask;
        uint pair_idx = HT_EMPTY_VALUE;

        for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
        {
            ulong prev = atom_cmpxchg(&pht_keys[slot], HT_EMPTY_KEY, key);

            if (prev == HT_EMPTY_KEY)
            {
                /* New pair — allocate from pool */
                uint idx = atomic_add(pair_next_free, 1U);
                if (idx < PAIR_CAPACITY) {
                    pair_word_a[idx] = lo;
                    pair_word_b[idx] = hi;
                    pair_count[idx] = 0.0;
                    pair_mi[idx] = 0.0;
                    pair_flags[idx] = 0;
                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                    pht_values[slot] = idx;
                    pair_idx = idx;
                }
                break;
            }
            if (prev == key)
            {
                /* Existing pair — spin for value */
                uint val = pht_values[slot];
                while (val == HT_EMPTY_VALUE) {
                    val = pht_values[slot];
                }
                pair_idx = val;
                break;
            }

            slot = (slot + 1) & mask;
        }

        if (pair_idx == HT_EMPTY_VALUE) continue;

        /* ── Increment counts ── */

        /* Pair count */
        atomic_add_double(&pair_count[pair_idx], 1.0);

        /* Word marginals (both words) */
        atomic_add_double(&word_count[word_i], 1.0);
        atomic_add_double(&word_count[word_j], 1.0);

        /* Mark dirty */
        pair_flags[pair_idx] = 1;

        /* Total count */
        atomic_add(total_pair_count, 1U);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  BATCH SENTENCE PROCESSING — BINARY SEARCH VARIANT
 *
 *  For large batches (1000+ sentences), uses binary search instead
 *  of linear scan to find which sentence a word belongs to.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void count_sentence_pairs_large(
    __global const uint*     flat_words,
    __global const uint*     sent_offsets,
    __global const uint*     sent_lengths,
    const uint               num_sentences,
    const uint               total_words,
    const uint               window_size,
    __global volatile ulong* pht_keys,
    __global volatile uint*  pht_values,
    __global uint*           pair_word_a,
    __global uint*           pair_word_b,
    __global volatile double* pair_count,
    __global double*         pair_mi,
    __global volatile uint*  pair_flags,
    __global volatile uint*  pair_next_free,
    __global volatile double* word_count,
    __global volatile uint*  total_pair_count)
{
    uint tid = get_global_id(0);
    if (tid >= total_words) return;

    uint word_i = flat_words[tid];

    /* Binary search for sentence index */
    uint lo_s = 0, hi_s = num_sentences;
    while (lo_s < hi_s) {
        uint mid = (lo_s + hi_s) / 2;
        if (sent_offsets[mid] + sent_lengths[mid] <= tid)
            lo_s = mid + 1;
        else
            hi_s = mid;
    }
    uint sent_idx = lo_s;
    if (sent_idx >= num_sentences) return;

    uint sent_start = sent_offsets[sent_idx];
    uint sent_len   = sent_lengths[sent_idx];
    uint pos_in_sent = tid - sent_start;

    /* Verify we're actually in this sentence */
    if (pos_in_sent >= sent_len) return;

    uint max_j = min(pos_in_sent + window_size, sent_len - 1);

    for (uint j = pos_in_sent + 1; j <= max_j; j++)
    {
        uint word_j = flat_words[sent_start + j];
        if (word_i == word_j) continue;

        uint lo = min(word_i, word_j);
        uint hi = max(word_i, word_j);
        ulong key = ((ulong)lo << 32) | (ulong)hi;

        ulong ht_cap = PAIR_HT_CAPACITY;
        ulong mask = ht_cap - 1;
        ulong slot = ht_hash(key) & mask;
        uint pair_idx = HT_EMPTY_VALUE;

        for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
        {
            ulong prev = atom_cmpxchg(&pht_keys[slot], HT_EMPTY_KEY, key);

            if (prev == HT_EMPTY_KEY)
            {
                uint idx = atomic_add(pair_next_free, 1U);
                if (idx < PAIR_CAPACITY) {
                    pair_word_a[idx] = lo;
                    pair_word_b[idx] = hi;
                    pair_count[idx] = 0.0;
                    pair_mi[idx] = 0.0;
                    pair_flags[idx] = 0;
                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                    pht_values[slot] = idx;
                    pair_idx = idx;
                }
                break;
            }
            if (prev == key)
            {
                uint val = pht_values[slot];
                while (val == HT_EMPTY_VALUE) {
                    val = pht_values[slot];
                }
                pair_idx = val;
                break;
            }

            slot = (slot + 1) & mask;
        }

        if (pair_idx == HT_EMPTY_VALUE) continue;

        atomic_add_double(&pair_count[pair_idx], 1.0);
        atomic_add_double(&word_count[word_i], 1.0);
        atomic_add_double(&word_count[word_j], 1.0);
        pair_flags[pair_idx] = 1;
        atomic_add(total_pair_count, 1U);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  READBACK: Dump pair data for verification
 *
 *  Reads pair pool entries into flat output arrays.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void read_pairs(
    __global const uint*   pair_word_a,
    __global const uint*   pair_word_b,
    __global const double* pair_count,
    __global const double* pair_mi,
    __global const uint*   pair_flags,
    __global uint*         out_word_a,
    __global uint*         out_word_b,
    __global double*       out_count,
    __global double*       out_mi,
    __global uint*         out_flags,
    const uint             num_pairs)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    out_word_a[tid] = pair_word_a[tid];
    out_word_b[tid] = pair_word_b[tid];
    out_count[tid]  = pair_count[tid];
    out_mi[tid]     = pair_mi[tid];
    out_flags[tid]  = pair_flags[tid];
}
