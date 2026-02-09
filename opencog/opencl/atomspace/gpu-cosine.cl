/*
 * gpu-cosine.cl -- Cosine similarity and candidate generation on GPU
 *
 * Computes pairwise cosine similarity between words based on their
 * section (disjunct) vectors, entirely on GPU-resident data.
 *
 * Pipeline:
 *   1. compute_word_norms       — ||v||² for each word
 *   2. build_disjunct_chains    — reverse index: djh → linked list of sections
 *   3. accumulate_dot_products  — walk chains, accumulate dot(w1,w2)
 *   4. compute_cosines          — cosine = dot / (||v1|| × ||v2||)
 *   5. filter_candidates        — compact output above threshold
 *
 * No CPU↔GPU data transfer between stages.
 *
 * Requires capacity macros (set via -D flags):
 *   DJH_HT_CAPACITY        — disjunct reverse index hash table size (power of 2)
 *   CANDIDATE_CAPACITY      — max candidate pairs
 *   CANDIDATE_HT_CAPACITY   — candidate pair hash table size (power of 2)
 *
 * Appended after gpu-hashtable.cl, gpu-atomspace.cl at load time.
 * (gpu-sections.cl optional — only needed for full pipeline tests)
 */

/* ═══════════════════════════════════════════════════════════════
 *  STEP 1: COMPUTE WORD NORMS
 *
 *  One thread per section. Atomically adds count² to the word's
 *  norm accumulator. After all sections processed:
 *    word_norm_sq[w] = Σ_d count(w,d)²
 *
 *  where d ranges over all disjuncts for word w.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void compute_word_norms(
    __global const uint*    sec_word,
    __global const double*  sec_count,
    __global volatile double* word_norm_sq,
    const uint              num_sections)
{
    uint tid = get_global_id(0);
    if (tid >= num_sections) return;

    double count = sec_count[tid];
    if (count < 0.5) return;

    uint word = sec_word[tid];
    atomic_add_double(&word_norm_sq[word], count * count);
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 2: BUILD DISJUNCT REVERSE INDEX (linked list chains)
 *
 *  For each unique disjunct_hash, build a linked list of all
 *  sections that share that disjunct. Uses lock-free atomic
 *  prepend to a per-disjunct chain.
 *
 *  Data structures:
 *    djh_ht_keys[DJH_HT_CAPACITY]    — disjunct hash values
 *    djh_ht_values[DJH_HT_CAPACITY]  — chain head (section index)
 *    sec_chain_next[SECTION_CAPACITY] — per-section next pointer
 *
 *  Chain traversal: start at djh_ht_values[slot], follow
 *  sec_chain_next[] until HT_EMPTY_VALUE (end of chain).
 * ═══════════════════════════════════════════════════════════════ */

__kernel void build_disjunct_chains(
    __global const ulong*    sec_disjunct_hash,
    __global const double*   sec_count,
    __global volatile ulong* djh_ht_keys,
    __global volatile uint*  djh_ht_values,
    __global uint*           sec_chain_next,
    const uint               num_sections)
{
    uint tid = get_global_id(0);
    if (tid >= num_sections) return;

    /* Skip empty sections */
    if (sec_count[tid] < 0.5) {
        sec_chain_next[tid] = HT_EMPTY_VALUE;
        return;
    }

    ulong djh = sec_disjunct_hash[tid];
    if (djh == HT_EMPTY_KEY) {
        sec_chain_next[tid] = HT_EMPTY_VALUE;
        return;
    }

    ulong cap = DJH_HT_CAPACITY;
    ulong mask = cap - 1;
    ulong slot = ht_hash(djh) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(&djh_ht_keys[slot], HT_EMPTY_KEY, djh);

        if (prev == HT_EMPTY_KEY || prev == djh)
        {
            /* Atomically prepend this section to the chain.
             * atomic_xchg returns the old head (initially HT_EMPTY_VALUE
             * for a fresh slot, which serves as the end-of-chain sentinel).
             * Lock-free: concurrent prepends produce a valid chain. */
            uint old_head = atomic_xchg(&djh_ht_values[slot], tid);
            sec_chain_next[tid] = old_head;
            return;
        }

        slot = (slot + 1) & mask;
    }

    /* Hash table full — orphan this section */
    sec_chain_next[tid] = HT_EMPTY_VALUE;
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 3: ACCUMULATE DOT PRODUCTS
 *
 *  One thread per section. For each section (word_i, djh_i, count_i):
 *    - Look up djh_i in the disjunct reverse index
 *    - Pre-count chain length; skip if > MAX_CHAIN_LEN
 *      (very common disjuncts are uninformative, like stopwords)
 *    - Walk the chain of sections sharing this disjunct
 *    - For each other section (word_j, djh_i, count_j) where word_j != word_i:
 *      - Find-or-create candidate pair (word_i, word_j)
 *      - Atomically add count_i × count_j to the candidate's dot product
 *
 *  Uses canonical ordering (word_i < word_j) to avoid double-counting:
 *  each shared disjunct contributes exactly once to each pair's dot product.
 *
 *  Candidate pairs are stored in a hash table + pool (same pattern
 *  as atom pools from gpu-atomspace.cl).
 * ═══════════════════════════════════════════════════════════════ */

#ifndef MAX_CHAIN_LEN
#define MAX_CHAIN_LEN 200
#endif

__kernel void accumulate_dot_products(
    /* Section data */
    __global const uint*     sec_word,
    __global const ulong*    sec_disjunct_hash,
    __global const double*   sec_count,
    /* Disjunct reverse index */
    __global const ulong*    djh_ht_keys,
    __global const uint*     djh_ht_values,
    __global const uint*     sec_chain_next,
    /* Candidate hash table */
    __global volatile ulong* cand_ht_keys,
    __global volatile uint*  cand_ht_values,
    /* Candidate pool SoA */
    __global uint*            cand_word_a,
    __global uint*            cand_word_b,
    __global volatile double* cand_dot,
    __global volatile uint*   cand_next_free,
    /* Size */
    const uint               num_sections)
{
    uint tid = get_global_id(0);
    if (tid >= num_sections) return;

    uint my_word = sec_word[tid];
    double my_count = sec_count[tid];
    if (my_count < 0.5) return;

    ulong my_djh = sec_disjunct_hash[tid];
    if (my_djh == HT_EMPTY_KEY) return;

    /* ── Look up disjunct in reverse index ── */

    ulong cap = DJH_HT_CAPACITY;
    ulong mask = cap - 1;
    ulong slot = ht_hash(my_djh) & mask;
    uint chain_head = HT_EMPTY_VALUE;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong k = djh_ht_keys[slot];
        if (k == my_djh) {
            chain_head = djh_ht_values[slot];
            break;
        }
        if (k == HT_EMPTY_KEY) break;
        slot = (slot + 1) & mask;
    }

    if (chain_head == HT_EMPTY_VALUE) return;

    /* ── Pre-count chain length — skip overly common disjuncts ── */
    /* Disjuncts appearing in > MAX_CHAIN_LEN sections are too common
     * to be discriminative (like stopwords in information retrieval).
     * Skipping them entirely avoids O(K²) explosion for popular chains
     * and matches the CPU reverse-index filter. */

    uint chain_len = 0;
    uint pre = chain_head;
    while (pre != HT_EMPTY_VALUE && chain_len <= MAX_CHAIN_LEN)
    {
        chain_len++;
        pre = sec_chain_next[pre];
    }
    if (chain_len > MAX_CHAIN_LEN) return;

    /* ── Walk chain, accumulate dot products ── */

    uint cur = chain_head;

    while (cur != HT_EMPTY_VALUE)
    {
        uint other_word = sec_word[cur];
        double other_count = sec_count[cur];

        /* Only accumulate once per pair (canonical: my_word < other_word) */
        if (other_word != my_word && my_word < other_word && other_count >= 0.5)
        {
            /* ── Find or create candidate pair ── */

            uint lo = my_word;    /* already < other_word */
            uint hi = other_word;
            ulong cand_key = ((ulong)lo << 32) | (ulong)hi;

            ulong c_cap = CANDIDATE_HT_CAPACITY;
            ulong c_mask = c_cap - 1;
            ulong c_slot = ht_hash(cand_key) & c_mask;
            uint cand_idx = HT_EMPTY_VALUE;

            for (uint p = 0; p < HT_MAX_PROBES; p++)
            {
                ulong prev = atom_cmpxchg(
                    &cand_ht_keys[c_slot], HT_EMPTY_KEY, cand_key);

                if (prev == HT_EMPTY_KEY)
                {
                    /* New candidate — allocate from pool */
                    uint idx = atomic_add(cand_next_free, 1U);
                    if (idx < CANDIDATE_CAPACITY) {
                        cand_word_a[idx] = lo;
                        cand_word_b[idx] = hi;
                        cand_dot[idx] = 0.0;
                        mem_fence(CLK_GLOBAL_MEM_FENCE);
                        cand_ht_values[c_slot] = idx;
                        cand_idx = idx;
                    }
                    break;
                }
                if (prev == cand_key)
                {
                    /* Existing candidate — spin for value */
                    uint val = cand_ht_values[c_slot];
                    while (val == HT_EMPTY_VALUE) {
                        val = cand_ht_values[c_slot];
                    }
                    cand_idx = val;
                    break;
                }

                c_slot = (c_slot + 1) & c_mask;
            }

            if (cand_idx != HT_EMPTY_VALUE)
            {
                atomic_add_double(&cand_dot[cand_idx],
                                  my_count * other_count);
            }
        }

        cur = sec_chain_next[cur];
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 4: COMPUTE COSINES
 *
 *  One thread per candidate. Converts accumulated dot products
 *  to cosine similarity using pre-computed word norms.
 *
 *  cosine(w1, w2) = dot(w1, w2) / (||w1|| × ||w2||)
 * ═══════════════════════════════════════════════════════════════ */

/* MIN_NORM_SQ filters rare words: a word with N sections each count=1
 * has norm_sq = N. Setting MIN_NORM_SQ = 5.0 requires >= 5 section types.
 * This prevents hapax legomena from getting cosine=1.0 by chance. */
#ifndef MIN_NORM_SQ
#define MIN_NORM_SQ 50.0
#endif

__kernel void compute_cosines(
    __global const uint*   cand_word_a,
    __global const uint*   cand_word_b,
    __global const double* cand_dot,
    __global double*       cand_cosine,
    __global const double* word_norm_sq,
    const uint             num_candidates)
{
    uint tid = get_global_id(0);
    if (tid >= num_candidates) return;

    uint wa = cand_word_a[tid];
    uint wb = cand_word_b[tid];
    double dot = cand_dot[tid];

    double na = word_norm_sq[wa];
    double nb = word_norm_sq[wb];

    /* Skip rare words — not enough data for meaningful cosine */
    if (na < MIN_NORM_SQ || nb < MIN_NORM_SQ) {
        cand_cosine[tid] = 0.0;
        return;
    }

    double denom = sqrt(na) * sqrt(nb);
    double cos_val = (denom > 1e-10) ? (dot / denom) : 0.0;

    /* Clamp to valid range — prevents transient GPU state issues */
    if (cos_val > 1.0) cos_val = 1.0;
    if (cos_val < -1.0) cos_val = -1.0;

    cand_cosine[tid] = cos_val;
}

/* ═══════════════════════════════════════════════════════════════
 *  STEP 5: FILTER CANDIDATES
 *
 *  Compact candidates above a cosine threshold into contiguous
 *  output arrays. Returns (word_a, word_b, cosine) for each
 *  passing pair.
 *
 *  out_count must be initialized to 0 before kernel launch.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void filter_candidates(
    __global const uint*    cand_word_a,
    __global const uint*    cand_word_b,
    __global const double*  cand_cosine,
    const uint              num_candidates,
    const double            threshold,
    __global uint*          out_word_a,
    __global uint*          out_word_b,
    __global double*        out_cosine,
    __global volatile uint* out_count,
    const uint              max_output)
{
    uint tid = get_global_id(0);
    if (tid >= num_candidates) return;

    double cos_val = cand_cosine[tid];
    if (cos_val > threshold)
    {
        uint idx = atomic_add(out_count, 1U);
        if (idx < max_output) {
            out_word_a[idx]  = cand_word_a[tid];
            out_word_b[idx]  = cand_word_b[tid];
            out_cosine[idx]  = cos_val;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  READBACK: Dump candidate data for verification
 * ═══════════════════════════════════════════════════════════════ */

__kernel void read_candidates(
    __global const uint*   cand_word_a,
    __global const uint*   cand_word_b,
    __global const double* cand_dot,
    __global const double* cand_cosine,
    __global uint*         out_word_a,
    __global uint*         out_word_b,
    __global double*       out_dot,
    __global double*       out_cosine,
    const uint             num_candidates)
{
    uint tid = get_global_id(0);
    if (tid >= num_candidates) return;

    out_word_a[tid]  = cand_word_a[tid];
    out_word_b[tid]  = cand_word_b[tid];
    out_dot[tid]     = cand_dot[tid];
    out_cosine[tid]  = cand_cosine[tid];
}
