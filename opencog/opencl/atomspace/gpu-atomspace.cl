/*
 * gpu-atomspace.cl -- GPU-resident AtomSpace pools
 *
 * Three flat SoA (struct-of-arrays) pools for uniform-size atoms:
 *   - WordPool:    words/concepts (nodes)
 *   - PairPool:    word pairs (binary links)
 *   - SectionPool: word+disjunct (sections for connector vectors)
 *
 * Each pool has:
 *   - A bump allocator (atomic counter for next free slot)
 *   - A hash table mapping content hash → pool index
 *   - SoA arrays for each field (coalesced GPU access)
 *
 * All counting uses atomic operations (CAS-loop for doubles).
 *
 * This file is concatenated after gpu-hashtable.cl at load time,
 * so ht_hash(), atom_cmpxchg, HT_EMPTY_KEY etc. are available.
 */

/* ─── Double-precision atomic add via CAS loop ─── */

/*
 * OpenCL has no native atomicAdd for doubles. We implement it
 * via compare-and-swap on the 64-bit integer representation.
 * Uses atom_cmpxchg from cl_khr_int64_base_atomics.
 */
void atomic_add_double(__global volatile double* addr, double val)
{
    union { ulong i; double f; } next, expected, current;
    current.f = *addr;
    do {
        expected = current;
        next.f = expected.f + val;
        current.i = atom_cmpxchg(
            (__global volatile ulong*)addr,
            expected.i, next.i);
    } while (current.i != expected.i);
}

/* ─── Pool capacity constants (set by host at compile time) ─── */

/* These are defined as macros by the host via -D flags:
 *   WORD_CAPACITY     max words     (e.g., 131072 = 128K)
 *   PAIR_CAPACITY     max pairs     (e.g., 4194304 = 4M)
 *   SECTION_CAPACITY  max sections  (e.g., 1048576 = 1M)
 *   WORD_HT_CAPACITY  word hash table size (power of 2)
 *   PAIR_HT_CAPACITY  pair hash table size (power of 2)
 *   SECTION_HT_CAPACITY section hash table size (power of 2)
 */

/* ═══════════════════════════════════════════════════════════════
 *  WORD POOL — Nodes
 *
 *  SoA layout:
 *    word_name_hash[N]   ulong   — content hash (from string)
 *    word_count[N]       double  — marginal count
 *    word_mi_marginal[N] double  — sum of MI for marginal
 *    word_class_id[N]    uint    — class assignment (0 = none)
 *
 *  Hash table: name_hash → word_index
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Find or create words in the pool.
 *
 * For each input name_hash, looks up in the word hash table.
 * If found, returns the existing index. If not found, atomically
 * allocates a new slot and initializes it.
 *
 * Args:
 *   wht_keys, wht_values  - word hash table (keys + values)
 *   word_name_hash         - word pool: name hashes
 *   word_count             - word pool: counts (init to 0.0)
 *   word_class_id          - word pool: class IDs (init to 0)
 *   word_next_free         - atomic bump allocator (single uint)
 *   in_name_hashes         - input: name hashes to find/create
 *   out_indices            - output: pool index for each input
 *   num_items              - number of inputs
 */
__kernel void word_find_or_create(
    __global volatile ulong* wht_keys,
    __global volatile uint*  wht_values,
    __global ulong*          word_name_hash,
    __global double*         word_count,
    __global uint*           word_class_id,
    __global volatile uint*  word_next_free,
    __global const ulong*    in_name_hashes,
    __global uint*           out_indices,
    const uint               num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    ulong name_hash = in_name_hashes[tid];
    if (name_hash == HT_EMPTY_KEY) {
        out_indices[tid] = HT_EMPTY_VALUE;
        return;
    }

    ulong ht_cap = WORD_HT_CAPACITY;
    ulong mask = ht_cap - 1;
    ulong slot = ht_hash(name_hash) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(&wht_keys[slot], HT_EMPTY_KEY, name_hash);

        if (prev == HT_EMPTY_KEY)
        {
            /* We won the slot — allocate from pool and publish value */
            uint idx = atomic_add(word_next_free, 1U);
            if (idx < WORD_CAPACITY) {
                word_name_hash[idx] = name_hash;
                word_count[idx] = 0.0;
                word_class_id[idx] = 0;
                /* Write value LAST so spin-waiters see valid index */
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                wht_values[slot] = idx;
                out_indices[tid] = idx;
            } else {
                out_indices[tid] = HT_EMPTY_VALUE;
            }
            return;
        }
        if (prev == name_hash)
        {
            /* Already exists — spin until creator publishes value */
            uint val = wht_values[slot];
            while (val == HT_EMPTY_VALUE) {
                val = wht_values[slot];
            }
            out_indices[tid] = val;
            return;
        }

        slot = (slot + 1) & mask;
    }
    out_indices[tid] = HT_EMPTY_VALUE;
}

/* ═══════════════════════════════════════════════════════════════
 *  PAIR POOL — Binary Links (word pairs)
 *
 *  SoA layout:
 *    pair_word_a[N]  uint    — index into WordPool
 *    pair_word_b[N]  uint    — index into WordPool
 *    pair_count[N]   double  — joint count
 *    pair_mi[N]      double  — mutual information
 *    pair_flags[N]   uint    — dirty bit, etc.
 *
 *  Hash table key: (word_a << 32) | word_b  (packed 64-bit)
 * ═══════════════════════════════════════════════════════════════ */

/* Pack two 32-bit word indices into a single 64-bit hash key */
inline ulong pair_key(uint word_a, uint word_b)
{
    /* Always store with smaller index first for canonical ordering */
    uint lo = min(word_a, word_b);
    uint hi = max(word_a, word_b);
    return ((ulong)lo << 32) | (ulong)hi;
}

/*
 * Find or create word pairs in the pool.
 *
 * Args:
 *   pht_keys, pht_values  - pair hash table
 *   pair_word_a, pair_word_b - pair pool: word indices
 *   pair_count             - pair pool: counts (init to 0.0)
 *   pair_mi                - pair pool: MI values (init to 0.0)
 *   pair_flags             - pair pool: flags (init to 0)
 *   pair_next_free         - atomic bump allocator
 *   in_word_a, in_word_b   - input: word index pairs
 *   out_indices            - output: pair pool index for each input
 *   num_items              - number of inputs
 */
__kernel void pair_find_or_create(
    __global volatile ulong* pht_keys,
    __global volatile uint*  pht_values,
    __global uint*           pair_word_a,
    __global uint*           pair_word_b,
    __global double*         pair_count,
    __global double*         pair_mi,
    __global uint*           pair_flags,
    __global volatile uint*  pair_next_free,
    __global const uint*     in_word_a,
    __global const uint*     in_word_b,
    __global uint*           out_indices,
    const uint               num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    uint wa = in_word_a[tid];
    uint wb = in_word_b[tid];
    ulong key = pair_key(wa, wb);

    ulong ht_cap = PAIR_HT_CAPACITY;
    ulong mask = ht_cap - 1;
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(&pht_keys[slot], HT_EMPTY_KEY, key);

        if (prev == HT_EMPTY_KEY)
        {
            uint idx = atomic_add(pair_next_free, 1U);
            if (idx < PAIR_CAPACITY) {
                pair_word_a[idx] = min(wa, wb);
                pair_word_b[idx] = max(wa, wb);
                pair_count[idx] = 0.0;
                pair_mi[idx] = 0.0;
                pair_flags[idx] = 0;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                pht_values[slot] = idx;
                out_indices[tid] = idx;
            } else {
                out_indices[tid] = HT_EMPTY_VALUE;
            }
            return;
        }
        if (prev == key)
        {
            uint val = pht_values[slot];
            while (val == HT_EMPTY_VALUE) {
                val = pht_values[slot];
            }
            out_indices[tid] = val;
            return;
        }

        slot = (slot + 1) & mask;
    }
    out_indices[tid] = HT_EMPTY_VALUE;
}

/* ═══════════════════════════════════════════════════════════════
 *  SECTION POOL — Word + Disjunct
 *
 *  SoA layout:
 *    sec_word[N]          uint    — index into WordPool
 *    sec_disjunct_hash[N] ulong   — hash of full disjunct string
 *    sec_count[N]         double  — section count
 *
 *  Hash table key: hash(word_idx, disjunct_hash)
 * ═══════════════════════════════════════════════════════════════ */

/* Combine word index and disjunct hash into a single 64-bit key */
inline ulong section_key(uint word_idx, ulong disjunct_hash)
{
    /* Mix the word index into the disjunct hash */
    ulong key = disjunct_hash ^ ((ulong)word_idx * 0x9E3779B97F4A7C15UL);
    /* Ensure it's never the sentinel */
    if (key == HT_EMPTY_KEY) key = 0;
    return key;
}

/*
 * Find or create sections in the pool.
 *
 * Args:
 *   sht_keys, sht_values   - section hash table
 *   sec_word               - section pool: word indices
 *   sec_disjunct_hash      - section pool: disjunct hashes
 *   sec_count              - section pool: counts (init to 0.0)
 *   sec_next_free          - atomic bump allocator
 *   in_word_indices        - input: word indices
 *   in_disjunct_hashes     - input: disjunct hashes
 *   out_indices            - output: section pool index
 *   num_items              - number of inputs
 */
__kernel void section_find_or_create(
    __global volatile ulong* sht_keys,
    __global volatile uint*  sht_values,
    __global uint*           sec_word,
    __global ulong*          sec_disjunct_hash,
    __global double*         sec_count,
    __global volatile uint*  sec_next_free,
    __global const uint*     in_word_indices,
    __global const ulong*    in_disjunct_hashes,
    __global uint*           out_indices,
    const uint               num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    uint widx = in_word_indices[tid];
    ulong dhash = in_disjunct_hashes[tid];
    ulong key = section_key(widx, dhash);

    ulong ht_cap = SECTION_HT_CAPACITY;
    ulong mask = ht_cap - 1;
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(&sht_keys[slot], HT_EMPTY_KEY, key);

        if (prev == HT_EMPTY_KEY)
        {
            uint idx = atomic_add(sec_next_free, 1U);
            if (idx < SECTION_CAPACITY) {
                sec_word[idx] = widx;
                sec_disjunct_hash[idx] = dhash;
                sec_count[idx] = 0.0;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                sht_values[slot] = idx;
                out_indices[tid] = idx;
            } else {
                out_indices[tid] = HT_EMPTY_VALUE;
            }
            return;
        }
        if (prev == key)
        {
            uint val = sht_values[slot];
            while (val == HT_EMPTY_VALUE) {
                val = sht_values[slot];
            }
            out_indices[tid] = val;
            return;
        }

        slot = (slot + 1) & mask;
    }
    out_indices[tid] = HT_EMPTY_VALUE;
}

/* ═══════════════════════════════════════════════════════════════
 *  COUNTING OPERATIONS
 *
 *  Atomic double-precision increments on pool arrays.
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Increment pair counts by 1.0 for given pair indices.
 * Also increments word marginal counts for both words.
 */
__kernel void count_pairs(
    __global volatile double* pair_count,
    __global const uint*      pair_word_a,
    __global const uint*      pair_word_b,
    __global volatile double* word_count,
    __global volatile uint*   pair_flags,
    __global const uint*      pair_indices,
    const uint                num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    uint pidx = pair_indices[tid];
    if (pidx == HT_EMPTY_VALUE) return;

    /* Increment pair count */
    atomic_add_double(&pair_count[pidx], 1.0);

    /* Increment word marginals */
    uint wa = pair_word_a[pidx];
    uint wb = pair_word_b[pidx];
    atomic_add_double(&word_count[wa], 1.0);
    atomic_add_double(&word_count[wb], 1.0);

    /* Mark pair as dirty (needs MI recomputation) */
    pair_flags[pidx] = 1;
}

/*
 * Increment section counts by 1.0 for given section indices.
 */
__kernel void count_sections(
    __global volatile double* sec_count,
    __global const uint*      sec_indices,
    const uint                num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    uint sidx = sec_indices[tid];
    if (sidx == HT_EMPTY_VALUE) return;

    atomic_add_double(&sec_count[sidx], 1.0);
}

/* ═══════════════════════════════════════════════════════════════
 *  POOL STATISTICS
 *
 *  Read back pool sizes (bump allocator values).
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Read pool sizes into output array.
 * out_stats[0] = word count
 * out_stats[1] = pair count
 * out_stats[2] = section count
 */
__kernel void pool_stats(
    __global const uint* word_next_free,
    __global const uint* pair_next_free,
    __global const uint* sec_next_free,
    __global uint*       out_stats)
{
    if (get_global_id(0) != 0) return;
    out_stats[0] = *word_next_free;
    out_stats[1] = *pair_next_free;
    out_stats[2] = *sec_next_free;
}
