/*
 * gpu-sections.cl -- Section extraction from MST/PMFG edges on GPU
 *
 * Given MST edges for a batch of sentences (as position pairs within
 * each sentence), extract sections (word + disjunct) and increment
 * SectionPool counts. Disjunct hashing is done entirely on GPU —
 * no strings involved.
 *
 * A section = (word, disjunct), where the disjunct is the set of
 * connectors for that word in the parse. A connector = (partner_word,
 * direction), where direction is '+' (right) or '-' (left).
 *
 * On GPU: connector = (word_pool_idx, direction_bit).
 * Disjunct = FNV-1a hash of sorted connector sequence.
 *
 * Appended after gpu-hashtable.cl, gpu-atomspace.cl at load time.
 */

/* Maximum connectors per word position (degree limit).
 * PMFG on a 20-word sentence has at most 54 edges; per word,
 * planar graph average degree < 6. 32 is very safe. */
#define MAX_CONNECTORS 32

/* Connector direction bits */
#define DIR_LEFT  0U   /* partner is to the LEFT of this word */
#define DIR_RIGHT 1U   /* partner is to the RIGHT of this word */

/* ═══════════════════════════════════════════════════════════════
 *  FNV-1a HASH for disjunct encoding
 *
 *  Hash a sorted sequence of (word_pool_idx, direction) pairs
 *  into a single 64-bit disjunct hash.
 * ═══════════════════════════════════════════════════════════════ */

inline ulong fnv1a_init(void)
{
    return 0xcbf29ce484222325UL;
}

inline ulong fnv1a_mix(ulong hash, ulong val)
{
    hash ^= val;
    hash *= 0x100000001b3UL;
    return hash;
}

inline ulong hash_disjunct(uint* conn_words, uint* conn_dirs, uint count)
{
    ulong h = fnv1a_init();
    for (uint i = 0; i < count; i++) {
        /* Encode: (word_pool_idx << 1) | direction_bit */
        ulong encoded = ((ulong)conn_words[i] << 1) | (ulong)conn_dirs[i];
        h = fnv1a_mix(h, encoded);
    }
    /* Ensure it's never the hash table sentinel */
    if (h == HT_EMPTY_KEY) h = 0;
    return h;
}

/* ═══════════════════════════════════════════════════════════════
 *  INSERTION SORT for connectors
 *
 *  Sort by: direction first (LEFT=0 before RIGHT=1),
 *           then by word_pool_idx (ascending).
 *
 *  This gives a deterministic disjunct representation:
 *  all left connectors (sorted by partner) then all right
 *  connectors (sorted by partner).
 * ═══════════════════════════════════════════════════════════════ */

inline void sort_connectors(uint* conn_words, uint* conn_dirs, uint count)
{
    /* Insertion sort — count is small (typically < 10) */
    for (uint i = 1; i < count; i++) {
        uint w = conn_words[i];
        uint d = conn_dirs[i];
        uint j = i;
        while (j > 0) {
            /* Compare: direction first, then word index */
            int swap = 0;
            if (conn_dirs[j-1] > d)
                swap = 1;
            else if (conn_dirs[j-1] == d && conn_words[j-1] > w)
                swap = 1;

            if (!swap) break;

            conn_words[j] = conn_words[j-1];
            conn_dirs[j]  = conn_dirs[j-1];
            j--;
        }
        conn_words[j] = w;
        conn_dirs[j]  = d;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  EXTRACT SECTIONS FROM MST EDGES
 *
 *  One thread per word position across all sentences.
 *  Each thread:
 *    1. Finds its sentence (binary search)
 *    2. Scans MST edges for this sentence
 *    3. Collects connectors (partner word + direction)
 *    4. Sorts connectors deterministically
 *    5. Hashes sorted sequence → disjunct_hash
 *    6. Find-or-create section in SectionPool
 *    7. Atomically increments section count
 *
 *  Words with no edges (isolated) produce no section.
 *
 *  Args:
 *    -- Sentence data --
 *    flat_words        - word pool indices [total_words]
 *    sent_offsets      - start of each sentence in flat_words [num_sentences]
 *    sent_lengths      - length of each sentence [num_sentences]
 *    num_sentences     - number of sentences
 *    total_words       - total words across all sentences
 *
 *    -- MST edge data --
 *    edge_p1           - position 1 within sentence [total_edges]
 *    edge_p2           - position 2 within sentence [total_edges]
 *    edge_offsets      - start of each sentence's edges [num_sentences]
 *    edge_counts       - number of edges per sentence [num_sentences]
 *
 *    -- Section pool + hash table --
 *    sht_keys, sht_values  - section hash table
 *    sec_word              - section pool: word indices
 *    sec_disjunct_hash     - section pool: disjunct hashes
 *    sec_count             - section pool: counts
 *    sec_next_free         - section bump allocator
 *
 *    -- Stats --
 *    total_sections_created - atomic counter for new sections
 * ═══════════════════════════════════════════════════════════════ */

__kernel void extract_sections(
    /* sentence data */
    __global const uint*     flat_words,
    __global const uint*     sent_offsets,
    __global const uint*     sent_lengths,
    const uint               num_sentences,
    const uint               total_words,
    /* MST edge data */
    __global const uint*     edge_p1,
    __global const uint*     edge_p2,
    __global const uint*     edge_offsets,
    __global const uint*     edge_counts,
    /* section hash table */
    __global volatile ulong* sht_keys,
    __global volatile uint*  sht_values,
    /* section pool SoA */
    __global uint*           sec_word,
    __global ulong*          sec_disjunct_hash,
    __global volatile double* sec_count,
    __global volatile uint*  sec_next_free,
    /* stats */
    __global volatile uint*  total_sections_created)
{
    uint tid = get_global_id(0);
    if (tid >= total_words) return;

    /* ── Find which sentence this word belongs to (binary search) ── */

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

    /* Verify we're in this sentence */
    if (pos_in_sent >= sent_len) return;

    uint my_word = flat_words[tid];

    /* ── Collect connectors from MST edges ── */

    uint conn_words[MAX_CONNECTORS];
    uint conn_dirs[MAX_CONNECTORS];
    uint conn_count = 0;

    uint e_start = edge_offsets[sent_idx];
    uint e_count = edge_counts[sent_idx];

    for (uint e = 0; e < e_count && conn_count < MAX_CONNECTORS; e++)
    {
        uint p1 = edge_p1[e_start + e];
        uint p2 = edge_p2[e_start + e];

        if (p1 == pos_in_sent)
        {
            /* Edge goes from me to p2 → p2 is to the right */
            uint partner_word = flat_words[sent_start + p2];
            conn_words[conn_count] = partner_word;
            conn_dirs[conn_count]  = (p2 > pos_in_sent) ? DIR_RIGHT : DIR_LEFT;
            conn_count++;
        }
        else if (p2 == pos_in_sent)
        {
            /* Edge goes from p1 to me → p1 is to the left */
            uint partner_word = flat_words[sent_start + p1];
            conn_words[conn_count] = partner_word;
            conn_dirs[conn_count]  = (p1 < pos_in_sent) ? DIR_LEFT : DIR_RIGHT;
            conn_count++;
        }
    }

    /* No connectors = isolated word → no section */
    if (conn_count == 0) return;

    /* ── Sort connectors deterministically ── */

    sort_connectors(conn_words, conn_dirs, conn_count);

    /* ── Hash sorted connectors → disjunct_hash ── */

    ulong djh = hash_disjunct(conn_words, conn_dirs, conn_count);

    /* ── Find or create section in SectionPool ── */

    ulong key = section_key(my_word, djh);

    ulong ht_cap = SECTION_HT_CAPACITY;
    ulong mask = ht_cap - 1;
    ulong slot = ht_hash(key) & mask;
    uint sec_idx = HT_EMPTY_VALUE;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(&sht_keys[slot], HT_EMPTY_KEY, key);

        if (prev == HT_EMPTY_KEY)
        {
            /* New section — allocate from pool */
            uint idx = atomic_add(sec_next_free, 1U);
            if (idx < SECTION_CAPACITY) {
                sec_word[idx] = my_word;
                sec_disjunct_hash[idx] = djh;
                sec_count[idx] = 0.0;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                sht_values[slot] = idx;
                sec_idx = idx;
                atomic_add(total_sections_created, 1U);
            }
            break;
        }
        if (prev == key)
        {
            /* Existing section — spin for value */
            uint val = sht_values[slot];
            while (val == HT_EMPTY_VALUE) {
                val = sht_values[slot];
            }
            sec_idx = val;
            break;
        }

        slot = (slot + 1) & mask;
    }

    if (sec_idx == HT_EMPTY_VALUE) return;

    /* ── Increment section count ── */

    atomic_add_double(&sec_count[sec_idx], 1.0);
}

/* ═══════════════════════════════════════════════════════════════
 *  READBACK: Dump section data for verification
 *
 *  Reads section pool entries into flat output arrays.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void read_sections(
    __global const uint*   sec_word,
    __global const ulong*  sec_disjunct_hash,
    __global const double* sec_count,
    __global uint*         out_word,
    __global ulong*        out_disjunct_hash,
    __global double*       out_count,
    const uint             num_sections)
{
    uint tid = get_global_id(0);
    if (tid >= num_sections) return;

    out_word[tid]           = sec_word[tid];
    out_disjunct_hash[tid]  = sec_disjunct_hash[tid];
    out_count[tid]          = sec_count[tid];
}

/* ═══════════════════════════════════════════════════════════════
 *  EXTRACT SECTIONS + COUNT PAIRS (combined kernel)
 *
 *  Same as extract_sections but also counts section pairs
 *  within a window (for Level 1 vocabulary learning).
 *
 *  Each word position produces one section. Adjacent sections
 *  (within pair_window) form section pairs. Section pair counts
 *  are stored via the pair pool (reusing pair_find_or_create
 *  with section indices instead of word indices).
 *
 *  This is Phase 4's full pipeline: edges → sections → section pairs
 *  all in one kernel launch.
 * ═══════════════════════════════════════════════════════════════ */

/* Note: Section pair counting is deferred to Phase 5, where the
 * section pool is used for cosine similarity. The basic extract_sections
 * kernel above is sufficient for Phase 4. */
