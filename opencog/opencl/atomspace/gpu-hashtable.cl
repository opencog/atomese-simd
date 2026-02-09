/*
 * gpu-hashtable.cl -- Lock-free GPU hash table using linear probing
 *
 * Open-addressing hash table for mapping 64-bit keys to 32-bit values.
 * Designed for GPU-resident AtomSpace: maps atom content hashes to
 * pool indices (slots in flat SoA arrays).
 *
 * Based on: nosferalatu/SimpleGPUHashTable (CUDA) and
 *           LANL/CompactHash (OpenCL)
 *
 * Key properties:
 *   - Lock-free concurrent insert/lookup via atomic CAS
 *   - Linear probing with power-of-two table size
 *   - 64-bit keys (atom content hashes)
 *   - 32-bit values (pool indices)
 *   - Sentinel key 0xFFFFFFFFFFFFFFFF marks empty slots
 *   - Sentinel value 0xFFFFFFFF marks deleted entries
 *   - 50% load factor recommended (trade memory for speed)
 *
 * Requires: cl_khr_int64_base_atomics
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/* ─── Constants ─── */

/* Empty slot sentinel — no valid atom hash should equal this */
#define HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFUL
/* Deleted value sentinel — key stays, value marks as deleted */
#define HT_EMPTY_VALUE  0xFFFFFFFFU
/* Max probes before giving up (prevents infinite loops on full table) */
#define HT_MAX_PROBES   4096

/* ─── Hash function ─── */

/*
 * 64-bit finalizer from splitmix64 / Murmur3.
 * Produces well-distributed hashes from sequential or clustered keys.
 */
inline ulong ht_hash(ulong key)
{
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9UL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBUL;
    key ^= key >> 31;
    return key;
}

/* ─── Insert ─── */

/*
 * Insert key-value pairs into the hash table.
 *
 * Each work-item handles one (key, value) pair.
 * Uses atomic CAS on the key slot: if the slot is empty or already
 * contains our key, we write the value. Otherwise linear-probe forward.
 *
 * Args:
 *   table_keys   - key array   [capacity]  (ulong)
 *   table_values - value array  [capacity]  (uint)
 *   capacity     - table size (must be power of 2)
 *   in_keys      - keys to insert   [num_items]
 *   in_values    - values to insert  [num_items]
 *   num_items    - number of items to insert
 */
__kernel void ht_insert(
    __global volatile ulong* table_keys,
    __global uint*           table_values,
    const ulong              capacity,
    __global const ulong*    in_keys,
    __global const uint*     in_values,
    const uint               num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    ulong key   = in_keys[tid];
    uint  value = in_values[tid];

    /* Don't insert the sentinel key */
    if (key == HT_EMPTY_KEY) return;

    ulong mask = capacity - 1;  /* power-of-2 masking */
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        /* Try to claim this slot with our key */
        ulong prev = atom_cmpxchg(
            &table_keys[slot], HT_EMPTY_KEY, key);

        if (prev == HT_EMPTY_KEY || prev == key)
        {
            /* Slot was empty or already ours — write value */
            table_values[slot] = value;
            return;
        }

        /* Slot taken by another key — linear probe */
        slot = (slot + 1) & mask;
    }
    /* Table full or too many collisions — item not inserted */
}

/* ─── Lookup ─── */

/*
 * Look up keys in the hash table, returning their values.
 *
 * Each work-item looks up one key. If found, writes the value
 * to out_values[tid]. If not found, writes HT_EMPTY_VALUE.
 *
 * Args:
 *   table_keys   - key array   [capacity]  (ulong)
 *   table_values - value array  [capacity]  (uint)
 *   capacity     - table size (must be power of 2)
 *   query_keys   - keys to look up    [num_queries]
 *   out_values   - results written here [num_queries]
 *   num_queries  - number of lookups
 */
__kernel void ht_lookup(
    __global const ulong* table_keys,
    __global const uint*  table_values,
    const ulong           capacity,
    __global const ulong* query_keys,
    __global uint*        out_values,
    const uint            num_queries)
{
    uint tid = get_global_id(0);
    if (tid >= num_queries) return;

    ulong key = query_keys[tid];
    ulong mask = capacity - 1;
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong slot_key = table_keys[slot];

        if (slot_key == key)
        {
            /* Found — read value */
            out_values[tid] = table_values[slot];
            return;
        }
        if (slot_key == HT_EMPTY_KEY)
        {
            /* Empty slot — key not in table */
            out_values[tid] = HT_EMPTY_VALUE;
            return;
        }

        /* Different key — linear probe */
        slot = (slot + 1) & mask;
    }
    /* Max probes exhausted — not found */
    out_values[tid] = HT_EMPTY_VALUE;
}

/* ─── Delete ─── */

/*
 * Delete keys from the hash table.
 *
 * Tombstone deletion: key stays in table, value set to HT_EMPTY_VALUE.
 * Preserves probe chains for other keys. Deleted slots are NOT reused
 * for new inserts (keeps the algorithm simple).
 *
 * Args:
 *   table_keys   - key array   [capacity]  (ulong)
 *   table_values - value array  [capacity]  (uint)
 *   capacity     - table size (must be power of 2)
 *   del_keys     - keys to delete [num_deletes]
 *   num_deletes  - number of deletions
 */
__kernel void ht_delete(
    __global const ulong* table_keys,
    __global uint*        table_values,
    const ulong           capacity,
    __global const ulong* del_keys,
    const uint            num_deletes)
{
    uint tid = get_global_id(0);
    if (tid >= num_deletes) return;

    ulong key = del_keys[tid];
    ulong mask = capacity - 1;
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong slot_key = table_keys[slot];

        if (slot_key == key)
        {
            /* Found — tombstone the value */
            table_values[slot] = HT_EMPTY_VALUE;
            return;
        }
        if (slot_key == HT_EMPTY_KEY)
        {
            /* Not in table */
            return;
        }

        slot = (slot + 1) & mask;
    }
}

/* ─── Insert-or-increment ─── */

/*
 * Insert key with value, or atomically increment existing value.
 * This is the core operation for AtomSpace counting: if the pair
 * atom already exists, increment its count. Otherwise create it.
 *
 * IMPORTANT: table_values must be initialized to 0 (not 0xFF)
 * before using this kernel. Both the "new slot" and "existing slot"
 * paths use atomic_add(1), which avoids the race between CAS on
 * the key and write to the value.
 *
 * Args:
 *   table_keys   - key array   [capacity]  (ulong)
 *   table_values - value array  [capacity]  (uint, init to 0)
 *   capacity     - table size (must be power of 2)
 *   in_keys      - keys to insert/increment  [num_items]
 *   num_items    - number of items
 */
__kernel void ht_insert_or_increment(
    __global volatile ulong* table_keys,
    __global volatile uint*  table_values,
    const ulong              capacity,
    __global const ulong*    in_keys,
    const uint               num_items)
{
    uint tid = get_global_id(0);
    if (tid >= num_items) return;

    ulong key = in_keys[tid];
    if (key == HT_EMPTY_KEY) return;

    ulong mask = capacity - 1;
    ulong slot = ht_hash(key) & mask;

    for (uint probe = 0; probe < HT_MAX_PROBES; probe++)
    {
        ulong prev = atom_cmpxchg(
            &table_keys[slot], HT_EMPTY_KEY, key);

        if (prev == HT_EMPTY_KEY || prev == key)
        {
            /* Slot is ours (new or existing) — atomic increment.
             * Both paths do the same thing: no race between
             * key placement and value update. */
            atomic_add(&table_values[slot], 1U);
            return;
        }

        slot = (slot + 1) & mask;
    }
}

/* ─── Iterate (compact non-empty entries) ─── */

/*
 * Collect all non-empty, non-deleted entries into output arrays.
 * Uses atomic counter to pack results contiguously.
 *
 * Args:
 *   table_keys   - key array   [capacity]  (ulong)
 *   table_values - value array  [capacity]  (uint)
 *   capacity     - table size
 *   out_keys     - output keys   [max_output]
 *   out_values   - output values  [max_output]
 *   out_count    - atomic counter (single uint, initialized to 0)
 *   max_output   - max entries to output
 */
__kernel void ht_iterate(
    __global const ulong* table_keys,
    __global const uint*  table_values,
    const ulong           capacity,
    __global ulong*       out_keys,
    __global uint*        out_values,
    __global uint*        out_count,
    const uint            max_output)
{
    uint tid = get_global_id(0);
    if (tid >= capacity) return;

    ulong key = table_keys[tid];
    uint  val = table_values[tid];

    if (key != HT_EMPTY_KEY && val != HT_EMPTY_VALUE)
    {
        uint idx = atomic_add(out_count, 1U);
        if (idx < max_output)
        {
            out_keys[idx]   = key;
            out_values[idx] = val;
        }
    }
}
