/*
 * gpu-mi.cl -- MI computation on GPU-resident atom pools
 *
 * Computes mutual information directly from pair pool and word pool
 * buffers already resident in GPU memory. No CPU↔GPU data transfer
 * needed — the counting kernel (gpu-counting.cl) populates the same
 * buffers that this kernel reads.
 *
 * MI(x,y) = log2(count(x,y) * N / (count(x,*) * count(*,y)))
 *
 * Where:
 *   count(x,y) = pair_count[pair_idx]
 *   count(x,*) = word_count[pair_word_a[pair_idx]]  (left marginal)
 *   count(*,y) = word_count[pair_word_b[pair_idx]]  (right marginal)
 *   N          = total pair observations (passed as scalar arg)
 *
 * Appended after gpu-hashtable.cl, gpu-atomspace.cl, gpu-counting.cl
 * at load time.
 */

/* ═══════════════════════════════════════════════════════════════
 *  COMPUTE MI FOR ALL PAIRS
 *
 *  One thread per pair in the pool. Reads pair count and word
 *  marginals directly from GPU-resident SoA arrays.
 *
 *  Pairs with count < 1 get MI = 0.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void compute_mi_resident(
    __global const double* pair_count,
    __global const uint*   pair_word_a,
    __global const uint*   pair_word_b,
    __global double*       pair_mi,
    __global const double* word_count,
    const double           n_total,
    const uint             num_pairs)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    double count = pair_count[tid];
    if (count < 1.0) {
        pair_mi[tid] = 0.0;
        return;
    }

    uint wa = pair_word_a[tid];
    uint wb = pair_word_b[tid];
    double left  = word_count[wa];
    double right = word_count[wb];

    /* MI = log2(count * N / (left * right))
     *    = (ln(count) + ln(N) - ln(left) - ln(right)) / ln(2) */
    double eps = 1e-10;
    left  = fmax(left, eps);
    right = fmax(right, eps);
    double n = fmax(n_total, eps);

    double log2_factor = 1.4426950408889634;
    double mi = (log(count) + log(n) - log(left) - log(right)) * log2_factor;

    pair_mi[tid] = mi;
}

/* ═══════════════════════════════════════════════════════════════
 *  COMPUTE MI FOR DIRTY PAIRS ONLY
 *
 *  Same as above but only processes pairs where flags == 1.
 *  Clears the dirty flag after computation.
 *
 *  Use this for incremental MI updates after new sentences
 *  are counted — avoids recomputing all 2M+ pairs.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void compute_mi_dirty(
    __global const double*  pair_count,
    __global const uint*    pair_word_a,
    __global const uint*    pair_word_b,
    __global double*        pair_mi,
    __global volatile uint* pair_flags,
    __global const double*  word_count,
    const double            n_total,
    const uint              num_pairs)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    if (pair_flags[tid] != 1) return;

    double count = pair_count[tid];
    if (count < 1.0) {
        pair_mi[tid] = 0.0;
        pair_flags[tid] = 0;
        return;
    }

    uint wa = pair_word_a[tid];
    uint wb = pair_word_b[tid];
    double left  = word_count[wa];
    double right = word_count[wb];

    double eps = 1e-10;
    left  = fmax(left, eps);
    right = fmax(right, eps);
    double n = fmax(n_total, eps);

    double log2_factor = 1.4426950408889634;
    double mi = (log(count) + log(n) - log(left) - log(right)) * log2_factor;

    pair_mi[tid] = mi;
    pair_flags[tid] = 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  MI STATISTICS
 *
 *  Count pairs by MI value. Useful for diagnostics and
 *  determining warm-start thresholds.
 *
 *  Outputs (all single-uint atomic counters, init to 0):
 *    count_nonzero     — pairs with count > 0
 *    count_positive_mi — pairs with MI > 0
 *    count_above_thresh — pairs with MI > threshold
 * ═══════════════════════════════════════════════════════════════ */

__kernel void mi_stats(
    __global const double* pair_mi,
    __global const double* pair_count,
    const uint             num_pairs,
    const double           threshold,
    __global volatile uint* count_nonzero,
    __global volatile uint* count_positive_mi,
    __global volatile uint* count_above_thresh)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    double cnt = pair_count[tid];
    if (cnt < 0.5) return;

    atomic_add(count_nonzero, 1U);

    double mi = pair_mi[tid];
    if (mi > 0.0)
        atomic_add(count_positive_mi, 1U);
    if (mi > threshold)
        atomic_add(count_above_thresh, 1U);
}

/* ═══════════════════════════════════════════════════════════════
 *  MI THRESHOLD FILTER
 *
 *  Compact pairs above a MI threshold into contiguous output
 *  arrays. Returns (pair_index, mi_value) for each passing pair.
 *
 *  Use cases:
 *    - Warm-start bootstrap (only load high-MI pairs)
 *    - Building neighbor index for cosine similarity
 *    - Exporting significant pairs for visualization
 *
 *  out_count must be initialized to 0 before kernel launch.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void mi_filter(
    __global const double* pair_mi,
    __global const double* pair_count,
    const uint             num_pairs,
    const double           mi_threshold,
    __global uint*         out_pair_indices,
    __global double*       out_mi_values,
    __global volatile uint* out_count,
    const uint             max_output)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    double cnt = pair_count[tid];
    double mi  = pair_mi[tid];

    if (cnt > 0.5 && mi > mi_threshold) {
        uint idx = atomic_add(out_count, 1U);
        if (idx < max_output) {
            out_pair_indices[idx] = tid;
            out_mi_values[idx] = mi;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  READBACK: Dump pair data with MI for verification
 *
 *  Extended version of read_pairs from gpu-counting.cl that
 *  also includes word indices and MI values.
 * ═══════════════════════════════════════════════════════════════ */

__kernel void read_pairs_with_mi(
    __global const uint*   pair_word_a,
    __global const uint*   pair_word_b,
    __global const double* pair_count,
    __global const double* pair_mi,
    __global const uint*   pair_flags,
    __global const double* word_count,
    __global uint*         out_word_a,
    __global uint*         out_word_b,
    __global double*       out_count,
    __global double*       out_mi,
    __global double*       out_left_marginal,
    __global double*       out_right_marginal,
    const uint             num_pairs)
{
    uint tid = get_global_id(0);
    if (tid >= num_pairs) return;

    uint wa = pair_word_a[tid];
    uint wb = pair_word_b[tid];

    out_word_a[tid]         = wa;
    out_word_b[tid]         = wb;
    out_count[tid]          = pair_count[tid];
    out_mi[tid]             = pair_mi[tid];
    out_left_marginal[tid]  = word_count[wa];
    out_right_marginal[tid] = word_count[wb];
}
