/*
 * test-counting.c -- Test GPU sentence counting kernel
 *
 * Compile: gcc -O2 -o test-counting test-counting.c -lOpenCL -lm
 * Run:     ./test-counting
 *
 * Tests:
 *   1. Simple sentence pair counting (window=2, exact verification)
 *   2. Multi-sentence batch (verify no cross-boundary pairs)
 *   3. Read pairs kernel (readback verification)
 *   4. Binary search variant (count_sentence_pairs_large)
 *   5. Benchmark: 1000 sentences with window=6
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

/* ─── Pool capacities ─── */

#define WORD_CAPACITY        (128 * 1024)
#define PAIR_CAPACITY        (4 * 1024 * 1024)
#define SECTION_CAPACITY     (1024 * 1024)
#define WORD_HT_CAPACITY     (256 * 1024)
#define PAIR_HT_CAPACITY     (8 * 1024 * 1024)
#define SECTION_HT_CAPACITY  (2 * 1024 * 1024)

#define HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFULL
#define HT_EMPTY_VALUE  0xFFFFFFFFU

/* ─── Error checking ─── */

#define CL_CHECK(err, msg) do { \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", \
                (err), __FILE__, __LINE__, (msg)); \
        exit(1); \
    } \
} while(0)

/* ─── Read file ─── */

char* read_file(const char* path, size_t* len)
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(*len + 1);
    size_t n = fread(buf, 1, *len, f);
    buf[n] = '\0';
    *len = n;
    fclose(f);
    return buf;
}

/* ─── Timing ─── */

double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ─── Helper: reset pair pool and counters ─── */

void reset_pools(cl_command_queue queue,
                 cl_mem pht_keys, cl_mem pht_values,
                 cl_mem pair_count, cl_mem pair_mi, cl_mem pair_flags,
                 cl_mem word_count,
                 cl_mem pair_next_free, cl_mem total_pair_count)
{
    uint8_t pat_ff = 0xFF;
    uint8_t pat_00 = 0x00;
    uint32_t zero = 0;

    clEnqueueFillBuffer(queue, pht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pair_count, &pat_00, 1, 0,
        sizeof(double) * PAIR_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pair_mi, &pat_00, 1, 0,
        sizeof(double) * PAIR_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pair_flags, &pat_00, 1, 0,
        sizeof(uint32_t) * PAIR_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, word_count, &pat_00, 1, 0,
        sizeof(double) * WORD_CAPACITY, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, pair_next_free, CL_FALSE, 0,
        sizeof(uint32_t), &zero, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, total_pair_count, CL_FALSE, 0,
        sizeof(uint32_t), &zero, 0, NULL, NULL);
    clFinish(queue);
}

/* ─── Main ─── */

int main(int argc, char** argv)
{
    cl_int err;
    int pass_count = 0, fail_count = 0;

    printf("=== GPU Sentence Counting Test ===\n\n");

    /* ─── OpenCL setup ─── */

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CL_CHECK(err, "platform");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CL_CHECK(err, "device");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("GPU: %s\n", dev_name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "context");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    CL_CHECK(err, "queue");

    /* ─── Load and concatenate kernel sources ─── */

    size_t ht_len, as_len, ct_len;
    char* ht_src = read_file("gpu-hashtable.cl", &ht_len);
    char* as_src = read_file("gpu-atomspace.cl", &as_len);
    char* ct_src = read_file("gpu-counting.cl", &ct_len);

    size_t total_len = ht_len + 1 + as_len + 1 + ct_len;
    char* combined = malloc(total_len + 1);
    memcpy(combined, ht_src, ht_len);
    combined[ht_len] = '\n';
    memcpy(combined + ht_len + 1, as_src, as_len);
    combined[ht_len + 1 + as_len] = '\n';
    memcpy(combined + ht_len + 1 + as_len + 1, ct_src, ct_len);
    combined[total_len] = '\0';

    cl_program program = clCreateProgramWithSource(ctx, 1,
        (const char**)&combined, &total_len, &err);
    CL_CHECK(err, "create program");

    char build_opts[512];
    snprintf(build_opts, sizeof(build_opts),
        "-cl-std=CL1.2 "
        "-DWORD_CAPACITY=%d "
        "-DPAIR_CAPACITY=%d "
        "-DSECTION_CAPACITY=%d "
        "-DWORD_HT_CAPACITY=%d "
        "-DPAIR_HT_CAPACITY=%d "
        "-DSECTION_HT_CAPACITY=%d",
        WORD_CAPACITY, PAIR_CAPACITY, SECTION_CAPACITY,
        WORD_HT_CAPACITY, PAIR_HT_CAPACITY, SECTION_HT_CAPACITY);

    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        return 1;
    }
    printf("Kernels compiled successfully\n\n");

    /* ─── Create kernels ─── */

    cl_kernel k_count = clCreateKernel(program, "count_sentence_pairs", &err);
    CL_CHECK(err, "kernel count_sentence_pairs");
    cl_kernel k_count_large = clCreateKernel(program, "count_sentence_pairs_large", &err);
    CL_CHECK(err, "kernel count_sentence_pairs_large");
    cl_kernel k_read = clCreateKernel(program, "read_pairs", &err);
    CL_CHECK(err, "kernel read_pairs");

    size_t local_size = 256;

    /* ─── Allocate GPU buffers ─── */

    printf("Allocating GPU buffers...\n");
    uint32_t zero = 0;

    /* Pair hash table */
    cl_mem pht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * PAIR_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "pht_keys");
    cl_mem pht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "pht_values");

    /* Pair pool SoA */
    cl_mem pair_word_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_CAPACITY, NULL, &err);
    cl_mem pair_word_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_CAPACITY, NULL, &err);
    cl_mem pair_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * PAIR_CAPACITY, NULL, &err);
    cl_mem pair_mi = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * PAIR_CAPACITY, NULL, &err);
    cl_mem pair_flags = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_CAPACITY, NULL, &err);

    /* Pair bump allocator */
    cl_mem pair_next_free = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Word count array (marginals) — used directly by counting kernel */
    cl_mem word_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * WORD_CAPACITY, NULL, &err);

    /* Total pair count (global atomic counter) */
    cl_mem total_pair_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Initial reset */
    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    printf("GPU buffers ready\n\n");

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 1: Simple sentence pair counting
     *
     *  Sentence: word indices [0, 1, 2, 3] — 4 words
     *  Window = 2
     *
     *  Expected pairs (one thread per word position):
     *    pos 0: (0,1), (0,2)
     *    pos 1: (1,2), (1,3)
     *    pos 2: (2,3)
     *    pos 3: —
     *
     *  = 5 unique pairs, 5 count events
     *  Word marginals: word0=2, word1=3, word2=3, word3=2
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 1: Simple sentence (window=2) ---\n");

    uint32_t sent1_words[] = {0, 1, 2, 3};
    uint32_t sent1_offset = 0;
    uint32_t sent1_length = 4;
    cl_uint num_sentences = 1;
    cl_uint tw = 4;
    cl_uint window_size = 2;

    cl_mem d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * tw, sent1_words, &err);
    cl_mem d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &sent1_offset, &err);
    cl_mem d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &sent1_length, &err);

    /* Set all 16 kernel args for count_sentence_pairs */
    clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count, 3, sizeof(cl_uint), &num_sentences);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &tw);
    clSetKernelArg(k_count, 5, sizeof(cl_uint), &window_size);
    clSetKernelArg(k_count, 6, sizeof(cl_mem), &pht_keys);
    clSetKernelArg(k_count, 7, sizeof(cl_mem), &pht_values);
    clSetKernelArg(k_count, 8, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_count, 9, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_count, 10, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_count, 11, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_count, 12, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_count, 13, sizeof(cl_mem), &pair_next_free);
    clSetKernelArg(k_count, 14, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_count, 15, sizeof(cl_mem), &total_pair_count);

    double t0 = now_ms();
    size_t gs = ((tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue count");
    clFinish(queue);
    double t1 = now_ms();

    /* Read back results */
    uint32_t h_num_pairs;
    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs, 0, NULL, NULL);

    uint32_t h_total;
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total, 0, NULL, NULL);

    /* Read word marginals */
    double wc[4];
    clEnqueueReadBuffer(queue, word_count, CL_TRUE, 0,
        sizeof(double) * 4, wc, 0, NULL, NULL);

    printf("  Pairs created: %u (expected 5)\n", h_num_pairs);
    printf("  Total count events: %u (expected 5)\n", h_total);
    printf("  Word marginals: [0]=%.0f [1]=%.0f [2]=%.0f [3]=%.0f\n",
           wc[0], wc[1], wc[2], wc[3]);
    printf("    Expected:     [0]=2  [1]=3  [2]=3  [3]=2\n");
    printf("  Time: %.2f ms\n", t1 - t0);

    int t1_pass = (h_num_pairs == 5) && (h_total == 5) &&
                  (fabs(wc[0] - 2.0) < 0.5) && (fabs(wc[1] - 3.0) < 0.5) &&
                  (fabs(wc[2] - 3.0) < 0.5) && (fabs(wc[3] - 2.0) < 0.5);
    printf("  %s\n\n", t1_pass ? "PASS" : "FAIL");
    if (t1_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 2: Multi-sentence batch
     *
     *  Sentence 1: [0, 1, 2, 3]  — 4 words (same as test 1)
     *  Sentence 2: [4, 5, 6]     — 3 words (disjoint vocabulary)
     *
     *  flat_words  = [0, 1, 2, 3, 4, 5, 6]
     *  sent_offsets = [0, 4]
     *  sent_lengths = [4, 3]
     *
     *  Window = 2:
     *    Sentence 1: (0,1) (0,2) (1,2) (1,3) (2,3) = 5 pairs
     *    Sentence 2: (4,5) (4,6) (5,6)              = 3 pairs
     *    Total: 8 unique pairs, 8 count events
     *
     *  KEY: no cross-boundary pairs (e.g., no (3,4) pair)
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 2: Multi-sentence batch (window=2) ---\n");

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    uint32_t multi_words[] = {0, 1, 2, 3, 4, 5, 6};
    uint32_t multi_offsets[] = {0, 4};
    uint32_t multi_lengths[] = {4, 3};
    cl_uint multi_ns = 2;
    cl_uint multi_tw = 7;

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * multi_tw, multi_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * multi_ns, multi_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * multi_ns, multi_lengths, &err);

    /* Update sentence-specific args (6-15 stay same) */
    clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count, 3, sizeof(cl_uint), &multi_ns);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &multi_tw);
    clSetKernelArg(k_count, 5, sizeof(cl_uint), &window_size);

    t0 = now_ms();
    gs = ((multi_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue count multi");
    clFinish(queue);
    t1 = now_ms();

    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total, 0, NULL, NULL);

    printf("  Pairs created: %u (expected 8)\n", h_num_pairs);
    printf("  Total count events: %u (expected 8)\n", h_total);
    printf("  Time: %.2f ms\n", t1 - t0);

    /* Verify no cross-boundary pairs exist.
     * Words 0-3 belong to sentence 1, words 4-6 to sentence 2.
     * No pair should have one word from each sentence. */
    uint32_t* h_pa = malloc(sizeof(uint32_t) * h_num_pairs);
    uint32_t* h_pb = malloc(sizeof(uint32_t) * h_num_pairs);
    clEnqueueReadBuffer(queue, pair_word_a, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_pairs, h_pa, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, pair_word_b, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_pairs, h_pb, 0, NULL, NULL);

    int cross_boundary = 0;
    for (uint32_t i = 0; i < h_num_pairs; i++) {
        int a_sent = (h_pa[i] <= 3) ? 1 : 2;
        int b_sent = (h_pb[i] <= 3) ? 1 : 2;
        if (a_sent != b_sent) {
            cross_boundary++;
            printf("  CROSS-BOUNDARY: pair(%u, %u)\n", h_pa[i], h_pb[i]);
        }
    }
    printf("  Cross-boundary pairs: %d (expected 0)\n", cross_boundary);

    int t2_pass = (h_num_pairs == 8) && (h_total == 8) && (cross_boundary == 0);
    printf("  %s\n\n", t2_pass ? "PASS" : "FAIL");
    if (t2_pass) pass_count++; else fail_count++;

    free(h_pa);
    free(h_pb);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 3: Read pairs (readback kernel)
     *
     *  Uses the read_pairs kernel to copy pair data from the pool
     *  into output arrays. Verifies all pairs from Test 2 have
     *  count=1 and dirty flag=1.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 3: Read pairs (readback verification) ---\n");

    cl_mem d_out_wa = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * h_num_pairs, NULL, &err);
    cl_mem d_out_wb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * h_num_pairs, NULL, &err);
    cl_mem d_out_cnt = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * h_num_pairs, NULL, &err);
    cl_mem d_out_mi = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * h_num_pairs, NULL, &err);
    cl_mem d_out_flags = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * h_num_pairs, NULL, &err);

    cl_uint np = h_num_pairs;
    clSetKernelArg(k_read, 0, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_read, 1, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_read, 2, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_read, 3, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_read, 4, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_read, 5, sizeof(cl_mem), &d_out_wa);
    clSetKernelArg(k_read, 6, sizeof(cl_mem), &d_out_wb);
    clSetKernelArg(k_read, 7, sizeof(cl_mem), &d_out_cnt);
    clSetKernelArg(k_read, 8, sizeof(cl_mem), &d_out_mi);
    clSetKernelArg(k_read, 9, sizeof(cl_mem), &d_out_flags);
    clSetKernelArg(k_read, 10, sizeof(cl_uint), &np);

    gs = ((h_num_pairs + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_read, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue read_pairs");
    clFinish(queue);

    uint32_t* r_wa = malloc(sizeof(uint32_t) * h_num_pairs);
    uint32_t* r_wb = malloc(sizeof(uint32_t) * h_num_pairs);
    double*   r_cnt = malloc(sizeof(double) * h_num_pairs);
    uint32_t* r_flags = malloc(sizeof(uint32_t) * h_num_pairs);

    clEnqueueReadBuffer(queue, d_out_wa, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_pairs, r_wa, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_wb, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_pairs, r_wb, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_cnt, CL_TRUE, 0,
        sizeof(double) * h_num_pairs, r_cnt, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_flags, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_pairs, r_flags, 0, NULL, NULL);

    int all_counted = 1;
    int all_dirty = 1;
    int all_canonical = 1;
    double sum_counts = 0;

    printf("  Pairs:\n");
    for (uint32_t i = 0; i < h_num_pairs; i++) {
        printf("    [%u] (%u, %u) count=%.0f flags=%u\n",
               i, r_wa[i], r_wb[i], r_cnt[i], r_flags[i]);
        if (r_cnt[i] < 0.5) all_counted = 0;
        if (r_flags[i] != 1) all_dirty = 0;
        if (r_wa[i] > r_wb[i]) all_canonical = 0;
        sum_counts += r_cnt[i];
    }

    printf("  Sum of counts: %.0f (expected %u)\n", sum_counts, h_total);
    printf("  All counts > 0: %s\n", all_counted ? "YES" : "NO");
    printf("  All dirty flags: %s\n", all_dirty ? "YES" : "NO");
    printf("  All canonical (a <= b): %s\n", all_canonical ? "YES" : "NO");

    int t3_pass = all_counted && all_dirty && all_canonical &&
                  (fabs(sum_counts - h_total) < 0.5);
    printf("  %s\n\n", t3_pass ? "PASS" : "FAIL");
    if (t3_pass) pass_count++; else fail_count++;

    free(r_wa); free(r_wb); free(r_cnt); free(r_flags);
    clReleaseMemObject(d_out_wa);
    clReleaseMemObject(d_out_wb);
    clReleaseMemObject(d_out_cnt);
    clReleaseMemObject(d_out_mi);
    clReleaseMemObject(d_out_flags);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 4: Binary search variant
     *
     *  Same input as Test 2, using count_sentence_pairs_large.
     *  Should produce identical results.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 4: Binary search variant ---\n");

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    /* Set all 16 kernel args for count_sentence_pairs_large */
    clSetKernelArg(k_count_large, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count_large, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count_large, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count_large, 3, sizeof(cl_uint), &multi_ns);
    clSetKernelArg(k_count_large, 4, sizeof(cl_uint), &multi_tw);
    clSetKernelArg(k_count_large, 5, sizeof(cl_uint), &window_size);
    clSetKernelArg(k_count_large, 6, sizeof(cl_mem), &pht_keys);
    clSetKernelArg(k_count_large, 7, sizeof(cl_mem), &pht_values);
    clSetKernelArg(k_count_large, 8, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_count_large, 9, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_count_large, 10, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_count_large, 11, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_count_large, 12, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_count_large, 13, sizeof(cl_mem), &pair_next_free);
    clSetKernelArg(k_count_large, 14, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_count_large, 15, sizeof(cl_mem), &total_pair_count);

    t0 = now_ms();
    gs = ((multi_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_count_large, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue count_large");
    clFinish(queue);
    t1 = now_ms();

    uint32_t h_pairs_large, h_total_large;
    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_pairs_large, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total_large, 0, NULL, NULL);

    printf("  Pairs created: %u (expected 8)\n", h_pairs_large);
    printf("  Total count events: %u (expected 8)\n", h_total_large);
    printf("  Time: %.2f ms\n", t1 - t0);

    /* Verify word marginals match Test 2's expected values */
    double wc4[7];
    clEnqueueReadBuffer(queue, word_count, CL_TRUE, 0,
        sizeof(double) * 7, wc4, 0, NULL, NULL);
    printf("  Word marginals: [0]=%.0f [1]=%.0f [2]=%.0f [3]=%.0f [4]=%.0f [5]=%.0f [6]=%.0f\n",
           wc4[0], wc4[1], wc4[2], wc4[3], wc4[4], wc4[5], wc4[6]);
    printf("    Expected:     [0]=2  [1]=3  [2]=3  [3]=2  [4]=2  [5]=2  [6]=2\n");

    int t4_pass = (h_pairs_large == 8) && (h_total_large == 8) &&
                  (fabs(wc4[0] - 2.0) < 0.5) && (fabs(wc4[1] - 3.0) < 0.5) &&
                  (fabs(wc4[2] - 3.0) < 0.5) && (fabs(wc4[3] - 2.0) < 0.5) &&
                  (fabs(wc4[4] - 2.0) < 0.5) && (fabs(wc4[5] - 2.0) < 0.5) &&
                  (fabs(wc4[6] - 2.0) < 0.5);
    printf("  %s\n\n", t4_pass ? "PASS" : "FAIL");
    if (t4_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 5: Benchmark — 1000 sentences, window=6
     *
     *  500-word vocabulary, Zipf-like distribution, 5-20 words per
     *  sentence. Compares linear scan vs binary search variant.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 5: Benchmark (1000 sentences, window=6) ---\n");

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    /* Generate 1000 sentences with Zipf-like word distribution */
    int bench_ns = 1000;
    uint32_t* bench_offsets = malloc(sizeof(uint32_t) * bench_ns);
    uint32_t* bench_lengths = malloc(sizeof(uint32_t) * bench_ns);

    uint64_t rng = 0xCAFEBABEDEADBEEFULL;
    uint32_t bench_tw = 0;
    for (int s = 0; s < bench_ns; s++) {
        bench_offsets[s] = bench_tw;
        rng += 0x9E3779B97F4A7C15ULL;
        uint32_t slen = 5 + (uint32_t)((rng >> 32) % 16);
        bench_lengths[s] = slen;
        bench_tw += slen;
    }

    uint32_t* bench_words = malloc(sizeof(uint32_t) * bench_tw);
    rng = 0xFEEDFACE12345678ULL;
    for (uint32_t i = 0; i < bench_tw; i++) {
        rng += 0x9E3779B97F4A7C15ULL;
        double u = (double)(rng >> 32) / (double)0xFFFFFFFFU;
        bench_words[i] = (uint32_t)(u * u * 499.0);
    }

    printf("  Sentences: %d, Total words: %u, Avg len: %.1f\n",
           bench_ns, bench_tw, (float)bench_tw / bench_ns);

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_tw, bench_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, bench_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, bench_lengths, &err);

    cl_uint bench_window = 6;
    cl_uint bns = bench_ns;

    /* Run linear scan variant */
    clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count, 3, sizeof(cl_uint), &bns);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &bench_tw);
    clSetKernelArg(k_count, 5, sizeof(cl_uint), &bench_window);

    t0 = now_ms();
    gs = ((bench_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue bench count");
    clFinish(queue);
    t1 = now_ms();

    uint32_t bench_pairs, bench_total;
    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &bench_pairs, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &bench_total, 0, NULL, NULL);

    double linear_ms = t1 - t0;
    printf("  Linear scan: %.2f ms\n", linear_ms);
    printf("  Unique pairs: %u\n", bench_pairs);
    printf("  Total count events: %u\n", bench_total);
    printf("  Throughput: %.0f K sentences/sec, %.0f K pair-events/sec\n",
           bench_ns / (linear_ms / 1000.0) / 1000.0,
           bench_total / (linear_ms / 1000.0) / 1000.0);

    /* Run binary search variant with same data */
    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    clSetKernelArg(k_count_large, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count_large, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count_large, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count_large, 3, sizeof(cl_uint), &bns);
    clSetKernelArg(k_count_large, 4, sizeof(cl_uint), &bench_tw);
    clSetKernelArg(k_count_large, 5, sizeof(cl_uint), &bench_window);
    clSetKernelArg(k_count_large, 6, sizeof(cl_mem), &pht_keys);
    clSetKernelArg(k_count_large, 7, sizeof(cl_mem), &pht_values);
    clSetKernelArg(k_count_large, 8, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_count_large, 9, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_count_large, 10, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_count_large, 11, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_count_large, 12, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_count_large, 13, sizeof(cl_mem), &pair_next_free);
    clSetKernelArg(k_count_large, 14, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_count_large, 15, sizeof(cl_mem), &total_pair_count);

    t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_count_large, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue bench count_large");
    clFinish(queue);
    t1 = now_ms();

    uint32_t bench_pairs_l, bench_total_l;
    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &bench_pairs_l, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &bench_total_l, 0, NULL, NULL);

    double binary_ms = t1 - t0;
    printf("  Binary search: %.2f ms\n", binary_ms);
    printf("  Unique pairs: %u (expected %u)\n", bench_pairs_l, bench_pairs);
    printf("  Total count events: %u (expected %u)\n", bench_total_l, bench_total);
    printf("  Throughput: %.0f K sentences/sec, %.0f K pair-events/sec\n",
           bench_ns / (binary_ms / 1000.0) / 1000.0,
           bench_total_l / (binary_ms / 1000.0) / 1000.0);

    int t5_pass = (bench_pairs_l == bench_pairs) && (bench_total_l == bench_total);
    printf("  Linear vs Binary match: %s\n", t5_pass ? "PASS" : "FAIL");
    printf("  %s\n\n", t5_pass ? "PASS" : "FAIL");
    if (t5_pass) pass_count++; else fail_count++;

    /* ═══ Summary ═══ */

    printf("=== Results: %d PASS, %d FAIL ===\n", pass_count, fail_count);

    /* ─── Cleanup ─── */

    clReleaseMemObject(pht_keys);
    clReleaseMemObject(pht_values);
    clReleaseMemObject(pair_word_a);
    clReleaseMemObject(pair_word_b);
    clReleaseMemObject(pair_count);
    clReleaseMemObject(pair_mi);
    clReleaseMemObject(pair_flags);
    clReleaseMemObject(pair_next_free);
    clReleaseMemObject(word_count);
    clReleaseMemObject(total_pair_count);
    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);

    clReleaseKernel(k_count);
    clReleaseKernel(k_count_large);
    clReleaseKernel(k_read);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(bench_offsets);
    free(bench_lengths);
    free(bench_words);
    free(combined);
    free(ht_src);
    free(as_src);
    free(ct_src);

    return fail_count > 0 ? 1 : 0;
}
