/*
 * test-mi.c -- Test GPU-resident MI computation
 *
 * Compile: gcc -O2 -o test-mi test-mi.c -lOpenCL -lm
 * Run:     ./test-mi
 *
 * Tests the full pipeline: count sentences → compute MI → verify.
 * All data stays on GPU — no CPU↔GPU marshaling for MI.
 *
 * Tests:
 *   1. Manual MI verification (known counts → expected MI)
 *   2. Pipeline: count → MI (sentences → pairs → MI in one flow)
 *   3. Dirty-only MI (incremental recompute)
 *   4. MI statistics (count positive/above-threshold)
 *   5. MI filter (compact high-MI pairs)
 *   6. Benchmark: 1000 sentences → MI on all pairs
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

/* ─── CPU MI for verification ─── */

double cpu_mi(double count, double left_marg, double right_marg, double n)
{
    if (count < 1.0 || left_marg < 1e-10 || right_marg < 1e-10 || n < 1e-10)
        return 0.0;
    return log2(count * n / (left_marg * right_marg));
}

/* ─── Reset pools ─── */

void reset_pools(cl_command_queue queue,
                 cl_mem pht_keys, cl_mem pht_values,
                 cl_mem pair_count, cl_mem pair_mi, cl_mem pair_flags,
                 cl_mem word_count,
                 cl_mem pair_next_free, cl_mem total_pair_count)
{
    uint8_t pat_ff = 0xFF, pat_00 = 0x00;
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

    printf("=== GPU MI Computation Test ===\n\n");

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

    /* ─── Load and concatenate all 4 kernel sources ─── */

    size_t ht_len, as_len, ct_len, mi_len;
    char* ht_src = read_file("gpu-hashtable.cl", &ht_len);
    char* as_src = read_file("gpu-atomspace.cl", &as_len);
    char* ct_src = read_file("gpu-counting.cl", &ct_len);
    char* mi_src = read_file("gpu-mi.cl", &mi_len);

    size_t total_len = ht_len + 1 + as_len + 1 + ct_len + 1 + mi_len;
    char* combined = malloc(total_len + 1);
    size_t off = 0;
    memcpy(combined + off, ht_src, ht_len); off += ht_len;
    combined[off++] = '\n';
    memcpy(combined + off, as_src, as_len); off += as_len;
    combined[off++] = '\n';
    memcpy(combined + off, ct_src, ct_len); off += ct_len;
    combined[off++] = '\n';
    memcpy(combined + off, mi_src, mi_len); off += mi_len;
    combined[off] = '\0';

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
    printf("All kernels compiled successfully\n\n");

    /* ─── Create kernels ─── */

    cl_kernel k_count = clCreateKernel(program, "count_sentence_pairs", &err);
    CL_CHECK(err, "kernel count_sentence_pairs");
    cl_kernel k_mi_all = clCreateKernel(program, "compute_mi_resident", &err);
    CL_CHECK(err, "kernel compute_mi_resident");
    cl_kernel k_mi_dirty = clCreateKernel(program, "compute_mi_dirty", &err);
    CL_CHECK(err, "kernel compute_mi_dirty");
    cl_kernel k_mi_stats = clCreateKernel(program, "mi_stats", &err);
    CL_CHECK(err, "kernel mi_stats");
    cl_kernel k_mi_filter = clCreateKernel(program, "mi_filter", &err);
    CL_CHECK(err, "kernel mi_filter");
    cl_kernel k_read_mi = clCreateKernel(program, "read_pairs_with_mi", &err);
    CL_CHECK(err, "kernel read_pairs_with_mi");

    size_t local_size = 256;

    /* ─── Allocate GPU buffers ─── */

    uint32_t zero = 0;

    cl_mem pht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * PAIR_HT_CAPACITY, NULL, &err);
    cl_mem pht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_HT_CAPACITY, NULL, &err);
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
    cl_mem pair_next_free = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem word_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * WORD_CAPACITY, NULL, &err);
    cl_mem total_pair_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    printf("GPU buffers ready\n\n");

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 1: Manual MI verification
     *
     *  Manually set pair counts and word marginals, then verify
     *  that compute_mi_resident produces correct MI values.
     *
     *  Setup (3 pairs, 4 words):
     *    pair 0: (word0, word1) count=10  → MI = log2(10*100/(30*40))
     *    pair 1: (word0, word2) count=5   → MI = log2(5*100/(30*20))
     *    pair 2: (word1, word3) count=20  → MI = log2(20*100/(40*50))
     *
     *  Word marginals: word0=30, word1=40, word2=20, word3=50
     *  N = 100 (total pair observations)
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 1: Manual MI verification ---\n");

    /* Write pair data directly to GPU */
    uint32_t h_pa[] = {0, 0, 1};
    uint32_t h_pb[] = {1, 2, 3};
    double   h_pc[] = {10.0, 5.0, 20.0};
    double   h_wc[] = {30.0, 40.0, 20.0, 50.0};
    uint32_t h_np = 3;

    clEnqueueWriteBuffer(queue, pair_word_a, CL_FALSE, 0,
        sizeof(uint32_t) * 3, h_pa, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, pair_word_b, CL_FALSE, 0,
        sizeof(uint32_t) * 3, h_pb, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, pair_count, CL_FALSE, 0,
        sizeof(double) * 3, h_pc, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, word_count, CL_FALSE, 0,
        sizeof(double) * 4, h_wc, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, pair_next_free, CL_FALSE, 0,
        sizeof(uint32_t), &h_np, 0, NULL, NULL);
    clFinish(queue);

    /* Compute expected MI on CPU */
    double n_total = 100.0;
    double expected_mi[3];
    expected_mi[0] = cpu_mi(10.0, 30.0, 40.0, 100.0);  /* log2(10*100/1200) */
    expected_mi[1] = cpu_mi(5.0,  30.0, 20.0, 100.0);   /* log2(5*100/600) */
    expected_mi[2] = cpu_mi(20.0, 40.0, 50.0, 100.0);  /* log2(20*100/2000) */

    printf("  Expected MI: [%.4f, %.4f, %.4f]\n",
           expected_mi[0], expected_mi[1], expected_mi[2]);

    /* Run MI kernel */
    cl_uint np = 3;
    clSetKernelArg(k_mi_all, 0, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_mi_all, 1, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_mi_all, 2, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_mi_all, 3, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_mi_all, 4, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_mi_all, 5, sizeof(cl_double), &n_total);
    clSetKernelArg(k_mi_all, 6, sizeof(cl_uint), &np);

    size_t gs = ((np + local_size - 1) / local_size) * local_size;
    double t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_mi_all, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue mi_all");
    clFinish(queue);
    double t1 = now_ms();

    /* Read back MI values */
    double gpu_mi[3];
    clEnqueueReadBuffer(queue, pair_mi, CL_TRUE, 0,
        sizeof(double) * 3, gpu_mi, 0, NULL, NULL);

    printf("  GPU MI:      [%.4f, %.4f, %.4f]\n",
           gpu_mi[0], gpu_mi[1], gpu_mi[2]);
    printf("  Time: %.2f ms\n", t1 - t0);

    int t1_pass = 1;
    for (int i = 0; i < 3; i++) {
        double diff = fabs(gpu_mi[i] - expected_mi[i]);
        if (diff > 0.001) {
            printf("  MISMATCH pair %d: gpu=%.6f expected=%.6f diff=%.6f\n",
                   i, gpu_mi[i], expected_mi[i], diff);
            t1_pass = 0;
        }
    }
    printf("  %s\n\n", t1_pass ? "PASS" : "FAIL");
    if (t1_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 2: Full pipeline — count sentences → compute MI
     *
     *  Run count_sentence_pairs on test sentences, then immediately
     *  run compute_mi_resident on the same GPU-resident data.
     *  All data stays on GPU — zero transfers between stages.
     *
     *  Sentence 1: [0, 1, 2, 3]  (4 words, window=2)
     *  Sentence 2: [4, 5, 6]     (3 words)
     *
     *  Expected: 8 pairs from counting, all get valid MI > 0.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 2: Full pipeline (count → MI) ---\n");

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    uint32_t sent_words[] = {0, 1, 2, 3, 4, 5, 6};
    uint32_t sent_offsets[] = {0, 4};
    uint32_t sent_lengths[] = {4, 3};
    cl_uint num_sentences = 2;
    cl_uint tw = 7;
    cl_uint window_size = 2;

    cl_mem d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * tw, sent_words, &err);
    cl_mem d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * num_sentences, sent_offsets, &err);
    cl_mem d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * num_sentences, sent_lengths, &err);

    /* Stage 1: Count pairs */
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

    t0 = now_ms();
    gs = ((tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue count");
    clFinish(queue);
    double t_count = now_ms() - t0;

    /* Read num_pairs and total for MI computation */
    uint32_t h_num_pairs, h_total;
    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total, 0, NULL, NULL);

    printf("  Stage 1 (count): %u pairs, %u events in %.2f ms\n",
           h_num_pairs, h_total, t_count);

    /* Stage 2: Compute MI — data stays on GPU! */
    double n = (double)h_total;
    clSetKernelArg(k_mi_all, 0, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_mi_all, 1, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_mi_all, 2, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_mi_all, 3, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_mi_all, 4, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_mi_all, 5, sizeof(cl_double), &n);
    clSetKernelArg(k_mi_all, 6, sizeof(cl_uint), &h_num_pairs);

    t0 = now_ms();
    gs = ((h_num_pairs + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_mi_all, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue mi_all pipeline");
    clFinish(queue);
    double t_mi = now_ms() - t0;

    printf("  Stage 2 (MI):    %u pairs in %.2f ms\n", h_num_pairs, t_mi);
    printf("  Total pipeline:  %.2f ms (zero transfers between stages)\n",
           t_count + t_mi);

    /* Read back and verify with read_pairs_with_mi */
    cl_mem d_out_wa = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * h_num_pairs, NULL, &err);
    cl_mem d_out_wb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * h_num_pairs, NULL, &err);
    cl_mem d_out_cnt = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * h_num_pairs, NULL, &err);
    cl_mem d_out_mi = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * h_num_pairs, NULL, &err);
    cl_mem d_out_lm = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * h_num_pairs, NULL, &err);
    cl_mem d_out_rm = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * h_num_pairs, NULL, &err);

    clSetKernelArg(k_read_mi, 0, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_read_mi, 1, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_read_mi, 2, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_read_mi, 3, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_read_mi, 4, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_read_mi, 5, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_read_mi, 6, sizeof(cl_mem), &d_out_wa);
    clSetKernelArg(k_read_mi, 7, sizeof(cl_mem), &d_out_wb);
    clSetKernelArg(k_read_mi, 8, sizeof(cl_mem), &d_out_cnt);
    clSetKernelArg(k_read_mi, 9, sizeof(cl_mem), &d_out_mi);
    clSetKernelArg(k_read_mi, 10, sizeof(cl_mem), &d_out_lm);
    clSetKernelArg(k_read_mi, 11, sizeof(cl_mem), &d_out_rm);
    clSetKernelArg(k_read_mi, 12, sizeof(cl_uint), &h_num_pairs);

    gs = ((h_num_pairs + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_read_mi, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);

    uint32_t* r_wa = malloc(sizeof(uint32_t) * h_num_pairs);
    uint32_t* r_wb = malloc(sizeof(uint32_t) * h_num_pairs);
    double*   r_cnt = malloc(sizeof(double) * h_num_pairs);
    double*   r_mi = malloc(sizeof(double) * h_num_pairs);
    double*   r_lm = malloc(sizeof(double) * h_num_pairs);
    double*   r_rm = malloc(sizeof(double) * h_num_pairs);

    clEnqueueReadBuffer(queue, d_out_wa, CL_TRUE, 0, sizeof(uint32_t)*h_num_pairs, r_wa, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_wb, CL_TRUE, 0, sizeof(uint32_t)*h_num_pairs, r_wb, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_cnt, CL_TRUE, 0, sizeof(double)*h_num_pairs, r_cnt, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_mi, CL_TRUE, 0, sizeof(double)*h_num_pairs, r_mi, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_lm, CL_TRUE, 0, sizeof(double)*h_num_pairs, r_lm, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_rm, CL_TRUE, 0, sizeof(double)*h_num_pairs, r_rm, 0, NULL, NULL);

    int t2_pass = 1;
    int all_mi_valid = 1;
    printf("  Pair details:\n");
    for (uint32_t i = 0; i < h_num_pairs; i++) {
        double exp = cpu_mi(r_cnt[i], r_lm[i], r_rm[i], n);
        double diff = fabs(r_mi[i] - exp);
        printf("    [%u] (%u,%u) cnt=%.0f lm=%.0f rm=%.0f MI=%.4f (exp=%.4f) %s\n",
               i, r_wa[i], r_wb[i], r_cnt[i], r_lm[i], r_rm[i],
               r_mi[i], exp, (diff < 0.001) ? "OK" : "MISMATCH");
        if (diff > 0.001) { t2_pass = 0; all_mi_valid = 0; }
    }
    printf("  All MI values match CPU: %s\n", all_mi_valid ? "YES" : "NO");
    t2_pass = t2_pass && (h_num_pairs == 8);
    printf("  %s\n\n", t2_pass ? "PASS" : "FAIL");
    if (t2_pass) pass_count++; else fail_count++;

    free(r_wa); free(r_wb); free(r_cnt); free(r_mi); free(r_lm); free(r_rm);
    clReleaseMemObject(d_out_wa); clReleaseMemObject(d_out_wb);
    clReleaseMemObject(d_out_cnt); clReleaseMemObject(d_out_mi);
    clReleaseMemObject(d_out_lm); clReleaseMemObject(d_out_rm);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 3: Dirty-only MI recompute
     *
     *  After counting, all pairs have flags=1 (dirty).
     *  Run compute_mi_dirty — should compute MI and clear flags.
     *  Then add more sentences, creating new dirty pairs.
     *  Run compute_mi_dirty again — should only recompute dirty ones.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 3: Dirty-only MI recompute ---\n");

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    /* Count first batch: [0, 1, 2] window=2 → pairs (0,1),(0,2),(1,2) */
    uint32_t batch1_words[] = {0, 1, 2};
    uint32_t batch1_offset = 0;
    uint32_t batch1_length = 3;
    cl_uint b1_ns = 1, b1_tw = 3, b1_ws = 2;

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 3, batch1_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &batch1_offset, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &batch1_length, &err);

    clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count, 3, sizeof(cl_uint), &b1_ns);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &b1_tw);
    clSetKernelArg(k_count, 5, sizeof(cl_uint), &b1_ws);

    clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &local_size, &local_size, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total, 0, NULL, NULL);

    printf("  Batch 1: %u pairs, %u events\n", h_num_pairs, h_total);

    /* Run dirty MI — should process all 3 pairs and clear flags */
    n = (double)h_total;
    clSetKernelArg(k_mi_dirty, 0, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_mi_dirty, 1, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_mi_dirty, 2, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_mi_dirty, 3, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_mi_dirty, 4, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_mi_dirty, 5, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_mi_dirty, 6, sizeof(cl_double), &n);
    clSetKernelArg(k_mi_dirty, 7, sizeof(cl_uint), &h_num_pairs);

    gs = ((h_num_pairs + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_mi_dirty, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);

    /* Verify flags are cleared */
    uint32_t flags_after[3];
    clEnqueueReadBuffer(queue, pair_flags, CL_TRUE, 0,
        sizeof(uint32_t) * 3, flags_after, 0, NULL, NULL);

    int all_clear = (flags_after[0] == 0 && flags_after[1] == 0 && flags_after[2] == 0);
    printf("  After dirty MI: flags=[%u,%u,%u] (all 0?) %s\n",
           flags_after[0], flags_after[1], flags_after[2],
           all_clear ? "YES" : "NO");

    /* Read MI values after first batch */
    double mi_batch1[3];
    clEnqueueReadBuffer(queue, pair_mi, CL_TRUE, 0,
        sizeof(double) * 3, mi_batch1, 0, NULL, NULL);
    printf("  MI after batch 1: [%.4f, %.4f, %.4f]\n",
           mi_batch1[0], mi_batch1[1], mi_batch1[2]);

    /* Count second batch: [1, 2, 3] window=2 → new pair (1,3),(2,3) + existing (1,2) */
    uint32_t batch2_words[] = {1, 2, 3};
    uint32_t batch2_offset = 0;
    uint32_t batch2_length = 3;

    clReleaseMemObject(d_flat_words);
    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 3, batch2_words, &err);

    clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &b1_tw);

    clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &local_size, &local_size, 0, NULL, NULL);
    clFinish(queue);

    uint32_t h_num_pairs2, h_total2;
    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs2, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total2, 0, NULL, NULL);

    printf("  Batch 2: %u total pairs, %u total events\n", h_num_pairs2, h_total2);

    /* Run dirty MI again — should only recompute dirty pairs */
    n = (double)h_total2;
    clSetKernelArg(k_mi_dirty, 6, sizeof(cl_double), &n);
    clSetKernelArg(k_mi_dirty, 7, sizeof(cl_uint), &h_num_pairs2);

    gs = ((h_num_pairs2 + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_mi_dirty, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);

    /* Read all MI values */
    double mi_batch2[5];
    uint32_t flags_after2[5];
    clEnqueueReadBuffer(queue, pair_mi, CL_TRUE, 0,
        sizeof(double) * h_num_pairs2, mi_batch2, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, pair_flags, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_pairs2, flags_after2, 0, NULL, NULL);

    printf("  MI after batch 2:\n");
    int t3_pass = all_clear;
    for (uint32_t i = 0; i < h_num_pairs2; i++) {
        printf("    pair[%u] MI=%.4f flags=%u\n", i, mi_batch2[i], flags_after2[i]);
        if (flags_after2[i] != 0) t3_pass = 0;
        /* MI should be non-zero for all counted pairs */
        if (fabs(mi_batch2[i]) < 0.001) t3_pass = 0;
    }
    printf("  All flags cleared: %s\n",
           (t3_pass) ? "YES" : "NO");
    printf("  %s\n\n", t3_pass ? "PASS" : "FAIL");
    if (t3_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 4: MI statistics
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 4: MI statistics ---\n");

    /* Use the data from Test 3 (5 pairs, all with MI > 0) */
    cl_mem d_cnt_nz = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem d_cnt_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem d_cnt_at = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    cl_double threshold = 1.0;

    clSetKernelArg(k_mi_stats, 0, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_mi_stats, 1, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_mi_stats, 2, sizeof(cl_uint), &h_num_pairs2);
    clSetKernelArg(k_mi_stats, 3, sizeof(cl_double), &threshold);
    clSetKernelArg(k_mi_stats, 4, sizeof(cl_mem), &d_cnt_nz);
    clSetKernelArg(k_mi_stats, 5, sizeof(cl_mem), &d_cnt_pos);
    clSetKernelArg(k_mi_stats, 6, sizeof(cl_mem), &d_cnt_at);

    gs = ((h_num_pairs2 + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_mi_stats, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);

    uint32_t cnt_nz, cnt_pos, cnt_at;
    clEnqueueReadBuffer(queue, d_cnt_nz, CL_TRUE, 0, sizeof(uint32_t), &cnt_nz, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_cnt_pos, CL_TRUE, 0, sizeof(uint32_t), &cnt_pos, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_cnt_at, CL_TRUE, 0, sizeof(uint32_t), &cnt_at, 0, NULL, NULL);

    printf("  Pairs with count > 0: %u (expected %u)\n", cnt_nz, h_num_pairs2);
    printf("  Pairs with MI > 0:    %u\n", cnt_pos);
    printf("  Pairs with MI > %.1f: %u\n", threshold, cnt_at);

    /* Count expected on CPU */
    int exp_pos = 0, exp_at = 0;
    for (uint32_t i = 0; i < h_num_pairs2; i++) {
        if (mi_batch2[i] > 0.0) exp_pos++;
        if (mi_batch2[i] > threshold) exp_at++;
    }

    int t4_pass = (cnt_nz == h_num_pairs2) && (cnt_pos == (uint32_t)exp_pos)
                  && (cnt_at == (uint32_t)exp_at);
    printf("  Expected: pos=%d above=%.1f=%d\n", exp_pos, threshold, exp_at);
    printf("  %s\n\n", t4_pass ? "PASS" : "FAIL");
    if (t4_pass) pass_count++; else fail_count++;

    clReleaseMemObject(d_cnt_nz);
    clReleaseMemObject(d_cnt_pos);
    clReleaseMemObject(d_cnt_at);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 5: MI filter (compact high-MI pairs)
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 5: MI filter ---\n");

    uint32_t max_output = 100;
    cl_mem d_filt_idx = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * max_output, NULL, &err);
    cl_mem d_filt_mi = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * max_output, NULL, &err);
    cl_mem d_filt_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    cl_double mi_thresh = 1.0;

    clSetKernelArg(k_mi_filter, 0, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_mi_filter, 1, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_mi_filter, 2, sizeof(cl_uint), &h_num_pairs2);
    clSetKernelArg(k_mi_filter, 3, sizeof(cl_double), &mi_thresh);
    clSetKernelArg(k_mi_filter, 4, sizeof(cl_mem), &d_filt_idx);
    clSetKernelArg(k_mi_filter, 5, sizeof(cl_mem), &d_filt_mi);
    clSetKernelArg(k_mi_filter, 6, sizeof(cl_mem), &d_filt_cnt);
    clSetKernelArg(k_mi_filter, 7, sizeof(cl_uint), &max_output);

    gs = ((h_num_pairs2 + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_mi_filter, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);

    uint32_t filt_count;
    clEnqueueReadBuffer(queue, d_filt_cnt, CL_TRUE, 0,
        sizeof(uint32_t), &filt_count, 0, NULL, NULL);

    printf("  Pairs with MI > %.1f: %u (expected %u)\n",
           mi_thresh, filt_count, cnt_at);

    if (filt_count > 0 && filt_count <= max_output) {
        uint32_t* f_idx = malloc(sizeof(uint32_t) * filt_count);
        double*   f_mi  = malloc(sizeof(double) * filt_count);
        clEnqueueReadBuffer(queue, d_filt_idx, CL_TRUE, 0,
            sizeof(uint32_t) * filt_count, f_idx, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_filt_mi, CL_TRUE, 0,
            sizeof(double) * filt_count, f_mi, 0, NULL, NULL);

        printf("  Filtered pairs:\n");
        for (uint32_t i = 0; i < filt_count; i++) {
            printf("    pair[%u] MI=%.4f\n", f_idx[i], f_mi[i]);
        }
        free(f_idx); free(f_mi);
    }

    int t5_pass = (filt_count == cnt_at);
    printf("  %s\n\n", t5_pass ? "PASS" : "FAIL");
    if (t5_pass) pass_count++; else fail_count++;

    clReleaseMemObject(d_filt_idx);
    clReleaseMemObject(d_filt_mi);
    clReleaseMemObject(d_filt_cnt);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 6: Benchmark — 1000 sentences → MI
     *
     *  Full pipeline: count 1000 sentences then compute MI.
     *  Measures the end-to-end time with zero CPU↔GPU transfers
     *  between stages.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 6: Benchmark (1000 sentences → MI) ---\n");

    reset_pools(queue, pht_keys, pht_values, pair_count, pair_mi,
                pair_flags, word_count, pair_next_free, total_pair_count);

    /* Generate 1000 sentences */
    int bench_ns = 1000;
    uint32_t* bench_offsets = malloc(sizeof(uint32_t) * bench_ns);
    uint32_t* bench_lengths = malloc(sizeof(uint32_t) * bench_ns);

    uint64_t rng = 0xCAFEBABEDEADBEEFULL;
    uint32_t bench_tw = 0;
    for (int s = 0; s < bench_ns; s++) {
        bench_offsets[s] = bench_tw;
        rng += 0x9E3779B97F4A7C15ULL;
        bench_lengths[s] = 5 + (uint32_t)((rng >> 32) % 16);
        bench_tw += bench_lengths[s];
    }

    uint32_t* bench_words = malloc(sizeof(uint32_t) * bench_tw);
    rng = 0xFEEDFACE12345678ULL;
    for (uint32_t i = 0; i < bench_tw; i++) {
        rng += 0x9E3779B97F4A7C15ULL;
        double u = (double)(rng >> 32) / (double)0xFFFFFFFFU;
        bench_words[i] = (uint32_t)(u * u * 499.0);
    }

    printf("  Sentences: %d, Total words: %u\n", bench_ns, bench_tw);

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_tw, bench_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, bench_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, bench_lengths, &err);

    cl_uint bns = bench_ns;
    cl_uint bench_window = 6;

    clSetKernelArg(k_count, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_count, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_count, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_count, 3, sizeof(cl_uint), &bns);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &bench_tw);
    clSetKernelArg(k_count, 5, sizeof(cl_uint), &bench_window);
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

    /* Stage 1: Count */
    t0 = now_ms();
    gs = ((bench_tw + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();
    double count_ms = t1 - t0;

    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total, 0, NULL, NULL);

    printf("  Count: %u pairs, %u events in %.2f ms\n",
           h_num_pairs, h_total, count_ms);

    /* Stage 2: MI on all pairs */
    n = (double)h_total;
    clSetKernelArg(k_mi_all, 5, sizeof(cl_double), &n);
    clSetKernelArg(k_mi_all, 6, sizeof(cl_uint), &h_num_pairs);

    t0 = now_ms();
    gs = ((h_num_pairs + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_mi_all, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();
    double mi_ms = t1 - t0;

    printf("  MI:    %u pairs in %.2f ms (%.1f M pairs/sec)\n",
           h_num_pairs, mi_ms,
           h_num_pairs / (mi_ms / 1000.0) / 1e6);

    /* Stage 3: MI stats */
    cl_mem d_s_nz = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem d_s_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem d_s_at = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    threshold = 1.0;
    clSetKernelArg(k_mi_stats, 0, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_mi_stats, 1, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_mi_stats, 2, sizeof(cl_uint), &h_num_pairs);
    clSetKernelArg(k_mi_stats, 3, sizeof(cl_double), &threshold);
    clSetKernelArg(k_mi_stats, 4, sizeof(cl_mem), &d_s_nz);
    clSetKernelArg(k_mi_stats, 5, sizeof(cl_mem), &d_s_pos);
    clSetKernelArg(k_mi_stats, 6, sizeof(cl_mem), &d_s_at);

    t0 = now_ms();
    clEnqueueNDRangeKernel(queue, k_mi_stats, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    double stats_ms = now_ms() - t0;

    clEnqueueReadBuffer(queue, d_s_nz, CL_TRUE, 0, sizeof(uint32_t), &cnt_nz, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_s_pos, CL_TRUE, 0, sizeof(uint32_t), &cnt_pos, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_s_at, CL_TRUE, 0, sizeof(uint32_t), &cnt_at, 0, NULL, NULL);

    printf("  Stats: %u nonzero, %u positive MI, %u MI>%.1f in %.2f ms\n",
           cnt_nz, cnt_pos, cnt_at, threshold, stats_ms);

    /* Stage 4: Dirty MI (incremental — count another batch, recompute dirty only) */

    /* Reset only the counting state, keep existing MI values */
    /* First record current pair count */
    uint32_t pairs_before = h_num_pairs;

    /* Count same sentences again (adds to counts, marks dirty) */
    clSetKernelArg(k_count, 3, sizeof(cl_uint), &bns);
    clSetKernelArg(k_count, 4, sizeof(cl_uint), &bench_tw);

    t0 = now_ms();
    gs = ((bench_tw + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_count, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    double count2_ms = now_ms() - t0;

    clEnqueueReadBuffer(queue, pair_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_pairs, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_pair_count, CL_TRUE, 0,
        sizeof(uint32_t), &h_total, 0, NULL, NULL);

    printf("  Count (batch 2): %u pairs, %u events in %.2f ms\n",
           h_num_pairs, h_total, count2_ms);
    printf("  New pairs: %u (existing reused: %u)\n",
           h_num_pairs - pairs_before, pairs_before);

    /* Dirty MI recompute */
    n = (double)h_total;
    clSetKernelArg(k_mi_dirty, 6, sizeof(cl_double), &n);
    clSetKernelArg(k_mi_dirty, 7, sizeof(cl_uint), &h_num_pairs);

    t0 = now_ms();
    gs = ((h_num_pairs + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_mi_dirty, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    double dirty_ms = now_ms() - t0;

    printf("  Dirty MI: %u pairs scanned in %.2f ms\n", h_num_pairs, dirty_ms);
    printf("\n  === Pipeline Summary ===\n");
    printf("  Count (1000 sent):    %.2f ms\n", count_ms);
    printf("  MI (all %u pairs):    %.2f ms\n", pairs_before, mi_ms);
    printf("  Stats:                %.2f ms\n", stats_ms);
    printf("  Count (batch 2):      %.2f ms\n", count2_ms);
    printf("  MI (dirty only):      %.2f ms\n", dirty_ms);
    printf("  Total pipeline:       %.2f ms\n",
           count_ms + mi_ms + stats_ms + count2_ms + dirty_ms);
    printf("  CPU↔GPU transfers:    0 (all data GPU-resident)\n");

    int t6_pass = (h_num_pairs > 0) && (h_total > 0) && (cnt_nz > 0);
    printf("  %s\n\n", t6_pass ? "PASS" : "FAIL");
    if (t6_pass) pass_count++; else fail_count++;

    clReleaseMemObject(d_s_nz);
    clReleaseMemObject(d_s_pos);
    clReleaseMemObject(d_s_at);

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
    clReleaseKernel(k_mi_all);
    clReleaseKernel(k_mi_dirty);
    clReleaseKernel(k_mi_stats);
    clReleaseKernel(k_mi_filter);
    clReleaseKernel(k_read_mi);
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
    free(mi_src);

    return fail_count > 0 ? 1 : 0;
}
