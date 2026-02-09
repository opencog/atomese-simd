/*
 * test-substitute.c -- Test GPU class substitution kernel
 *
 * Compile: gcc -O2 -o test-substitute test-substitute.c -lOpenCL -lm
 * Run:     ./test-substitute
 *
 * Tests:
 *   1. Class assignment — batch set word_class_id
 *   2. Pair substitution — replace word indices (no dedup)
 *   3. Pair merge — duplicate pairs after substitution get merged
 *   4. Self-pair elimination — both words → same class → dropped
 *   5. Section word substitution
 *   6. Benchmark: 100K pairs, 1000 class assignments, substitute + rebuild
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

/* ─── Main ─── */

int main(int argc, char** argv)
{
    cl_int err;
    int pass_count = 0, fail_count = 0;

    printf("=== GPU Class Substitution Test ===\n\n");

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

    size_t len_ht, len_as, len_sub;
    char* src_ht  = read_file("gpu-hashtable.cl", &len_ht);
    char* src_as  = read_file("gpu-atomspace.cl", &len_as);
    char* src_sub = read_file("gpu-substitute.cl", &len_sub);

    size_t total_len = len_ht + 1 + len_as + 1 + len_sub;
    char* combined = malloc(total_len + 1);
    memcpy(combined, src_ht, len_ht);
    combined[len_ht] = '\n';
    memcpy(combined + len_ht + 1, src_as, len_as);
    combined[len_ht + 1 + len_as] = '\n';
    memcpy(combined + len_ht + 1 + len_as + 1, src_sub, len_sub);
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

    cl_kernel k_assign   = clCreateKernel(program, "assign_classes", &err);
    CL_CHECK(err, "kernel assign_classes");
    cl_kernel k_sub_pair = clCreateKernel(program, "substitute_pairs", &err);
    CL_CHECK(err, "kernel substitute_pairs");
    cl_kernel k_rebuild  = clCreateKernel(program, "rebuild_pair_index", &err);
    CL_CHECK(err, "kernel rebuild_pair_index");
    cl_kernel k_sub_sec  = clCreateKernel(program, "substitute_section_words", &err);
    CL_CHECK(err, "kernel substitute_section_words");

    size_t local_size = 256;

    /* ─── Allocate GPU buffers ─── */

    printf("Allocating GPU buffers...\n");
    uint32_t zero = 0;
    uint8_t pat_ff = 0xFF;
    uint8_t pat_00 = 0x00;

    /* Word class IDs */
    cl_mem word_class_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * WORD_CAPACITY, NULL, &err);
    CL_CHECK(err, "word_class_id");
    clEnqueueFillBuffer(queue, word_class_id, &pat_00, 1, 0,
        sizeof(uint32_t) * WORD_CAPACITY, 0, NULL, NULL);

    /* Pair pool */
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

    /* Pair hash table */
    cl_mem pht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * PAIR_HT_CAPACITY, NULL, &err);
    cl_mem pht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_HT_CAPACITY, NULL, &err);

    /* Section pool (for test 5) */
    cl_mem sec_word = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_CAPACITY, NULL, &err);
    cl_mem sec_count_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * SECTION_CAPACITY, NULL, &err);

    /* Counters */
    cl_mem num_changed = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem num_eliminated = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_mem num_merged = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    clFinish(queue);
    printf("GPU buffers ready\n\n");

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 1: Class assignment
     *
     *  Assign: word 10 → class 100, word 20 → class 100
     *  Verify word_class_id[10] = 100, word_class_id[20] = 100
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 1: Class assignment ---\n");

    /* Reset word_class_id */
    clEnqueueFillBuffer(queue, word_class_id, &pat_00, 1, 0,
        sizeof(uint32_t) * WORD_CAPACITY, 0, NULL, NULL);
    clFinish(queue);

    {
        uint32_t word_indices[] = {10, 20};
        uint32_t class_ids[] = {100, 100};
        cl_uint na = 2;

        cl_mem d_widx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * na, word_indices, &err);
        cl_mem d_cids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * na, class_ids, &err);

        clSetKernelArg(k_assign, 0, sizeof(cl_mem), &word_class_id);
        clSetKernelArg(k_assign, 1, sizeof(cl_mem), &d_widx);
        clSetKernelArg(k_assign, 2, sizeof(cl_mem), &d_cids);
        clSetKernelArg(k_assign, 3, sizeof(cl_uint), &na);

        size_t gs = ((na + local_size - 1) / local_size) * local_size;
        err = clEnqueueNDRangeKernel(queue, k_assign, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        CL_CHECK(err, "enqueue assign");
        clFinish(queue);

        /* Read back */
        uint32_t h_cls[32];
        clEnqueueReadBuffer(queue, word_class_id, CL_TRUE, 0,
            sizeof(uint32_t) * 32, h_cls, 0, NULL, NULL);

        printf("  word_class_id[10] = %u (expected 100)\n", h_cls[10]);
        printf("  word_class_id[20] = %u (expected 100)\n", h_cls[20]);
        printf("  word_class_id[0]  = %u (expected 0)\n", h_cls[0]);
        printf("  word_class_id[15] = %u (expected 0)\n", h_cls[15]);

        int pass = (h_cls[10] == 100) && (h_cls[20] == 100) &&
                   (h_cls[0] == 0) && (h_cls[15] == 0);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;

        clReleaseMemObject(d_widx);
        clReleaseMemObject(d_cids);
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 2: Pair substitution (no dedup)
     *
     *  Pairs: (10, 30) count=5, (20, 40) count=3
     *  Classes: word 10 → class 100 (already assigned from test 1)
     *
     *  After substitution:
     *    pair 0: (30, 100) count=5  [was (10, 30), recanonized]
     *    pair 1: (20, 40) count=3   [word 20=class 100, so → (40, 100)]
     *
     *  Wait — word 20 is ALSO class 100 from test 1! So:
     *    pair 1: (40, 100) count=3
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 2: Pair substitution (no dedup) ---\n");

    {
        uint32_t h_wa[] = {10, 20};
        uint32_t h_wb[] = {30, 40};
        double   h_cnt[] = {5.0, 3.0};
        double   h_mi[] = {2.5, 1.5};
        uint32_t h_flg[] = {0, 0};
        cl_uint np = 2;

        clEnqueueWriteBuffer(queue, pair_word_a, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_wa, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_word_b, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_wb, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_count, CL_FALSE, 0,
            sizeof(double) * np, h_cnt, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_mi, CL_FALSE, 0,
            sizeof(double) * np, h_mi, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_flags, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_flg, 0, NULL, NULL);

        clEnqueueWriteBuffer(queue, num_changed, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_eliminated, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clFinish(queue);

        clSetKernelArg(k_sub_pair, 0, sizeof(cl_mem), &pair_word_a);
        clSetKernelArg(k_sub_pair, 1, sizeof(cl_mem), &pair_word_b);
        clSetKernelArg(k_sub_pair, 2, sizeof(cl_mem), &pair_count);
        clSetKernelArg(k_sub_pair, 3, sizeof(cl_mem), &pair_mi);
        clSetKernelArg(k_sub_pair, 4, sizeof(cl_mem), &pair_flags);
        clSetKernelArg(k_sub_pair, 5, sizeof(cl_mem), &word_class_id);
        clSetKernelArg(k_sub_pair, 6, sizeof(cl_mem), &num_changed);
        clSetKernelArg(k_sub_pair, 7, sizeof(cl_mem), &num_eliminated);
        clSetKernelArg(k_sub_pair, 8, sizeof(cl_uint), &np);

        size_t gs = ((np + local_size - 1) / local_size) * local_size;
        err = clEnqueueNDRangeKernel(queue, k_sub_pair, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        CL_CHECK(err, "enqueue substitute_pairs");
        clFinish(queue);

        /* Read back */
        uint32_t r_wa[2], r_wb[2], r_flg[2];
        double r_cnt[2];
        clEnqueueReadBuffer(queue, pair_word_a, CL_TRUE, 0,
            sizeof(uint32_t) * np, r_wa, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, pair_word_b, CL_TRUE, 0,
            sizeof(uint32_t) * np, r_wb, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, pair_count, CL_TRUE, 0,
            sizeof(double) * np, r_cnt, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, pair_flags, CL_TRUE, 0,
            sizeof(uint32_t) * np, r_flg, 0, NULL, NULL);

        uint32_t r_nchanged;
        clEnqueueReadBuffer(queue, num_changed, CL_TRUE, 0,
            sizeof(uint32_t), &r_nchanged, 0, NULL, NULL);

        /* Pair 0: (10,30) → (100,30) → canonical (30,100), count=5 */
        /* Pair 1: (20,40) → (100,40) → canonical (40,100), count=3 */
        printf("  Pair 0: (%u, %u) count=%.0f flags=%u\n",
               r_wa[0], r_wb[0], r_cnt[0], r_flg[0]);
        printf("    Expected: (30, 100) count=5 flags=1\n");
        printf("  Pair 1: (%u, %u) count=%.0f flags=%u\n",
               r_wa[1], r_wb[1], r_cnt[1], r_flg[1]);
        printf("    Expected: (40, 100) count=3 flags=1\n");
        printf("  Changed: %u (expected 2)\n", r_nchanged);

        int pass = (r_wa[0] == 30) && (r_wb[0] == 100) &&
                   (fabs(r_cnt[0] - 5.0) < 0.01) && (r_flg[0] == 1) &&
                   (r_wa[1] == 40) && (r_wb[1] == 100) &&
                   (fabs(r_cnt[1] - 3.0) < 0.01) && (r_flg[1] == 1) &&
                   (r_nchanged == 2);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 3: Pair merge — duplicate pairs after substitution
     *
     *  Fresh setup:
     *    Pair 0: (10, 30) count=5
     *    Pair 1: (20, 30) count=3
     *    Classes: word 10 → class 100, word 20 → class 100
     *
     *  After substitute_pairs:
     *    Pair 0: (30, 100) count=5
     *    Pair 1: (30, 100) count=3   ← duplicate!
     *
     *  After rebuild_pair_index:
     *    One pair (30, 100) count=8, other zeroed
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 3: Pair merge (duplicate pairs) ---\n");

    /* Reset class IDs and reassign */
    clEnqueueFillBuffer(queue, word_class_id, &pat_00, 1, 0,
        sizeof(uint32_t) * WORD_CAPACITY, 0, NULL, NULL);
    clFinish(queue);

    {
        uint32_t word_indices[] = {10, 20};
        uint32_t class_ids[] = {100, 100};
        cl_uint na = 2;
        cl_mem d_widx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * na, word_indices, &err);
        cl_mem d_cids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * na, class_ids, &err);
        clSetKernelArg(k_assign, 0, sizeof(cl_mem), &word_class_id);
        clSetKernelArg(k_assign, 1, sizeof(cl_mem), &d_widx);
        clSetKernelArg(k_assign, 2, sizeof(cl_mem), &d_cids);
        clSetKernelArg(k_assign, 3, sizeof(cl_uint), &na);
        size_t gs = ((na + local_size - 1) / local_size) * local_size;
        clEnqueueNDRangeKernel(queue, k_assign, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        clFinish(queue);
        clReleaseMemObject(d_widx);
        clReleaseMemObject(d_cids);
    }

    {
        uint32_t h_wa[] = {10, 20};
        uint32_t h_wb[] = {30, 30};
        double h_cnt[] = {5.0, 3.0};
        double h_mi[] = {2.5, 1.5};
        uint32_t h_flg[] = {0, 0};
        cl_uint np = 2;

        clEnqueueWriteBuffer(queue, pair_word_a, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_wa, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_word_b, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_wb, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_count, CL_FALSE, 0,
            sizeof(double) * np, h_cnt, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_mi, CL_FALSE, 0,
            sizeof(double) * np, h_mi, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_flags, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_flg, 0, NULL, NULL);

        clEnqueueWriteBuffer(queue, num_changed, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_eliminated, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_merged, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clFinish(queue);

        /* Step 1: Substitute */
        clSetKernelArg(k_sub_pair, 8, sizeof(cl_uint), &np);
        size_t gs2 = ((np + local_size - 1) / local_size) * local_size;
        clEnqueueNDRangeKernel(queue, k_sub_pair, 1, NULL,
            &gs2, &local_size, 0, NULL, NULL);
        clFinish(queue);

        /* Step 2: Clear pair HT and rebuild */
        clEnqueueFillBuffer(queue, pht_keys, &pat_ff, 1, 0,
            sizeof(uint64_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, pht_values, &pat_ff, 1, 0,
            sizeof(uint32_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
        clFinish(queue);

        clSetKernelArg(k_rebuild, 0, sizeof(cl_mem), &pair_word_a);
        clSetKernelArg(k_rebuild, 1, sizeof(cl_mem), &pair_word_b);
        clSetKernelArg(k_rebuild, 2, sizeof(cl_mem), &pair_count);
        clSetKernelArg(k_rebuild, 3, sizeof(cl_mem), &pair_mi);
        clSetKernelArg(k_rebuild, 4, sizeof(cl_mem), &pair_flags);
        clSetKernelArg(k_rebuild, 5, sizeof(cl_mem), &pht_keys);
        clSetKernelArg(k_rebuild, 6, sizeof(cl_mem), &pht_values);
        clSetKernelArg(k_rebuild, 7, sizeof(cl_mem), &num_merged);
        clSetKernelArg(k_rebuild, 8, sizeof(cl_uint), &np);

        clEnqueueNDRangeKernel(queue, k_rebuild, 1, NULL,
            &gs2, &local_size, 0, NULL, NULL);
        clFinish(queue);

        /* Read back */
        double r_cnt[2];
        clEnqueueReadBuffer(queue, pair_count, CL_TRUE, 0,
            sizeof(double) * np, r_cnt, 0, NULL, NULL);

        uint32_t r_nmerged;
        clEnqueueReadBuffer(queue, num_merged, CL_TRUE, 0,
            sizeof(uint32_t), &r_nmerged, 0, NULL, NULL);

        /* One pair should have count=8, other count=0 */
        double total = r_cnt[0] + r_cnt[1];
        int one_is_eight = (fabs(r_cnt[0] - 8.0) < 0.01) || (fabs(r_cnt[1] - 8.0) < 0.01);
        int one_is_zero = (fabs(r_cnt[0]) < 0.01) || (fabs(r_cnt[1]) < 0.01);

        printf("  Pair 0 count: %.1f\n", r_cnt[0]);
        printf("  Pair 1 count: %.1f\n", r_cnt[1]);
        printf("  Total: %.1f (expected 8.0)\n", total);
        printf("  Merged: %u (expected 1)\n", r_nmerged);

        int pass = (fabs(total - 8.0) < 0.01) && one_is_eight &&
                   one_is_zero && (r_nmerged == 1);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 4: Self-pair elimination
     *
     *  Pair: (10, 20) count=7
     *  Classes: word 10 → class 100, word 20 → class 100
     *
     *  After substitution: (100, 100) → self-pair → eliminated
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 4: Self-pair elimination ---\n");

    /* Class IDs still set from test 3: word 10→100, word 20→100 */

    {
        uint32_t h_wa[] = {10};
        uint32_t h_wb[] = {20};
        double h_cnt[] = {7.0};
        double h_mi[] = {3.0};
        uint32_t h_flg[] = {0};
        cl_uint np = 1;

        clEnqueueWriteBuffer(queue, pair_word_a, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_wa, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_word_b, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_wb, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_count, CL_FALSE, 0,
            sizeof(double) * np, h_cnt, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_mi, CL_FALSE, 0,
            sizeof(double) * np, h_mi, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_flags, CL_FALSE, 0,
            sizeof(uint32_t) * np, h_flg, 0, NULL, NULL);

        clEnqueueWriteBuffer(queue, num_changed, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_eliminated, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clFinish(queue);

        clSetKernelArg(k_sub_pair, 8, sizeof(cl_uint), &np);
        size_t gs = ((np + local_size - 1) / local_size) * local_size;
        clEnqueueNDRangeKernel(queue, k_sub_pair, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        clFinish(queue);

        double r_cnt;
        uint32_t r_nelim;
        clEnqueueReadBuffer(queue, pair_count, CL_TRUE, 0,
            sizeof(double), &r_cnt, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, num_eliminated, CL_TRUE, 0,
            sizeof(uint32_t), &r_nelim, 0, NULL, NULL);

        printf("  Pair count after: %.1f (expected 0.0 — eliminated)\n", r_cnt);
        printf("  Eliminated: %u (expected 1)\n", r_nelim);

        int pass = (fabs(r_cnt) < 0.01) && (r_nelim == 1);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 5: Section word substitution
     *
     *  Sections: (word=10, count=5), (word=30, count=2), (word=20, count=4)
     *  Classes: word 10 → 100, word 20 → 100 (from test 3)
     *
     *  After substitution:
     *    Section 0: word=100 (was 10)
     *    Section 1: word=30  (unchanged)
     *    Section 2: word=100 (was 20)
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 5: Section word substitution ---\n");

    {
        uint32_t h_sw[] = {10, 30, 20};
        double h_sc[] = {5.0, 2.0, 4.0};
        cl_uint ns = 3;

        clEnqueueWriteBuffer(queue, sec_word, CL_FALSE, 0,
            sizeof(uint32_t) * ns, h_sw, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, sec_count_buf, CL_FALSE, 0,
            sizeof(double) * ns, h_sc, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_changed, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clFinish(queue);

        clSetKernelArg(k_sub_sec, 0, sizeof(cl_mem), &sec_word);
        clSetKernelArg(k_sub_sec, 1, sizeof(cl_mem), &sec_count_buf);
        clSetKernelArg(k_sub_sec, 2, sizeof(cl_mem), &word_class_id);
        clSetKernelArg(k_sub_sec, 3, sizeof(cl_mem), &num_changed);
        clSetKernelArg(k_sub_sec, 4, sizeof(cl_uint), &ns);

        size_t gs = ((ns + local_size - 1) / local_size) * local_size;
        clEnqueueNDRangeKernel(queue, k_sub_sec, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        clFinish(queue);

        uint32_t r_sw[3];
        uint32_t r_nch;
        clEnqueueReadBuffer(queue, sec_word, CL_TRUE, 0,
            sizeof(uint32_t) * ns, r_sw, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, num_changed, CL_TRUE, 0,
            sizeof(uint32_t), &r_nch, 0, NULL, NULL);

        printf("  Section 0 word: %u (expected 100)\n", r_sw[0]);
        printf("  Section 1 word: %u (expected 30)\n", r_sw[1]);
        printf("  Section 2 word: %u (expected 100)\n", r_sw[2]);
        printf("  Changed: %u (expected 2)\n", r_nch);

        int pass = (r_sw[0] == 100) && (r_sw[1] == 30) && (r_sw[2] == 100) &&
                   (r_nch == 2);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 6: Benchmark — 100K pairs, 1000 class assignments
     *
     *  100 words assigned to 20 classes.
     *  100K pairs with random word indices (0..999).
     *  Measure substitute + rebuild time.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 6: Benchmark (100K pairs, 100 word→20 class) ---\n");

    /* Reset class IDs */
    clEnqueueFillBuffer(queue, word_class_id, &pat_00, 1, 0,
        sizeof(uint32_t) * WORD_CAPACITY, 0, NULL, NULL);
    clFinish(queue);

    srand(42);
    uint32_t bench_np = 100000;
    uint32_t num_classes = 20;
    uint32_t words_per_class = 5;
    uint32_t total_assigned = num_classes * words_per_class;  /* 100 */
    uint32_t vocab = 1000;

    /* Assign classes */
    {
        uint32_t* h_widx = malloc(sizeof(uint32_t) * total_assigned);
        uint32_t* h_cids = malloc(sizeof(uint32_t) * total_assigned);
        for (uint32_t c = 0; c < num_classes; c++) {
            for (uint32_t w = 0; w < words_per_class; w++) {
                uint32_t idx = c * words_per_class + w;
                h_widx[idx] = c * words_per_class + w;  /* words 0..99 */
                h_cids[idx] = 10000 + c;  /* class IDs 10000..10019 */
            }
        }

        cl_mem d_widx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * total_assigned, h_widx, &err);
        cl_mem d_cids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t) * total_assigned, h_cids, &err);

        clSetKernelArg(k_assign, 0, sizeof(cl_mem), &word_class_id);
        clSetKernelArg(k_assign, 1, sizeof(cl_mem), &d_widx);
        clSetKernelArg(k_assign, 2, sizeof(cl_mem), &d_cids);
        cl_uint na2 = total_assigned;
        clSetKernelArg(k_assign, 3, sizeof(cl_uint), &na2);
        size_t gs = ((na2 + local_size - 1) / local_size) * local_size;
        clEnqueueNDRangeKernel(queue, k_assign, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        clFinish(queue);

        free(h_widx); free(h_cids);
        clReleaseMemObject(d_widx);
        clReleaseMemObject(d_cids);
    }

    /* Generate random pairs */
    {
        uint32_t* h_wa = malloc(sizeof(uint32_t) * bench_np);
        uint32_t* h_wb = malloc(sizeof(uint32_t) * bench_np);
        double* h_cnt = malloc(sizeof(double) * bench_np);
        double* h_mi_arr = malloc(sizeof(double) * bench_np);
        uint32_t* h_flg = malloc(sizeof(uint32_t) * bench_np);

        for (uint32_t i = 0; i < bench_np; i++) {
            uint32_t a = rand() % vocab;
            uint32_t b = rand() % vocab;
            while (b == a) b = rand() % vocab;
            h_wa[i] = (a < b) ? a : b;
            h_wb[i] = (a < b) ? b : a;
            h_cnt[i] = 1.0 + (rand() % 100);
            h_mi_arr[i] = 0.5 + (double)(rand() % 100) / 50.0;
            h_flg[i] = 0;
        }

        clEnqueueWriteBuffer(queue, pair_word_a, CL_FALSE, 0,
            sizeof(uint32_t) * bench_np, h_wa, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_word_b, CL_FALSE, 0,
            sizeof(uint32_t) * bench_np, h_wb, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_count, CL_FALSE, 0,
            sizeof(double) * bench_np, h_cnt, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_mi, CL_FALSE, 0,
            sizeof(double) * bench_np, h_mi_arr, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, pair_flags, CL_FALSE, 0,
            sizeof(uint32_t) * bench_np, h_flg, 0, NULL, NULL);

        clEnqueueWriteBuffer(queue, num_changed, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_eliminated, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, num_merged, CL_FALSE, 0,
            sizeof(uint32_t), &zero, 0, NULL, NULL);
        clFinish(queue);

        /* Substitute */
        cl_uint np = bench_np;
        clSetKernelArg(k_sub_pair, 8, sizeof(cl_uint), &np);

        double t0 = now_ms();
        size_t gs = ((np + local_size - 1) / local_size) * local_size;
        clEnqueueNDRangeKernel(queue, k_sub_pair, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        clFinish(queue);
        double t1 = now_ms();

        /* Rebuild */
        clEnqueueFillBuffer(queue, pht_keys, &pat_ff, 1, 0,
            sizeof(uint64_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, pht_values, &pat_ff, 1, 0,
            sizeof(uint32_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
        clFinish(queue);

        clSetKernelArg(k_rebuild, 0, sizeof(cl_mem), &pair_word_a);
        clSetKernelArg(k_rebuild, 1, sizeof(cl_mem), &pair_word_b);
        clSetKernelArg(k_rebuild, 2, sizeof(cl_mem), &pair_count);
        clSetKernelArg(k_rebuild, 3, sizeof(cl_mem), &pair_mi);
        clSetKernelArg(k_rebuild, 4, sizeof(cl_mem), &pair_flags);
        clSetKernelArg(k_rebuild, 5, sizeof(cl_mem), &pht_keys);
        clSetKernelArg(k_rebuild, 6, sizeof(cl_mem), &pht_values);
        clSetKernelArg(k_rebuild, 7, sizeof(cl_mem), &num_merged);
        clSetKernelArg(k_rebuild, 8, sizeof(cl_uint), &np);

        double t2 = now_ms();
        clEnqueueNDRangeKernel(queue, k_rebuild, 1, NULL,
            &gs, &local_size, 0, NULL, NULL);
        clFinish(queue);
        double t3 = now_ms();

        uint32_t r_nch, r_nel, r_nmg;
        clEnqueueReadBuffer(queue, num_changed, CL_TRUE, 0,
            sizeof(uint32_t), &r_nch, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, num_eliminated, CL_TRUE, 0,
            sizeof(uint32_t), &r_nel, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, num_merged, CL_TRUE, 0,
            sizeof(uint32_t), &r_nmg, 0, NULL, NULL);

        printf("  Pairs: %u\n", bench_np);
        printf("  Substitute: %.2f ms\n", t1 - t0);
        printf("  Rebuild:    %.2f ms\n", t3 - t2);
        printf("  Total:      %.2f ms\n", (t1 - t0) + (t3 - t2));
        printf("  Changed: %u, Eliminated: %u, Merged: %u\n",
               r_nch, r_nel, r_nmg);
        printf("  Throughput: %.1fM pairs/sec\n",
               bench_np / (((t1 - t0) + (t3 - t2)) / 1000.0) / 1e6);

        int pass = (r_nch > 0) && ((t1 - t0) + (t3 - t2) < 1000.0);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;

        free(h_wa); free(h_wb); free(h_cnt); free(h_mi_arr); free(h_flg);
    }

    /* ─── Summary ─── */

    printf("=== Results: %d PASS, %d FAIL ===\n", pass_count, fail_count);

    /* Cleanup */
    free(src_ht); free(src_as); free(src_sub); free(combined);
    clReleaseMemObject(word_class_id);
    clReleaseMemObject(pair_word_a);
    clReleaseMemObject(pair_word_b);
    clReleaseMemObject(pair_count);
    clReleaseMemObject(pair_mi);
    clReleaseMemObject(pair_flags);
    clReleaseMemObject(pht_keys);
    clReleaseMemObject(pht_values);
    clReleaseMemObject(sec_word);
    clReleaseMemObject(sec_count_buf);
    clReleaseMemObject(num_changed);
    clReleaseMemObject(num_eliminated);
    clReleaseMemObject(num_merged);
    clReleaseKernel(k_assign);
    clReleaseKernel(k_sub_pair);
    clReleaseKernel(k_rebuild);
    clReleaseKernel(k_sub_sec);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return fail_count > 0 ? 1 : 0;
}
