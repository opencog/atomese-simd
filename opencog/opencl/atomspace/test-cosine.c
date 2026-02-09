/*
 * test-cosine.c -- Test GPU cosine similarity and candidate generation
 *
 * Compile: gcc -O2 -o test-cosine test-cosine.c -lOpenCL -lm
 * Run:     ./test-cosine
 *
 * Tests:
 *   1. Known cosine (2 words, 4 sections, exact verification)
 *   2. Three words — all pairwise cosines
 *   3. Identical vectors → cosine = 1.0
 *   4. No shared disjuncts → 0 candidates
 *   5. Filter candidates above threshold
 *   6. Benchmark: 1000 sentences → sections → cosines (full pipeline)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

/* ─── Pool capacities ─── */

#define WORD_CAPACITY          (128 * 1024)
#define PAIR_CAPACITY          (4 * 1024 * 1024)
#define SECTION_CAPACITY       (1024 * 1024)
#define WORD_HT_CAPACITY       (256 * 1024)
#define PAIR_HT_CAPACITY       (8 * 1024 * 1024)
#define SECTION_HT_CAPACITY    (2 * 1024 * 1024)

/* Phase 5 capacities */
#define DJH_HT_CAPACITY        (2 * 1024 * 1024)
#define CANDIDATE_CAPACITY     (512 * 1024)
#define CANDIDATE_HT_CAPACITY  (1024 * 1024)

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

/* ─── Helper: Concatenate sources ─── */

char* concat_sources(const char** srcs, size_t* lens, int n, size_t* total)
{
    *total = 0;
    for (int i = 0; i < n; i++) *total += lens[i] + 1;
    char* buf = malloc(*total + 1);
    size_t pos = 0;
    for (int i = 0; i < n; i++) {
        memcpy(buf + pos, srcs[i], lens[i]);
        pos += lens[i];
        buf[pos++] = '\n';
    }
    buf[pos] = '\0';
    *total = pos;
    return buf;
}

/* ─── Helper: reset cosine pipeline buffers ─── */

void reset_cosine_buffers(cl_command_queue queue,
    cl_mem djh_ht_keys, cl_mem djh_ht_values,
    cl_mem sec_chain_next, cl_mem word_norm_sq,
    cl_mem cand_ht_keys, cl_mem cand_ht_values,
    cl_mem cand_dot, cl_mem cand_cosine,
    cl_mem cand_next_free)
{
    uint8_t pat_ff = 0xFF;
    uint8_t pat_00 = 0x00;
    uint32_t zero = 0;

    /* Disjunct reverse index HT */
    clEnqueueFillBuffer(queue, djh_ht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * DJH_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, djh_ht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * DJH_HT_CAPACITY, 0, NULL, NULL);

    /* Section chain pointers */
    clEnqueueFillBuffer(queue, sec_chain_next, &pat_ff, 1, 0,
        sizeof(uint32_t) * SECTION_CAPACITY, 0, NULL, NULL);

    /* Word norms */
    clEnqueueFillBuffer(queue, word_norm_sq, &pat_00, 1, 0,
        sizeof(double) * WORD_CAPACITY, 0, NULL, NULL);

    /* Candidate HT */
    clEnqueueFillBuffer(queue, cand_ht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * CANDIDATE_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, cand_ht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * CANDIDATE_HT_CAPACITY, 0, NULL, NULL);

    /* Candidate pool */
    clEnqueueFillBuffer(queue, cand_dot, &pat_00, 1, 0,
        sizeof(double) * CANDIDATE_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, cand_cosine, &pat_00, 1, 0,
        sizeof(double) * CANDIDATE_CAPACITY, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cand_next_free, CL_FALSE, 0,
        sizeof(uint32_t), &zero, 0, NULL, NULL);

    clFinish(queue);
}

/* ─── Helper: reset section pool ─── */

void reset_section_pool(cl_command_queue queue,
    cl_mem sht_keys, cl_mem sht_values,
    cl_mem sec_count, cl_mem sec_next_free)
{
    uint8_t pat_ff = 0xFF;
    uint8_t pat_00 = 0x00;
    uint32_t zero = 0;

    clEnqueueFillBuffer(queue, sht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * SECTION_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, sht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * SECTION_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, sec_count, &pat_00, 1, 0,
        sizeof(double) * SECTION_CAPACITY, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, sec_next_free, CL_FALSE, 0,
        sizeof(uint32_t), &zero, 0, NULL, NULL);
    clFinish(queue);
}

/* ─── Helper: manually populate sections ─── */

void upload_sections(cl_command_queue queue,
    cl_mem sec_word, cl_mem sec_disjunct_hash, cl_mem sec_count,
    cl_mem sec_next_free,
    uint32_t* words, uint64_t* djhs, double* counts, uint32_t n)
{
    clEnqueueWriteBuffer(queue, sec_word, CL_FALSE, 0,
        sizeof(uint32_t) * n, words, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, sec_disjunct_hash, CL_FALSE, 0,
        sizeof(uint64_t) * n, djhs, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, sec_count, CL_FALSE, 0,
        sizeof(double) * n, counts, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, sec_next_free, CL_FALSE, 0,
        sizeof(uint32_t), &n, 0, NULL, NULL);
    clFinish(queue);
}

/* ─── Helper: run cosine pipeline ─── */

void run_cosine_pipeline(cl_command_queue queue,
    cl_kernel k_norms, cl_kernel k_chains,
    cl_kernel k_dots, cl_kernel k_cosines,
    uint32_t num_sections,
    cl_mem cand_next_free, uint32_t* out_num_candidates)
{
    size_t local = 256;
    size_t gs;
    cl_int err;

    /* Step 1: Word norms */
    gs = ((num_sections + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(queue, k_norms, 1, NULL,
        &gs, &local, 0, NULL, NULL);
    CL_CHECK(err, "enqueue compute_word_norms");

    /* Step 2: Disjunct chains */
    err = clEnqueueNDRangeKernel(queue, k_chains, 1, NULL,
        &gs, &local, 0, NULL, NULL);
    CL_CHECK(err, "enqueue build_disjunct_chains");

    /* Step 3: Dot products */
    err = clEnqueueNDRangeKernel(queue, k_dots, 1, NULL,
        &gs, &local, 0, NULL, NULL);
    CL_CHECK(err, "enqueue accumulate_dot_products");

    clFinish(queue);

    /* Read candidate count */
    clEnqueueReadBuffer(queue, cand_next_free, CL_TRUE, 0,
        sizeof(uint32_t), out_num_candidates, 0, NULL, NULL);

    if (*out_num_candidates == 0) return;

    /* Step 4: Cosines */
    gs = ((*out_num_candidates + local - 1) / local) * local;
    cl_uint nc = *out_num_candidates;
    clSetKernelArg(k_cosines, 5, sizeof(cl_uint), &nc);
    err = clEnqueueNDRangeKernel(queue, k_cosines, 1, NULL,
        &gs, &local, 0, NULL, NULL);
    CL_CHECK(err, "enqueue compute_cosines");

    clFinish(queue);
}

/* ─── Main ─── */

int main(int argc, char** argv)
{
    cl_int err;
    int pass_count = 0, fail_count = 0;

    printf("=== GPU Cosine Similarity Test ===\n\n");

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

    size_t len_ht, len_as, len_sc, len_cos;
    char* src_ht  = read_file("gpu-hashtable.cl", &len_ht);
    char* src_as  = read_file("gpu-atomspace.cl", &len_as);
    char* src_sc  = read_file("gpu-sections.cl", &len_sc);
    char* src_cos = read_file("gpu-cosine.cl", &len_cos);

    const char* srcs[] = {src_ht, src_as, src_sc, src_cos};
    size_t lens[] = {len_ht, len_as, len_sc, len_cos};
    size_t total_len;
    char* combined = concat_sources(srcs, lens, 4, &total_len);

    cl_program program = clCreateProgramWithSource(ctx, 1,
        (const char**)&combined, &total_len, &err);
    CL_CHECK(err, "create program");

    char build_opts[1024];
    snprintf(build_opts, sizeof(build_opts),
        "-cl-std=CL1.2 "
        "-DWORD_CAPACITY=%d "
        "-DPAIR_CAPACITY=%d "
        "-DSECTION_CAPACITY=%d "
        "-DWORD_HT_CAPACITY=%d "
        "-DPAIR_HT_CAPACITY=%d "
        "-DSECTION_HT_CAPACITY=%d "
        "-DDJH_HT_CAPACITY=%d "
        "-DCANDIDATE_CAPACITY=%d "
        "-DCANDIDATE_HT_CAPACITY=%d",
        WORD_CAPACITY, PAIR_CAPACITY, SECTION_CAPACITY,
        WORD_HT_CAPACITY, PAIR_HT_CAPACITY, SECTION_HT_CAPACITY,
        DJH_HT_CAPACITY, CANDIDATE_CAPACITY, CANDIDATE_HT_CAPACITY);

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

    cl_kernel k_norms   = clCreateKernel(program, "compute_word_norms", &err);
    CL_CHECK(err, "kernel compute_word_norms");
    cl_kernel k_chains  = clCreateKernel(program, "build_disjunct_chains", &err);
    CL_CHECK(err, "kernel build_disjunct_chains");
    cl_kernel k_dots    = clCreateKernel(program, "accumulate_dot_products", &err);
    CL_CHECK(err, "kernel accumulate_dot_products");
    cl_kernel k_cosines = clCreateKernel(program, "compute_cosines", &err);
    CL_CHECK(err, "kernel compute_cosines");
    cl_kernel k_filter  = clCreateKernel(program, "filter_candidates", &err);
    CL_CHECK(err, "kernel filter_candidates");
    cl_kernel k_extract = clCreateKernel(program, "extract_sections", &err);
    CL_CHECK(err, "kernel extract_sections");

    /* ─── Allocate GPU buffers ─── */

    printf("Allocating GPU buffers...\n");
    uint32_t zero = 0;

    /* Section pool */
    cl_mem sec_word = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_CAPACITY, NULL, &err);
    cl_mem sec_disjunct_hash = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * SECTION_CAPACITY, NULL, &err);
    cl_mem sec_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * SECTION_CAPACITY, NULL, &err);
    cl_mem sec_next_free = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Section hash table (for extract_sections) */
    cl_mem sht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * SECTION_HT_CAPACITY, NULL, &err);
    cl_mem sht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_HT_CAPACITY, NULL, &err);

    /* Total sections counter (for extract_sections) */
    cl_mem total_sections_created = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Disjunct reverse index HT */
    cl_mem djh_ht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * DJH_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "djh_ht_keys");
    cl_mem djh_ht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * DJH_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "djh_ht_values");

    /* Section chain pointers */
    cl_mem sec_chain_next = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_CAPACITY, NULL, &err);
    CL_CHECK(err, "sec_chain_next");

    /* Word norms */
    cl_mem word_norm_sq = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * WORD_CAPACITY, NULL, &err);
    CL_CHECK(err, "word_norm_sq");

    /* Candidate HT */
    cl_mem cand_ht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * CANDIDATE_HT_CAPACITY, NULL, &err);
    cl_mem cand_ht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * CANDIDATE_HT_CAPACITY, NULL, &err);

    /* Candidate pool */
    cl_mem cand_word_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * CANDIDATE_CAPACITY, NULL, &err);
    cl_mem cand_word_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * CANDIDATE_CAPACITY, NULL, &err);
    cl_mem cand_dot = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * CANDIDATE_CAPACITY, NULL, &err);
    cl_mem cand_cosine = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * CANDIDATE_CAPACITY, NULL, &err);
    cl_mem cand_next_free = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* ─── Set kernel args (persistent across tests) ─── */

    /* compute_word_norms: (sec_word, sec_count, word_norm_sq, num_sections) */
    clSetKernelArg(k_norms, 0, sizeof(cl_mem), &sec_word);
    clSetKernelArg(k_norms, 1, sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_norms, 2, sizeof(cl_mem), &word_norm_sq);
    /* arg 3 = num_sections — set per test */

    /* build_disjunct_chains: (sec_djh, sec_count, djh_ht_keys, djh_ht_values,
     *                         sec_chain_next, num_sections) */
    clSetKernelArg(k_chains, 0, sizeof(cl_mem), &sec_disjunct_hash);
    clSetKernelArg(k_chains, 1, sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_chains, 2, sizeof(cl_mem), &djh_ht_keys);
    clSetKernelArg(k_chains, 3, sizeof(cl_mem), &djh_ht_values);
    clSetKernelArg(k_chains, 4, sizeof(cl_mem), &sec_chain_next);
    /* arg 5 = num_sections — set per test */

    /* accumulate_dot_products: 14 args */
    clSetKernelArg(k_dots, 0,  sizeof(cl_mem), &sec_word);
    clSetKernelArg(k_dots, 1,  sizeof(cl_mem), &sec_disjunct_hash);
    clSetKernelArg(k_dots, 2,  sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_dots, 3,  sizeof(cl_mem), &djh_ht_keys);
    clSetKernelArg(k_dots, 4,  sizeof(cl_mem), &djh_ht_values);
    clSetKernelArg(k_dots, 5,  sizeof(cl_mem), &sec_chain_next);
    clSetKernelArg(k_dots, 6,  sizeof(cl_mem), &cand_ht_keys);
    clSetKernelArg(k_dots, 7,  sizeof(cl_mem), &cand_ht_values);
    clSetKernelArg(k_dots, 8,  sizeof(cl_mem), &cand_word_a);
    clSetKernelArg(k_dots, 9,  sizeof(cl_mem), &cand_word_b);
    clSetKernelArg(k_dots, 10, sizeof(cl_mem), &cand_dot);
    clSetKernelArg(k_dots, 11, sizeof(cl_mem), &cand_next_free);
    /* arg 12 = num_sections — set per test */

    /* compute_cosines: (cand_word_a, cand_word_b, cand_dot, cand_cosine,
     *                   word_norm_sq, num_candidates) */
    clSetKernelArg(k_cosines, 0, sizeof(cl_mem), &cand_word_a);
    clSetKernelArg(k_cosines, 1, sizeof(cl_mem), &cand_word_b);
    clSetKernelArg(k_cosines, 2, sizeof(cl_mem), &cand_dot);
    clSetKernelArg(k_cosines, 3, sizeof(cl_mem), &cand_cosine);
    clSetKernelArg(k_cosines, 4, sizeof(cl_mem), &word_norm_sq);
    /* arg 5 = num_candidates — set in pipeline */

    /* filter_candidates */
    clSetKernelArg(k_filter, 0, sizeof(cl_mem), &cand_word_a);
    clSetKernelArg(k_filter, 1, sizeof(cl_mem), &cand_word_b);
    clSetKernelArg(k_filter, 2, sizeof(cl_mem), &cand_cosine);
    /* args 3-8 set per test */

    printf("GPU buffers ready\n\n");

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 1: Known cosine (2 words, 4 sections)
     *
     *  Section 0: word=0, djh=0x100, count=3.0
     *  Section 1: word=0, djh=0x200, count=4.0
     *  Section 2: word=1, djh=0x100, count=5.0
     *  Section 3: word=1, djh=0x300, count=2.0
     *
     *  word 0: {0x100: 3, 0x200: 4} → norm² = 25, norm = 5
     *  word 1: {0x100: 5, 0x300: 2} → norm² = 29, norm = √29
     *  Shared: 0x100 → dot = 3×5 = 15
     *  Cosine = 15 / (5 × √29) = 0.5571
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 1: Known cosine (2 words, 4 sections) ---\n");

    reset_cosine_buffers(queue, djh_ht_keys, djh_ht_values,
        sec_chain_next, word_norm_sq, cand_ht_keys, cand_ht_values,
        cand_dot, cand_cosine, cand_next_free);

    {
        uint32_t sw[] = {0, 0, 1, 1};
        uint64_t sd[] = {0x100, 0x200, 0x100, 0x300};
        double   sc[] = {3.0, 4.0, 5.0, 2.0};
        cl_uint  ns = 4;

        upload_sections(queue, sec_word, sec_disjunct_hash, sec_count,
                        sec_next_free, sw, sd, sc, ns);

        /* Set num_sections for each kernel */
        clSetKernelArg(k_norms,  3, sizeof(cl_uint), &ns);
        clSetKernelArg(k_chains, 5, sizeof(cl_uint), &ns);
        clSetKernelArg(k_dots,  12, sizeof(cl_uint), &ns);

        uint32_t num_cands = 0;
        double t0 = now_ms();
        run_cosine_pipeline(queue, k_norms, k_chains, k_dots, k_cosines,
                            ns, cand_next_free, &num_cands);
        double t1 = now_ms();

        /* Read results */
        uint32_t h_wa[4], h_wb[4];
        double h_dot[4], h_cos[4];
        if (num_cands > 0) {
            clEnqueueReadBuffer(queue, cand_word_a, CL_TRUE, 0,
                sizeof(uint32_t) * num_cands, h_wa, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, cand_word_b, CL_TRUE, 0,
                sizeof(uint32_t) * num_cands, h_wb, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, cand_dot, CL_TRUE, 0,
                sizeof(double) * num_cands, h_dot, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, cand_cosine, CL_TRUE, 0,
                sizeof(double) * num_cands, h_cos, 0, NULL, NULL);
        }

        /* Expected: 1 candidate pair (0,1) with dot=15.0, cosine≈0.5571 */
        double expected_cos = 15.0 / (5.0 * sqrt(29.0));

        printf("  Candidates: %u (expected 1)\n", num_cands);
        if (num_cands > 0) {
            printf("  Pair: (%u, %u)  dot=%.1f  cosine=%.4f\n",
                   h_wa[0], h_wb[0], h_dot[0], h_cos[0]);
            printf("  Expected: (0, 1)  dot=15.0  cosine=%.4f\n", expected_cos);
        }
        printf("  Time: %.2f ms\n", t1 - t0);

        int pass = (num_cands == 1) &&
                   (h_wa[0] == 0) && (h_wb[0] == 1) &&
                   (fabs(h_dot[0] - 15.0) < 0.01) &&
                   (fabs(h_cos[0] - expected_cos) < 0.001);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 2: Three words — all pairwise cosines
     *
     *  word 0: {X=0x10: 1, Y=0x20: 2}   norm² = 5
     *  word 1: {X=0x10: 3, Z=0x30: 1}   norm² = 10
     *  word 2: {Y=0x20: 2, Z=0x30: 4}   norm² = 20
     *
     *  dot(0,1) = 1×3 = 3       cos = 3/√50  ≈ 0.4243
     *  dot(0,2) = 2×2 = 4       cos = 4/√100 = 0.4000
     *  dot(1,2) = 1×4 = 4       cos = 4/√200 ≈ 0.2828
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 2: Three words, all pairwise cosines ---\n");

    reset_cosine_buffers(queue, djh_ht_keys, djh_ht_values,
        sec_chain_next, word_norm_sq, cand_ht_keys, cand_ht_values,
        cand_dot, cand_cosine, cand_next_free);

    {
        uint32_t sw[] = {0, 0, 1, 1, 2, 2};
        uint64_t sd[] = {0x10, 0x20, 0x10, 0x30, 0x20, 0x30};
        double   sc[] = {1.0, 2.0, 3.0, 1.0, 2.0, 4.0};
        cl_uint  ns = 6;

        upload_sections(queue, sec_word, sec_disjunct_hash, sec_count,
                        sec_next_free, sw, sd, sc, ns);

        clSetKernelArg(k_norms,  3, sizeof(cl_uint), &ns);
        clSetKernelArg(k_chains, 5, sizeof(cl_uint), &ns);
        clSetKernelArg(k_dots,  12, sizeof(cl_uint), &ns);

        uint32_t num_cands = 0;
        double t0 = now_ms();
        run_cosine_pipeline(queue, k_norms, k_chains, k_dots, k_cosines,
                            ns, cand_next_free, &num_cands);
        double t1 = now_ms();

        uint32_t h_wa[8], h_wb[8];
        double h_dot[8], h_cos[8];
        if (num_cands > 0) {
            clEnqueueReadBuffer(queue, cand_word_a, CL_TRUE, 0,
                sizeof(uint32_t) * num_cands, h_wa, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, cand_word_b, CL_TRUE, 0,
                sizeof(uint32_t) * num_cands, h_wb, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, cand_dot, CL_TRUE, 0,
                sizeof(double) * num_cands, h_dot, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, cand_cosine, CL_TRUE, 0,
                sizeof(double) * num_cands, h_cos, 0, NULL, NULL);
        }

        double exp_01 = 3.0 / sqrt(50.0);
        double exp_02 = 4.0 / sqrt(100.0);
        double exp_12 = 4.0 / sqrt(200.0);

        printf("  Candidates: %u (expected 3)\n", num_cands);

        /* Find each pair in results */
        double got_01 = -1, got_02 = -1, got_12 = -1;
        double got_dot_01 = -1, got_dot_02 = -1, got_dot_12 = -1;
        for (uint32_t i = 0; i < num_cands; i++) {
            if (h_wa[i] == 0 && h_wb[i] == 1)
                { got_01 = h_cos[i]; got_dot_01 = h_dot[i]; }
            if (h_wa[i] == 0 && h_wb[i] == 2)
                { got_02 = h_cos[i]; got_dot_02 = h_dot[i]; }
            if (h_wa[i] == 1 && h_wb[i] == 2)
                { got_12 = h_cos[i]; got_dot_12 = h_dot[i]; }
        }

        printf("  (0,1): dot=%.1f cos=%.4f (exp dot=3.0 cos=%.4f)\n",
               got_dot_01, got_01, exp_01);
        printf("  (0,2): dot=%.1f cos=%.4f (exp dot=4.0 cos=%.4f)\n",
               got_dot_02, got_02, exp_02);
        printf("  (1,2): dot=%.1f cos=%.4f (exp dot=4.0 cos=%.4f)\n",
               got_dot_12, got_12, exp_12);
        printf("  Time: %.2f ms\n", t1 - t0);

        int pass = (num_cands == 3) &&
                   (fabs(got_01 - exp_01) < 0.001) &&
                   (fabs(got_02 - exp_02) < 0.001) &&
                   (fabs(got_12 - exp_12) < 0.001);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 3: Identical vectors → cosine = 1.0
     *
     *  word 0: {X=0x10: 3, Y=0x20: 4}
     *  word 1: {X=0x10: 3, Y=0x20: 4}
     *
     *  dot = 9+16 = 25, norms = 5 each, cosine = 25/25 = 1.0
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 3: Identical vectors → cosine = 1.0 ---\n");

    reset_cosine_buffers(queue, djh_ht_keys, djh_ht_values,
        sec_chain_next, word_norm_sq, cand_ht_keys, cand_ht_values,
        cand_dot, cand_cosine, cand_next_free);

    {
        uint32_t sw[] = {0, 0, 1, 1};
        uint64_t sd[] = {0x10, 0x20, 0x10, 0x20};
        double   sc[] = {3.0, 4.0, 3.0, 4.0};
        cl_uint  ns = 4;

        upload_sections(queue, sec_word, sec_disjunct_hash, sec_count,
                        sec_next_free, sw, sd, sc, ns);

        clSetKernelArg(k_norms,  3, sizeof(cl_uint), &ns);
        clSetKernelArg(k_chains, 5, sizeof(cl_uint), &ns);
        clSetKernelArg(k_dots,  12, sizeof(cl_uint), &ns);

        uint32_t num_cands = 0;
        run_cosine_pipeline(queue, k_norms, k_chains, k_dots, k_cosines,
                            ns, cand_next_free, &num_cands);

        double h_cos = 0;
        if (num_cands > 0) {
            clEnqueueReadBuffer(queue, cand_cosine, CL_TRUE, 0,
                sizeof(double), &h_cos, 0, NULL, NULL);
        }

        printf("  Candidates: %u (expected 1)\n", num_cands);
        printf("  Cosine: %.4f (expected 1.0000)\n", h_cos);

        int pass = (num_cands == 1) && (fabs(h_cos - 1.0) < 0.001);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 4: No shared disjuncts → 0 candidates
     *
     *  word 0: {X=0x10: 1}
     *  word 1: {Y=0x20: 1}
     *
     *  No shared disjuncts → no chain overlap → 0 candidates
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 4: No shared disjuncts → 0 candidates ---\n");

    reset_cosine_buffers(queue, djh_ht_keys, djh_ht_values,
        sec_chain_next, word_norm_sq, cand_ht_keys, cand_ht_values,
        cand_dot, cand_cosine, cand_next_free);

    {
        uint32_t sw[] = {0, 1};
        uint64_t sd[] = {0x10, 0x20};
        double   sc[] = {1.0, 1.0};
        cl_uint  ns = 2;

        upload_sections(queue, sec_word, sec_disjunct_hash, sec_count,
                        sec_next_free, sw, sd, sc, ns);

        clSetKernelArg(k_norms,  3, sizeof(cl_uint), &ns);
        clSetKernelArg(k_chains, 5, sizeof(cl_uint), &ns);
        clSetKernelArg(k_dots,  12, sizeof(cl_uint), &ns);

        uint32_t num_cands = 0;
        run_cosine_pipeline(queue, k_norms, k_chains, k_dots, k_cosines,
                            ns, cand_next_free, &num_cands);

        printf("  Candidates: %u (expected 0)\n", num_cands);

        int pass = (num_cands == 0);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 5: Filter candidates above threshold
     *
     *  Reuse test 2's scenario (3 words):
     *    cos(0,1) ≈ 0.4243
     *    cos(0,2) = 0.4000
     *    cos(1,2) ≈ 0.2828
     *
     *  Filter at 0.35 → should get 2 candidates (0,1) and (0,2)
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 5: Filter candidates above threshold ---\n");

    reset_cosine_buffers(queue, djh_ht_keys, djh_ht_values,
        sec_chain_next, word_norm_sq, cand_ht_keys, cand_ht_values,
        cand_dot, cand_cosine, cand_next_free);

    {
        uint32_t sw[] = {0, 0, 1, 1, 2, 2};
        uint64_t sd[] = {0x10, 0x20, 0x10, 0x30, 0x20, 0x30};
        double   sc[] = {1.0, 2.0, 3.0, 1.0, 2.0, 4.0};
        cl_uint  ns = 6;

        upload_sections(queue, sec_word, sec_disjunct_hash, sec_count,
                        sec_next_free, sw, sd, sc, ns);

        clSetKernelArg(k_norms,  3, sizeof(cl_uint), &ns);
        clSetKernelArg(k_chains, 5, sizeof(cl_uint), &ns);
        clSetKernelArg(k_dots,  12, sizeof(cl_uint), &ns);

        uint32_t num_cands = 0;
        run_cosine_pipeline(queue, k_norms, k_chains, k_dots, k_cosines,
                            ns, cand_next_free, &num_cands);

        /* Now filter */
        cl_double threshold = 0.35;
        cl_uint max_output = 64;
        cl_mem out_wa = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            sizeof(uint32_t) * max_output, NULL, &err);
        cl_mem out_wb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            sizeof(uint32_t) * max_output, NULL, &err);
        cl_mem out_cos = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            sizeof(double) * max_output, NULL, &err);
        cl_mem out_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(uint32_t), &zero, &err);

        clSetKernelArg(k_filter, 3, sizeof(cl_uint), &num_cands);
        clSetKernelArg(k_filter, 4, sizeof(cl_double), &threshold);
        clSetKernelArg(k_filter, 5, sizeof(cl_mem), &out_wa);
        clSetKernelArg(k_filter, 6, sizeof(cl_mem), &out_wb);
        clSetKernelArg(k_filter, 7, sizeof(cl_mem), &out_cos);
        clSetKernelArg(k_filter, 8, sizeof(cl_mem), &out_count);
        clSetKernelArg(k_filter, 9, sizeof(cl_uint), &max_output);

        size_t local = 256;
        size_t gs = ((num_cands + local - 1) / local) * local;
        err = clEnqueueNDRangeKernel(queue, k_filter, 1, NULL,
            &gs, &local, 0, NULL, NULL);
        CL_CHECK(err, "enqueue filter");
        clFinish(queue);

        uint32_t n_filtered;
        clEnqueueReadBuffer(queue, out_count, CL_TRUE, 0,
            sizeof(uint32_t), &n_filtered, 0, NULL, NULL);

        uint32_t fwa[8], fwb[8];
        double fcos[8];
        if (n_filtered > 0) {
            clEnqueueReadBuffer(queue, out_wa, CL_TRUE, 0,
                sizeof(uint32_t) * n_filtered, fwa, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, out_wb, CL_TRUE, 0,
                sizeof(uint32_t) * n_filtered, fwb, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, out_cos, CL_TRUE, 0,
                sizeof(double) * n_filtered, fcos, 0, NULL, NULL);
        }

        printf("  Total candidates: %u, filtered (>0.35): %u (expected 2)\n",
               num_cands, n_filtered);
        for (uint32_t i = 0; i < n_filtered; i++) {
            printf("    (%u, %u) cos=%.4f\n", fwa[i], fwb[i], fcos[i]);
        }

        /* cos(1,2) ≈ 0.2828 should be filtered out */
        int pass = (n_filtered == 2);
        printf("  %s\n\n", pass ? "PASS" : "FAIL");
        if (pass) pass_count++; else fail_count++;

        clReleaseMemObject(out_wa);
        clReleaseMemObject(out_wb);
        clReleaseMemObject(out_cos);
        clReleaseMemObject(out_count);
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 6: Benchmark — 1000 sentences → sections → cosines
     *
     *  Full pipeline:
     *    extract_sections (Phase 4) → cosine pipeline (Phase 5)
     *
     *  1000 sentences, 10-20 words each, chain MST parse.
     *  500 word vocabulary for realistic disjunct sharing.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 6: Benchmark (1000 sentences, full pipeline) ---\n");

    /* Reset everything */
    reset_section_pool(queue, sht_keys, sht_values, sec_count, sec_next_free);
    reset_cosine_buffers(queue, djh_ht_keys, djh_ht_values,
        sec_chain_next, word_norm_sq, cand_ht_keys, cand_ht_values,
        cand_dot, cand_cosine, cand_next_free);
    {
        uint32_t tsec_zero = 0;
        clEnqueueWriteBuffer(queue, total_sections_created, CL_TRUE, 0,
            sizeof(uint32_t), &tsec_zero, 0, NULL, NULL);
    }

    srand(42);
    uint32_t bench_ns = 1000;
    uint32_t vocab_size = 500;

    uint32_t max_words = bench_ns * 25;
    uint32_t max_edges = bench_ns * 25;
    uint32_t* b_words = malloc(sizeof(uint32_t) * max_words);
    uint32_t* b_sent_offsets = malloc(sizeof(uint32_t) * bench_ns);
    uint32_t* b_sent_lengths = malloc(sizeof(uint32_t) * bench_ns);
    uint32_t* b_edge_p1 = malloc(sizeof(uint32_t) * max_edges);
    uint32_t* b_edge_p2 = malloc(sizeof(uint32_t) * max_edges);
    uint32_t* b_edge_offsets = malloc(sizeof(uint32_t) * bench_ns);
    uint32_t* b_edge_counts = malloc(sizeof(uint32_t) * bench_ns);

    uint32_t word_pos = 0, edge_pos = 0;
    for (uint32_t s = 0; s < bench_ns; s++) {
        uint32_t slen = 10 + (rand() % 11);
        b_sent_offsets[s] = word_pos;
        b_sent_lengths[s] = slen;
        b_edge_offsets[s] = edge_pos;
        b_edge_counts[s] = slen - 1;

        for (uint32_t w = 0; w < slen; w++)
            b_words[word_pos++] = rand() % vocab_size;
        for (uint32_t e = 0; e < slen - 1; e++) {
            b_edge_p1[edge_pos] = e;
            b_edge_p2[edge_pos] = e + 1;
            edge_pos++;
        }
    }

    printf("  Sentences: %u, words: %u, edges: %u\n", bench_ns, word_pos, edge_pos);

    cl_mem d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * word_pos, b_words, &err);
    cl_mem d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_sent_offsets, &err);
    cl_mem d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_sent_lengths, &err);
    cl_mem d_edge_p1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * edge_pos, b_edge_p1, &err);
    cl_mem d_edge_p2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * edge_pos, b_edge_p2, &err);
    cl_mem d_edge_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_edge_offsets, &err);
    cl_mem d_edge_counts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_edge_counts, &err);

    /* Set extract_sections args */
    cl_uint tw = word_pos;
    clSetKernelArg(k_extract, 0,  sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_extract, 1,  sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_extract, 2,  sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_extract, 3,  sizeof(cl_uint), &bench_ns);
    clSetKernelArg(k_extract, 4,  sizeof(cl_uint), &tw);
    clSetKernelArg(k_extract, 5,  sizeof(cl_mem), &d_edge_p1);
    clSetKernelArg(k_extract, 6,  sizeof(cl_mem), &d_edge_p2);
    clSetKernelArg(k_extract, 7,  sizeof(cl_mem), &d_edge_offsets);
    clSetKernelArg(k_extract, 8,  sizeof(cl_mem), &d_edge_counts);
    clSetKernelArg(k_extract, 9,  sizeof(cl_mem), &sht_keys);
    clSetKernelArg(k_extract, 10, sizeof(cl_mem), &sht_values);
    clSetKernelArg(k_extract, 11, sizeof(cl_mem), &sec_word);
    clSetKernelArg(k_extract, 12, sizeof(cl_mem), &sec_disjunct_hash);
    clSetKernelArg(k_extract, 13, sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_extract, 14, sizeof(cl_mem), &sec_next_free);
    clSetKernelArg(k_extract, 15, sizeof(cl_mem), &total_sections_created);

    size_t local = 256;
    size_t gs;

    /* Phase 4: Extract sections */
    double t_start = now_ms();

    gs = ((tw + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local, 0, NULL, NULL);
    CL_CHECK(err, "enqueue extract");
    clFinish(queue);

    double t_extract = now_ms();

    /* Read section count */
    uint32_t h_num_sections;
    clEnqueueReadBuffer(queue, sec_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_sections, 0, NULL, NULL);

    printf("  Sections extracted: %u (%.2f ms)\n",
           h_num_sections, t_extract - t_start);

    /* Phase 5: Cosine pipeline */
    clSetKernelArg(k_norms,  3, sizeof(cl_uint), &h_num_sections);
    clSetKernelArg(k_chains, 5, sizeof(cl_uint), &h_num_sections);
    clSetKernelArg(k_dots,  12, sizeof(cl_uint), &h_num_sections);

    double t_cos_start = now_ms();

    uint32_t num_cands = 0;
    run_cosine_pipeline(queue, k_norms, k_chains, k_dots, k_cosines,
                        h_num_sections, cand_next_free, &num_cands);

    double t_cos_end = now_ms();

    double total_time = t_cos_end - t_start;
    double cos_time = t_cos_end - t_cos_start;

    printf("  Candidate pairs: %u\n", num_cands);
    printf("  Cosine pipeline: %.2f ms\n", cos_time);
    printf("  Full pipeline (extract + cosine): %.2f ms\n", total_time);
    printf("  Throughput: %.0f sentences/sec\n",
           bench_ns / (total_time / 1000.0));

    if (num_cands > 0) {
        /* Read a few top cosines to verify */
        uint32_t peek = (num_cands < 8) ? num_cands : 8;
        double h_cos[8];
        clEnqueueReadBuffer(queue, cand_cosine, CL_TRUE, 0,
            sizeof(double) * peek, h_cos, 0, NULL, NULL);

        /* Find max cosine */
        double max_cos = 0;
        for (uint32_t i = 0; i < peek; i++)
            if (h_cos[i] > max_cos) max_cos = h_cos[i];
        printf("  Max cosine (first %u): %.4f\n", peek, max_cos);
    }

    int t6_pass = (h_num_sections > 0) && (num_cands > 0) && (total_time < 5000.0);
    printf("  %s\n\n", t6_pass ? "PASS" : "FAIL");
    if (t6_pass) pass_count++; else fail_count++;

    /* ─── Summary ─── */

    printf("=== Results: %d PASS, %d FAIL ===\n", pass_count, fail_count);

    /* Cleanup */
    free(b_words); free(b_sent_offsets); free(b_sent_lengths);
    free(b_edge_p1); free(b_edge_p2); free(b_edge_offsets); free(b_edge_counts);
    free(src_ht); free(src_as); free(src_sc); free(src_cos); free(combined);

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);
    clReleaseMemObject(d_edge_p1);
    clReleaseMemObject(d_edge_p2);
    clReleaseMemObject(d_edge_offsets);
    clReleaseMemObject(d_edge_counts);
    clReleaseMemObject(sec_word);
    clReleaseMemObject(sec_disjunct_hash);
    clReleaseMemObject(sec_count);
    clReleaseMemObject(sec_next_free);
    clReleaseMemObject(sht_keys);
    clReleaseMemObject(sht_values);
    clReleaseMemObject(total_sections_created);
    clReleaseMemObject(djh_ht_keys);
    clReleaseMemObject(djh_ht_values);
    clReleaseMemObject(sec_chain_next);
    clReleaseMemObject(word_norm_sq);
    clReleaseMemObject(cand_ht_keys);
    clReleaseMemObject(cand_ht_values);
    clReleaseMemObject(cand_word_a);
    clReleaseMemObject(cand_word_b);
    clReleaseMemObject(cand_dot);
    clReleaseMemObject(cand_cosine);
    clReleaseMemObject(cand_next_free);
    clReleaseKernel(k_norms);
    clReleaseKernel(k_chains);
    clReleaseKernel(k_dots);
    clReleaseKernel(k_cosines);
    clReleaseKernel(k_filter);
    clReleaseKernel(k_extract);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return fail_count > 0 ? 1 : 0;
}
