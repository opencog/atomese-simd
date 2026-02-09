/*
 * test-sections.c -- Test GPU section extraction kernel
 *
 * Compile: gcc -O2 -o test-sections test-sections.c -lOpenCL -lm
 * Run:     ./test-sections
 *
 * Tests:
 *   1. Simple MST: 3-word sentence with 2 edges → 3 sections
 *   2. Star parse: all edges from one root → verify disjuncts
 *   3. Multi-sentence batch (no cross-boundary sections)
 *   4. Duplicate disjuncts: same parse seen twice → counts accumulate
 *   5. Readback kernel verification
 *   6. Benchmark: 1000 sentences with random MST edges
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

/* ─── CPU-side disjunct hash (must match GPU) ─── */

static uint64_t cpu_fnv1a_init(void) { return 0xcbf29ce484222325ULL; }

static uint64_t cpu_fnv1a_mix(uint64_t hash, uint64_t val)
{
    hash ^= val;
    hash *= 0x100000001b3ULL;
    return hash;
}

static uint64_t cpu_hash_disjunct(uint32_t* words, uint32_t* dirs, uint32_t count)
{
    uint64_t h = cpu_fnv1a_init();
    for (uint32_t i = 0; i < count; i++) {
        uint64_t encoded = ((uint64_t)words[i] << 1) | (uint64_t)dirs[i];
        h = cpu_fnv1a_mix(h, encoded);
    }
    if (h == HT_EMPTY_KEY) h = 0;
    return h;
}

/* CPU-side section_key (must match GPU) */
static uint64_t cpu_section_key(uint32_t word_idx, uint64_t disjunct_hash)
{
    uint64_t key = disjunct_hash ^ ((uint64_t)word_idx * 0x9E3779B97F4A7C15ULL);
    if (key == HT_EMPTY_KEY) key = 0;
    return key;
}

/* ─── Helper: reset section pool and hash table ─── */

void reset_section_pool(cl_command_queue queue,
                        cl_mem sht_keys, cl_mem sht_values,
                        cl_mem sec_count,
                        cl_mem sec_next_free, cl_mem total_sections)
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
    clEnqueueWriteBuffer(queue, total_sections, CL_FALSE, 0,
        sizeof(uint32_t), &zero, 0, NULL, NULL);
    clFinish(queue);
}

/* ─── Main ─── */

int main(int argc, char** argv)
{
    cl_int err;
    int pass_count = 0, fail_count = 0;

    printf("=== GPU Section Extraction Test ===\n\n");

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

    size_t ht_len, as_len, sc_len;
    char* ht_src = read_file("gpu-hashtable.cl", &ht_len);
    char* as_src = read_file("gpu-atomspace.cl", &as_len);
    char* sc_src = read_file("gpu-sections.cl", &sc_len);

    size_t total_len = ht_len + 1 + as_len + 1 + sc_len;
    char* combined = malloc(total_len + 1);
    memcpy(combined, ht_src, ht_len);
    combined[ht_len] = '\n';
    memcpy(combined + ht_len + 1, as_src, as_len);
    combined[ht_len + 1 + as_len] = '\n';
    memcpy(combined + ht_len + 1 + as_len + 1, sc_src, sc_len);
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

    cl_kernel k_extract = clCreateKernel(program, "extract_sections", &err);
    CL_CHECK(err, "kernel extract_sections");
    cl_kernel k_read = clCreateKernel(program, "read_sections", &err);
    CL_CHECK(err, "kernel read_sections");

    size_t local_size = 256;

    /* ─── Allocate GPU buffers ─── */

    printf("Allocating GPU buffers...\n");
    uint32_t zero = 0;

    /* Section hash table */
    cl_mem sht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * SECTION_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "sht_keys");
    cl_mem sht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "sht_values");

    /* Section pool SoA */
    cl_mem sec_word = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_CAPACITY, NULL, &err);
    cl_mem sec_disjunct_hash = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * SECTION_CAPACITY, NULL, &err);
    cl_mem sec_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * SECTION_CAPACITY, NULL, &err);

    /* Section bump allocator */
    cl_mem sec_next_free = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Total sections created (stats counter) */
    cl_mem total_sections = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Initial reset */
    reset_section_pool(queue, sht_keys, sht_values, sec_count,
                       sec_next_free, total_sections);

    printf("GPU buffers ready\n\n");

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 1: Simple MST — 3-word sentence with chain parse
     *
     *  Sentence: words [10, 20, 30] (word pool indices)
     *  MST edges: (0,1), (1,2)  — chain: 10—20—30
     *
     *  Expected sections:
     *    Word 10 (pos 0): connectors = [(20, RIGHT)]
     *      disjunct = "20+"
     *    Word 20 (pos 1): connectors = [(10, LEFT), (30, RIGHT)]
     *      disjunct = "10- 30+"
     *    Word 30 (pos 2): connectors = [(20, LEFT)]
     *      disjunct = "20-"
     *
     *  = 3 unique sections
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 1: Simple chain parse (3 words, 2 edges) ---\n");

    uint32_t t1_words[] = {10, 20, 30};
    uint32_t t1_sent_offsets[] = {0};
    uint32_t t1_sent_lengths[] = {3};
    uint32_t t1_edge_p1[] = {0, 1};
    uint32_t t1_edge_p2[] = {1, 2};
    uint32_t t1_edge_offsets[] = {0};
    uint32_t t1_edge_counts[] = {2};
    cl_uint t1_ns = 1, t1_tw = 3;

    cl_mem d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t1_tw, t1_words, &err);
    cl_mem d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t1_ns, t1_sent_offsets, &err);
    cl_mem d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t1_ns, t1_sent_lengths, &err);
    cl_mem d_edge_p1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 2, t1_edge_p1, &err);
    cl_mem d_edge_p2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 2, t1_edge_p2, &err);
    cl_mem d_edge_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t1_ns, t1_edge_offsets, &err);
    cl_mem d_edge_counts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t1_ns, t1_edge_counts, &err);

    /* Set kernel args */
    clSetKernelArg(k_extract, 0,  sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_extract, 1,  sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_extract, 2,  sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_extract, 3,  sizeof(cl_uint), &t1_ns);
    clSetKernelArg(k_extract, 4,  sizeof(cl_uint), &t1_tw);
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
    clSetKernelArg(k_extract, 15, sizeof(cl_mem), &total_sections);

    double t0 = now_ms();
    size_t gs = ((t1_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue extract");
    clFinish(queue);
    double t1 = now_ms();

    /* Read back results */
    uint32_t h_num_sections, h_total_created;
    clEnqueueReadBuffer(queue, sec_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_sections, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_sections, CL_TRUE, 0,
        sizeof(uint32_t), &h_total_created, 0, NULL, NULL);

    /* Read section data */
    uint32_t h_sec_words[8];
    uint64_t h_sec_djh[8];
    double   h_sec_counts[8];
    clEnqueueReadBuffer(queue, sec_word, CL_TRUE, 0,
        sizeof(uint32_t) * h_num_sections, h_sec_words, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sec_disjunct_hash, CL_TRUE, 0,
        sizeof(uint64_t) * h_num_sections, h_sec_djh, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sec_count, CL_TRUE, 0,
        sizeof(double) * h_num_sections, h_sec_counts, 0, NULL, NULL);

    /* Compute expected disjunct hashes on CPU */
    /* Word 10 (pos 0): [(20, RIGHT=1)] */
    uint32_t cw0[] = {20}; uint32_t cd0[] = {1};
    uint64_t exp_djh_0 = cpu_hash_disjunct(cw0, cd0, 1);
    /* Word 20 (pos 1): [(10, LEFT=0), (30, RIGHT=1)] — already sorted */
    uint32_t cw1[] = {10, 30}; uint32_t cd1[] = {0, 1};
    uint64_t exp_djh_1 = cpu_hash_disjunct(cw1, cd1, 2);
    /* Word 30 (pos 2): [(20, LEFT=0)] */
    uint32_t cw2[] = {20}; uint32_t cd2[] = {0};
    uint64_t exp_djh_2 = cpu_hash_disjunct(cw2, cd2, 1);

    printf("  Sections created: %u (expected 3)\n", h_num_sections);
    printf("  Stats counter:    %u (expected 3)\n", h_total_created);

    /* Verify each section exists with correct data */
    int found_10 = 0, found_20 = 0, found_30 = 0;
    for (uint32_t i = 0; i < h_num_sections; i++) {
        if (h_sec_words[i] == 10 && h_sec_djh[i] == exp_djh_0 &&
            fabs(h_sec_counts[i] - 1.0) < 0.01) found_10 = 1;
        if (h_sec_words[i] == 20 && h_sec_djh[i] == exp_djh_1 &&
            fabs(h_sec_counts[i] - 1.0) < 0.01) found_20 = 1;
        if (h_sec_words[i] == 30 && h_sec_djh[i] == exp_djh_2 &&
            fabs(h_sec_counts[i] - 1.0) < 0.01) found_30 = 1;
    }

    printf("  Section (word=10, djh=0x%016llx): %s\n",
           (unsigned long long)exp_djh_0, found_10 ? "found" : "MISSING");
    printf("  Section (word=20, djh=0x%016llx): %s\n",
           (unsigned long long)exp_djh_1, found_20 ? "found" : "MISSING");
    printf("  Section (word=30, djh=0x%016llx): %s\n",
           (unsigned long long)exp_djh_2, found_30 ? "found" : "MISSING");
    printf("  Time: %.2f ms\n", t1 - t0);

    int t1_pass = (h_num_sections == 3) && (h_total_created == 3) &&
                  found_10 && found_20 && found_30;
    printf("  %s\n\n", t1_pass ? "PASS" : "FAIL");
    if (t1_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 2: Star parse — root word connected to all others
     *
     *  Sentence: words [100, 101, 102, 103, 104] (5 words)
     *  MST edges: (2,0), (2,1), (2,3), (2,4) — word 102 is root
     *
     *  Expected sections:
     *    Word 100 (pos 0): [(102, RIGHT)]   — 1 connector
     *    Word 101 (pos 1): [(102, RIGHT)]   — 1 connector
     *    Word 102 (pos 2): [(100, LEFT), (101, LEFT), (103, RIGHT), (104, RIGHT)]
     *    Word 103 (pos 3): [(102, LEFT)]    — 1 connector
     *    Word 104 (pos 4): [(102, LEFT)]    — 1 connector
     *
     *  = 5 sections (4 unique disjuncts: leaf-left, leaf-right, root-4conn)
     *  But words 100 and 101 have same disjunct hash ONLY if their
     *  partner word pool index is the same (both connect to 102) AND
     *  direction is the same (both RIGHT). So disjunct hash matches!
     *  → 100 and 101 have same disjunct but different words → 2 sections
     *  Similarly 103 and 104 connect LEFT to 102 → same disjunct → 2 sections
     *
     *  Total unique sections (word, disjunct) pairs: 5
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 2: Star parse (5 words, root at center) ---\n");

    reset_section_pool(queue, sht_keys, sht_values, sec_count,
                       sec_next_free, total_sections);

    uint32_t t2_words[] = {100, 101, 102, 103, 104};
    uint32_t t2_sent_offsets[] = {0};
    uint32_t t2_sent_lengths[] = {5};
    uint32_t t2_edge_p1[] = {2, 2, 2, 2};
    uint32_t t2_edge_p2[] = {0, 1, 3, 4};
    uint32_t t2_edge_offsets[] = {0};
    uint32_t t2_edge_counts[] = {4};
    cl_uint t2_ns = 1, t2_tw = 5;

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);
    clReleaseMemObject(d_edge_p1);
    clReleaseMemObject(d_edge_p2);
    clReleaseMemObject(d_edge_offsets);
    clReleaseMemObject(d_edge_counts);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t2_tw, t2_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t2_ns, t2_sent_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t2_ns, t2_sent_lengths, &err);
    d_edge_p1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 4, t2_edge_p1, &err);
    d_edge_p2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 4, t2_edge_p2, &err);
    d_edge_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t2_ns, t2_edge_offsets, &err);
    d_edge_counts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t2_ns, t2_edge_counts, &err);

    /* Set kernel args */
    clSetKernelArg(k_extract, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_extract, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_extract, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_extract, 3, sizeof(cl_uint), &t2_ns);
    clSetKernelArg(k_extract, 4, sizeof(cl_uint), &t2_tw);
    clSetKernelArg(k_extract, 5, sizeof(cl_mem), &d_edge_p1);
    clSetKernelArg(k_extract, 6, sizeof(cl_mem), &d_edge_p2);
    clSetKernelArg(k_extract, 7, sizeof(cl_mem), &d_edge_offsets);
    clSetKernelArg(k_extract, 8, sizeof(cl_mem), &d_edge_counts);

    t0 = now_ms();
    gs = ((t2_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue extract t2");
    clFinish(queue);
    t1 = now_ms();

    clEnqueueReadBuffer(queue, sec_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_sections, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, total_sections, CL_TRUE, 0,
        sizeof(uint32_t), &h_total_created, 0, NULL, NULL);

    /* Compute expected disjunct for root word 102 (pos 2):
     * Connectors: (100, LEFT=0), (101, LEFT=0), (103, RIGHT=1), (104, RIGHT=1)
     * Sorted: dir 0 first sorted by word → (100,0), (101,0), (103,1), (104,1) */
    uint32_t root_cw[] = {100, 101, 103, 104};
    uint32_t root_cd[] = {0, 0, 1, 1};
    uint64_t exp_root_djh = cpu_hash_disjunct(root_cw, root_cd, 4);

    /* Read back all sections */
    uint32_t h2_sec_words[8];
    uint64_t h2_sec_djh[8];
    double   h2_sec_counts[8];
    uint32_t n_read = (h_num_sections < 8) ? h_num_sections : 8;
    clEnqueueReadBuffer(queue, sec_word, CL_TRUE, 0,
        sizeof(uint32_t) * n_read, h2_sec_words, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sec_disjunct_hash, CL_TRUE, 0,
        sizeof(uint64_t) * n_read, h2_sec_djh, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sec_count, CL_TRUE, 0,
        sizeof(double) * n_read, h2_sec_counts, 0, NULL, NULL);

    printf("  Sections created: %u (expected 5)\n", h_num_sections);

    /* Find root section */
    int found_root = 0;
    for (uint32_t i = 0; i < n_read; i++) {
        if (h2_sec_words[i] == 102 && h2_sec_djh[i] == exp_root_djh &&
            fabs(h2_sec_counts[i] - 1.0) < 0.01) {
            found_root = 1;
        }
    }
    printf("  Root section (word=102, 4 connectors): %s\n",
           found_root ? "found" : "MISSING");
    printf("  Time: %.2f ms\n", t1 - t0);

    int t2_pass = (h_num_sections == 5) && found_root;
    printf("  %s\n\n", t2_pass ? "PASS" : "FAIL");
    if (t2_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 3: Multi-sentence batch (no cross-boundary sections)
     *
     *  Sentence 1: [10, 20, 30]   edges: (0,1), (1,2) — chain
     *  Sentence 2: [40, 50, 60]   edges: (0,1), (0,2) — star from 40
     *
     *  flat_words  = [10, 20, 30, 40, 50, 60]
     *  flat_edges  = [(0,1), (1,2), (0,1), (0,2)]
     *  edge_offsets = [0, 2]
     *  edge_counts  = [2, 2]
     *
     *  Expected: 6 sections (3 per sentence), none spanning boundary
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 3: Multi-sentence batch ---\n");

    reset_section_pool(queue, sht_keys, sht_values, sec_count,
                       sec_next_free, total_sections);

    uint32_t t3_words[] = {10, 20, 30, 40, 50, 60};
    uint32_t t3_sent_offsets[] = {0, 3};
    uint32_t t3_sent_lengths[] = {3, 3};
    uint32_t t3_edge_p1[] = {0, 1, 0, 0};
    uint32_t t3_edge_p2[] = {1, 2, 1, 2};
    uint32_t t3_edge_offsets[] = {0, 2};
    uint32_t t3_edge_counts[] = {2, 2};
    cl_uint t3_ns = 2, t3_tw = 6;

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);
    clReleaseMemObject(d_edge_p1);
    clReleaseMemObject(d_edge_p2);
    clReleaseMemObject(d_edge_offsets);
    clReleaseMemObject(d_edge_counts);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t3_tw, t3_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t3_ns, t3_sent_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t3_ns, t3_sent_lengths, &err);
    d_edge_p1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 4, t3_edge_p1, &err);
    d_edge_p2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 4, t3_edge_p2, &err);
    d_edge_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t3_ns, t3_edge_offsets, &err);
    d_edge_counts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t3_ns, t3_edge_counts, &err);

    clSetKernelArg(k_extract, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_extract, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_extract, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_extract, 3, sizeof(cl_uint), &t3_ns);
    clSetKernelArg(k_extract, 4, sizeof(cl_uint), &t3_tw);
    clSetKernelArg(k_extract, 5, sizeof(cl_mem), &d_edge_p1);
    clSetKernelArg(k_extract, 6, sizeof(cl_mem), &d_edge_p2);
    clSetKernelArg(k_extract, 7, sizeof(cl_mem), &d_edge_offsets);
    clSetKernelArg(k_extract, 8, sizeof(cl_mem), &d_edge_counts);

    t0 = now_ms();
    gs = ((t3_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue extract t3");
    clFinish(queue);
    t1 = now_ms();

    clEnqueueReadBuffer(queue, sec_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_sections, 0, NULL, NULL);

    /* Verify: sentence 1 chain gives 3 sections, sentence 2 star gives 3.
     * But sentence 1's word 10 has disjunct "20+" and sentence 2's word 50
     * has disjunct "40-" — different disjuncts. All 6 words produce sections.
     * Are any (word, disjunct) pairs the same? No — different words, different
     * disjuncts. So 6 unique sections. */
    printf("  Sections created: %u (expected 6)\n", h_num_sections);
    printf("  Time: %.2f ms\n", t1 - t0);

    int t3_pass = (h_num_sections == 6);
    printf("  %s\n\n", t3_pass ? "PASS" : "FAIL");
    if (t3_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 4: Duplicate sections — same parse seen twice
     *
     *  Process the SAME sentence twice → section counts should be 2.0
     *
     *  Sentence: [10, 20, 30]  edges: (0,1), (1,2)
     *  Run extract_sections TWICE without resetting.
     *
     *  Expected: 3 sections, each with count = 2.0
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 4: Duplicate sections (same parse twice) ---\n");

    reset_section_pool(queue, sht_keys, sht_values, sec_count,
                       sec_next_free, total_sections);

    uint32_t t4_words[] = {10, 20, 30};
    uint32_t t4_sent_offsets[] = {0};
    uint32_t t4_sent_lengths[] = {3};
    uint32_t t4_edge_p1[] = {0, 1};
    uint32_t t4_edge_p2[] = {1, 2};
    uint32_t t4_edge_offsets[] = {0};
    uint32_t t4_edge_counts[] = {2};
    cl_uint t4_ns = 1, t4_tw = 3;

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);
    clReleaseMemObject(d_edge_p1);
    clReleaseMemObject(d_edge_p2);
    clReleaseMemObject(d_edge_offsets);
    clReleaseMemObject(d_edge_counts);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t4_tw, t4_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t4_ns, t4_sent_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t4_ns, t4_sent_lengths, &err);
    d_edge_p1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 2, t4_edge_p1, &err);
    d_edge_p2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * 2, t4_edge_p2, &err);
    d_edge_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t4_ns, t4_edge_offsets, &err);
    d_edge_counts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * t4_ns, t4_edge_counts, &err);

    clSetKernelArg(k_extract, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_extract, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_extract, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_extract, 3, sizeof(cl_uint), &t4_ns);
    clSetKernelArg(k_extract, 4, sizeof(cl_uint), &t4_tw);
    clSetKernelArg(k_extract, 5, sizeof(cl_mem), &d_edge_p1);
    clSetKernelArg(k_extract, 6, sizeof(cl_mem), &d_edge_p2);
    clSetKernelArg(k_extract, 7, sizeof(cl_mem), &d_edge_offsets);
    clSetKernelArg(k_extract, 8, sizeof(cl_mem), &d_edge_counts);

    /* Run TWICE */
    gs = ((t4_tw + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue extract t4a");
    clFinish(queue);

    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue extract t4b");
    clFinish(queue);

    clEnqueueReadBuffer(queue, sec_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_sections, 0, NULL, NULL);

    /* Read counts */
    double h4_counts[4];
    clEnqueueReadBuffer(queue, sec_count, CL_TRUE, 0,
        sizeof(double) * h_num_sections, h4_counts, 0, NULL, NULL);

    printf("  Sections created: %u (expected 3 — dedup works)\n", h_num_sections);

    int all_count_2 = 1;
    for (uint32_t i = 0; i < h_num_sections; i++) {
        printf("  Section %u count: %.1f (expected 2.0)\n", i, h4_counts[i]);
        if (fabs(h4_counts[i] - 2.0) > 0.01) all_count_2 = 0;
    }

    int t4_pass = (h_num_sections == 3) && all_count_2;
    printf("  %s\n\n", t4_pass ? "PASS" : "FAIL");
    if (t4_pass) pass_count++; else fail_count++;

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 5: Readback kernel
     *
     *  Use read_sections to verify section pool data matches
     *  what extract_sections stored. (Reuse state from test 4.)
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 5: Readback kernel ---\n");

    uint32_t n_secs = h_num_sections;
    cl_mem d_out_word = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * n_secs, NULL, &err);
    cl_mem d_out_djh = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * n_secs, NULL, &err);
    cl_mem d_out_count = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(double) * n_secs, NULL, &err);

    clSetKernelArg(k_read, 0, sizeof(cl_mem), &sec_word);
    clSetKernelArg(k_read, 1, sizeof(cl_mem), &sec_disjunct_hash);
    clSetKernelArg(k_read, 2, sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_read, 3, sizeof(cl_mem), &d_out_word);
    clSetKernelArg(k_read, 4, sizeof(cl_mem), &d_out_djh);
    clSetKernelArg(k_read, 5, sizeof(cl_mem), &d_out_count);
    clSetKernelArg(k_read, 6, sizeof(cl_uint), &n_secs);

    gs = ((n_secs + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_read, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue read_sections");
    clFinish(queue);

    uint32_t rb_words[4];
    uint64_t rb_djh[4];
    double   rb_counts[4];
    clEnqueueReadBuffer(queue, d_out_word, CL_TRUE, 0,
        sizeof(uint32_t) * n_secs, rb_words, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_djh, CL_TRUE, 0,
        sizeof(uint64_t) * n_secs, rb_djh, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_out_count, CL_TRUE, 0,
        sizeof(double) * n_secs, rb_counts, 0, NULL, NULL);

    /* Should match test 4's data */
    uint32_t h5_words[4];
    uint64_t h5_djh[4];
    double   h5_counts[4];
    clEnqueueReadBuffer(queue, sec_word, CL_TRUE, 0,
        sizeof(uint32_t) * n_secs, h5_words, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sec_disjunct_hash, CL_TRUE, 0,
        sizeof(uint64_t) * n_secs, h5_djh, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sec_count, CL_TRUE, 0,
        sizeof(double) * n_secs, h5_counts, 0, NULL, NULL);

    int readback_match = 1;
    for (uint32_t i = 0; i < n_secs; i++) {
        if (rb_words[i] != h5_words[i] || rb_djh[i] != h5_djh[i] ||
            fabs(rb_counts[i] - h5_counts[i]) > 0.01)
            readback_match = 0;
    }
    printf("  Readback matches direct read: %s\n", readback_match ? "yes" : "NO");

    int t5_pass = readback_match;
    printf("  %s\n\n", t5_pass ? "PASS" : "FAIL");
    if (t5_pass) pass_count++; else fail_count++;

    clReleaseMemObject(d_out_word);
    clReleaseMemObject(d_out_djh);
    clReleaseMemObject(d_out_count);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 6: Benchmark — 1000 sentences with random MST edges
     *
     *  Each sentence: 10-20 words, 9-19 MST edges (chain parse)
     *  Total: ~15000 words, ~14000 edges
     *
     *  Measures extract_sections throughput.
     * ═══════════════════════════════════════════════════════════════ */

    printf("--- Test 6: Benchmark (1000 sentences) ---\n");

    reset_section_pool(queue, sht_keys, sht_values, sec_count,
                       sec_next_free, total_sections);

    srand(42);
    uint32_t bench_ns = 1000;
    uint32_t vocab_size = 500;  /* word pool indices 0..499 */

    /* Generate sentences */
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
        uint32_t slen = 10 + (rand() % 11);  /* 10-20 words */
        b_sent_offsets[s] = word_pos;
        b_sent_lengths[s] = slen;
        b_edge_offsets[s] = edge_pos;
        b_edge_counts[s] = slen - 1;  /* chain parse */

        for (uint32_t w = 0; w < slen; w++) {
            b_words[word_pos++] = rand() % vocab_size;
        }
        /* Chain parse: pos 0-1, 1-2, ..., (slen-2)-(slen-1) */
        for (uint32_t e = 0; e < slen - 1; e++) {
            b_edge_p1[edge_pos] = e;
            b_edge_p2[edge_pos] = e + 1;
            edge_pos++;
        }
    }

    printf("  Total words: %u\n", word_pos);
    printf("  Total edges: %u\n", edge_pos);

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);
    clReleaseMemObject(d_edge_p1);
    clReleaseMemObject(d_edge_p2);
    clReleaseMemObject(d_edge_offsets);
    clReleaseMemObject(d_edge_counts);

    d_flat_words = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * word_pos, b_words, &err);
    d_sent_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_sent_offsets, &err);
    d_sent_lengths = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_sent_lengths, &err);
    d_edge_p1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * edge_pos, b_edge_p1, &err);
    d_edge_p2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * edge_pos, b_edge_p2, &err);
    d_edge_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_edge_offsets, &err);
    d_edge_counts = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bench_ns, b_edge_counts, &err);

    cl_uint tw_bench = word_pos;
    clSetKernelArg(k_extract, 0, sizeof(cl_mem), &d_flat_words);
    clSetKernelArg(k_extract, 1, sizeof(cl_mem), &d_sent_offsets);
    clSetKernelArg(k_extract, 2, sizeof(cl_mem), &d_sent_lengths);
    clSetKernelArg(k_extract, 3, sizeof(cl_uint), &bench_ns);
    clSetKernelArg(k_extract, 4, sizeof(cl_uint), &tw_bench);
    clSetKernelArg(k_extract, 5, sizeof(cl_mem), &d_edge_p1);
    clSetKernelArg(k_extract, 6, sizeof(cl_mem), &d_edge_p2);
    clSetKernelArg(k_extract, 7, sizeof(cl_mem), &d_edge_offsets);
    clSetKernelArg(k_extract, 8, sizeof(cl_mem), &d_edge_counts);

    /* Warm up */
    gs = ((tw_bench + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);

    /* Reset for actual benchmark */
    reset_section_pool(queue, sht_keys, sht_values, sec_count,
                       sec_next_free, total_sections);

    t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_extract, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue benchmark");
    clFinish(queue);
    t1 = now_ms();

    clEnqueueReadBuffer(queue, sec_next_free, CL_TRUE, 0,
        sizeof(uint32_t), &h_num_sections, 0, NULL, NULL);

    double elapsed = t1 - t0;
    double secs_per_sec = bench_ns / (elapsed / 1000.0);
    double words_per_sec = word_pos / (elapsed / 1000.0);

    printf("  Sections created: %u\n", h_num_sections);
    printf("  Time: %.2f ms\n", elapsed);
    printf("  Throughput: %.0f sentences/sec, %.0f words/sec\n",
           secs_per_sec, words_per_sec);
    printf("  Throughput: %.1fM sections/sec\n",
           h_num_sections / (elapsed / 1000.0) / 1e6);

    /* Sanity: every word should produce a section in a chain parse,
     * but some (word, disjunct) pairs may collide. So sections < words
     * but > 0. */
    int t6_pass = (h_num_sections > 0) && (h_num_sections <= word_pos) &&
                  (elapsed < 1000.0);
    printf("  %s\n\n", t6_pass ? "PASS" : "FAIL");
    if (t6_pass) pass_count++; else fail_count++;

    /* ─── Summary ─── */

    printf("=== Results: %d PASS, %d FAIL ===\n", pass_count, fail_count);

    /* Cleanup */
    free(b_words); free(b_sent_offsets); free(b_sent_lengths);
    free(b_edge_p1); free(b_edge_p2); free(b_edge_offsets); free(b_edge_counts);
    free(ht_src); free(as_src); free(sc_src); free(combined);

    clReleaseMemObject(d_flat_words);
    clReleaseMemObject(d_sent_offsets);
    clReleaseMemObject(d_sent_lengths);
    clReleaseMemObject(d_edge_p1);
    clReleaseMemObject(d_edge_p2);
    clReleaseMemObject(d_edge_offsets);
    clReleaseMemObject(d_edge_counts);
    clReleaseMemObject(sht_keys);
    clReleaseMemObject(sht_values);
    clReleaseMemObject(sec_word);
    clReleaseMemObject(sec_disjunct_hash);
    clReleaseMemObject(sec_count);
    clReleaseMemObject(sec_next_free);
    clReleaseMemObject(total_sections);
    clReleaseKernel(k_extract);
    clReleaseKernel(k_read);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return fail_count > 0 ? 1 : 0;
}
