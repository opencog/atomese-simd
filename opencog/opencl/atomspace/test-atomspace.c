/*
 * test-atomspace.c -- Test GPU-resident AtomSpace pools
 *
 * Compile: gcc -O2 -o test-atomspace test-atomspace.c -lOpenCL -lm
 * Run:     ./test-atomspace
 *
 * Tests:
 *   1. Create words (find-or-create with dedup)
 *   2. Create pairs (find-or-create from word indices)
 *   3. Create sections
 *   4. Count pairs (atomic double increment + marginals)
 *   5. Count sections
 *   6. Verify counts read back correctly
 *   7. Performance: create + count rates
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

/* ─── Pool capacities ─── */

#define WORD_CAPACITY        (128 * 1024)    /* 128K words */
#define PAIR_CAPACITY        (4 * 1024 * 1024) /* 4M pairs */
#define SECTION_CAPACITY     (1024 * 1024)   /* 1M sections */

/* Hash table sizes (2x pool capacity for 50% load) */
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

/* ─── Simple hash for word names (CPU-side) ─── */

uint64_t hash_word(const char* name)
{
    uint64_t h = 0x12345678DEADBEEFULL;
    while (*name) {
        h ^= (uint64_t)*name++;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 31;
    }
    return h;
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

    printf("=== GPU AtomSpace Pool Test ===\n\n");

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

    size_t ht_len, as_len;
    char* ht_src = read_file("gpu-hashtable.cl", &ht_len);
    char* as_src = read_file("gpu-atomspace.cl", &as_len);

    /* Concatenate: hashtable first, then atomspace */
    size_t total_len = ht_len + 1 + as_len;
    char* combined = malloc(total_len + 1);
    memcpy(combined, ht_src, ht_len);
    combined[ht_len] = '\n';
    memcpy(combined + ht_len + 1, as_src, as_len);
    combined[total_len] = '\0';

    cl_program program = clCreateProgramWithSource(ctx, 1,
        (const char**)&combined, &total_len, &err);
    CL_CHECK(err, "create program");

    /* Build with capacity defines */
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

    cl_kernel k_word_foc = clCreateKernel(program, "word_find_or_create", &err);
    CL_CHECK(err, "kernel word_find_or_create");
    cl_kernel k_pair_foc = clCreateKernel(program, "pair_find_or_create", &err);
    CL_CHECK(err, "kernel pair_find_or_create");
    cl_kernel k_sec_foc = clCreateKernel(program, "section_find_or_create", &err);
    CL_CHECK(err, "kernel section_find_or_create");
    cl_kernel k_count_pairs = clCreateKernel(program, "count_pairs", &err);
    CL_CHECK(err, "kernel count_pairs");
    cl_kernel k_count_sec = clCreateKernel(program, "count_sections", &err);
    CL_CHECK(err, "kernel count_sections");
    cl_kernel k_stats = clCreateKernel(program, "pool_stats", &err);
    CL_CHECK(err, "kernel pool_stats");

    size_t local_size = 256;

    /* ═══ ALLOCATE GPU BUFFERS ═══ */

    printf("Allocating GPU buffers...\n");
    uint8_t pat_ff = 0xFF;
    uint8_t pat_00 = 0x00;

    /* Word hash table */
    cl_mem wht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * WORD_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "wht_keys");
    cl_mem wht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * WORD_HT_CAPACITY, NULL, &err);
    CL_CHECK(err, "wht_values");

    /* Word pool SoA */
    cl_mem word_name_hash = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * WORD_CAPACITY, NULL, &err);
    cl_mem word_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(double) * WORD_CAPACITY, NULL, &err);
    cl_mem word_class_id = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * WORD_CAPACITY, NULL, &err);

    /* Word bump allocator */
    uint32_t zero = 0;
    cl_mem word_next_free = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);

    /* Pair hash table */
    cl_mem pht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * PAIR_HT_CAPACITY, NULL, &err);
    cl_mem pht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * PAIR_HT_CAPACITY, NULL, &err);

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

    /* Section hash table */
    cl_mem sht_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * SECTION_HT_CAPACITY, NULL, &err);
    cl_mem sht_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * SECTION_HT_CAPACITY, NULL, &err);

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

    /* Stats output */
    cl_mem d_stats = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * 3, NULL, &err);

    /* Initialize hash tables to empty */
    clEnqueueFillBuffer(queue, wht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * WORD_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, wht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * WORD_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * PAIR_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, sht_keys, &pat_ff, 1, 0,
        sizeof(uint64_t) * SECTION_HT_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, sht_values, &pat_ff, 1, 0,
        sizeof(uint32_t) * SECTION_HT_CAPACITY, 0, NULL, NULL);

    /* Initialize pool arrays to 0 */
    clEnqueueFillBuffer(queue, word_count, &pat_00, 1, 0,
        sizeof(double) * WORD_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, word_class_id, &pat_00, 1, 0,
        sizeof(uint32_t) * WORD_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pair_count, &pat_00, 1, 0,
        sizeof(double) * PAIR_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pair_mi, &pat_00, 1, 0,
        sizeof(double) * PAIR_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, pair_flags, &pat_00, 1, 0,
        sizeof(uint32_t) * PAIR_CAPACITY, 0, NULL, NULL);
    clEnqueueFillBuffer(queue, sec_count, &pat_00, 1, 0,
        sizeof(double) * SECTION_CAPACITY, 0, NULL, NULL);
    clFinish(queue);

    /* Calculate total GPU memory */
    size_t total_mem = 0;
    total_mem += sizeof(uint64_t) * WORD_HT_CAPACITY;  /* wht_keys */
    total_mem += sizeof(uint32_t) * WORD_HT_CAPACITY;  /* wht_values */
    total_mem += sizeof(uint64_t) * WORD_CAPACITY;      /* word_name_hash */
    total_mem += sizeof(double) * WORD_CAPACITY;        /* word_count */
    total_mem += sizeof(uint32_t) * WORD_CAPACITY;      /* word_class_id */
    total_mem += sizeof(uint64_t) * PAIR_HT_CAPACITY;   /* pht_keys */
    total_mem += sizeof(uint32_t) * PAIR_HT_CAPACITY;   /* pht_values */
    total_mem += sizeof(uint32_t) * PAIR_CAPACITY;      /* pair_word_a */
    total_mem += sizeof(uint32_t) * PAIR_CAPACITY;      /* pair_word_b */
    total_mem += sizeof(double) * PAIR_CAPACITY;        /* pair_count */
    total_mem += sizeof(double) * PAIR_CAPACITY;        /* pair_mi */
    total_mem += sizeof(uint32_t) * PAIR_CAPACITY;      /* pair_flags */
    total_mem += sizeof(uint64_t) * SECTION_HT_CAPACITY;/* sht_keys */
    total_mem += sizeof(uint32_t) * SECTION_HT_CAPACITY;/* sht_values */
    total_mem += sizeof(uint32_t) * SECTION_CAPACITY;   /* sec_word */
    total_mem += sizeof(uint64_t) * SECTION_CAPACITY;   /* sec_disjunct_hash */
    total_mem += sizeof(double) * SECTION_CAPACITY;     /* sec_count */
    printf("Total GPU memory: %zu MB\n\n", total_mem / (1024*1024));

    /* ═══ TEST 1: Create words ═══ */

    printf("--- Test 1: Create words ---\n");

    const char* test_words[] = {
        "the", "of", "and", "to", "a", "in", "was", "he", "she", "it",
        "that", "is", "for", "his", "with", "her", "had", "not", "at", "on",
        /* Duplicates to test dedup */
        "the", "of", "and", "the", "he", "she"
    };
    int num_words_in = 26;
    int num_unique_words = 20;

    uint64_t* h_word_hashes = malloc(sizeof(uint64_t) * num_words_in);
    for (int i = 0; i < num_words_in; i++) {
        h_word_hashes[i] = hash_word(test_words[i]);
    }

    cl_mem d_word_hashes = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * num_words_in, h_word_hashes, &err);
    uint32_t* h_word_indices = malloc(sizeof(uint32_t) * num_words_in);
    cl_mem d_word_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * num_words_in, NULL, &err);

    cl_uint nw = num_words_in;
    clSetKernelArg(k_word_foc, 0, sizeof(cl_mem), &wht_keys);
    clSetKernelArg(k_word_foc, 1, sizeof(cl_mem), &wht_values);
    clSetKernelArg(k_word_foc, 2, sizeof(cl_mem), &word_name_hash);
    clSetKernelArg(k_word_foc, 3, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_word_foc, 4, sizeof(cl_mem), &word_class_id);
    clSetKernelArg(k_word_foc, 5, sizeof(cl_mem), &word_next_free);
    clSetKernelArg(k_word_foc, 6, sizeof(cl_mem), &d_word_hashes);
    clSetKernelArg(k_word_foc, 7, sizeof(cl_mem), &d_word_out);
    clSetKernelArg(k_word_foc, 8, sizeof(cl_uint), &nw);

    double t0 = now_ms();
    size_t gs = ((num_words_in + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_word_foc, 1, NULL,
        &gs, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue word_foc");
    clFinish(queue);
    double t1 = now_ms();

    clEnqueueReadBuffer(queue, d_word_out, CL_TRUE, 0,
        sizeof(uint32_t) * num_words_in, h_word_indices, 0, NULL, NULL);

    printf("  Created %d words (with %d dups) in %.2f ms\n",
           num_words_in, num_words_in - num_unique_words, t1-t0);

    /* Check dedup: "the" appears at indices 0, 20, 23 — should all get same pool index */
    int dedup_ok = (h_word_indices[0] == h_word_indices[20] &&
                    h_word_indices[0] == h_word_indices[23]);
    printf("  'the' dedup: idx[0]=%u idx[20]=%u idx[23]=%u  %s\n",
           h_word_indices[0], h_word_indices[20], h_word_indices[23],
           dedup_ok ? "PASS" : "FAIL");

    /* "he"=7, "he"=24 */
    int dedup2 = (h_word_indices[7] == h_word_indices[24]);
    printf("  'he'  dedup: idx[7]=%u idx[24]=%u  %s\n",
           h_word_indices[7], h_word_indices[24],
           dedup2 ? "PASS" : "FAIL");

    /* Check pool stats */
    clSetKernelArg(k_stats, 0, sizeof(cl_mem), &word_next_free);
    clSetKernelArg(k_stats, 1, sizeof(cl_mem), &pair_next_free);
    clSetKernelArg(k_stats, 2, sizeof(cl_mem), &sec_next_free);
    clSetKernelArg(k_stats, 3, sizeof(cl_mem), &d_stats);

    size_t one = 1;
    clEnqueueNDRangeKernel(queue, k_stats, 1, NULL, &one, &one, 0, NULL, NULL);
    uint32_t stats[3];
    clEnqueueReadBuffer(queue, d_stats, CL_TRUE, 0, sizeof(stats), stats, 0, NULL, NULL);

    printf("  Pool: %u words, %u pairs, %u sections\n", stats[0], stats[1], stats[2]);
    printf("  %s\n\n", (stats[0] == num_unique_words) ? "PASS" : "FAIL");

    /* ═══ TEST 2: Create pairs ═══ */

    printf("--- Test 2: Create pairs ---\n");

    /* Create pairs for a sentence: "the cat was on the mat"
     * Words: the(0), cat(?), was(6), on(19), mat(?)
     * We need to create cat and mat first */
    uint64_t extra_hashes[2] = { hash_word("cat"), hash_word("mat") };
    cl_mem d_extra = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * 2, extra_hashes, &err);
    cl_mem d_extra_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * 2, NULL, &err);

    cl_uint n_extra = 2;
    clSetKernelArg(k_word_foc, 6, sizeof(cl_mem), &d_extra);
    clSetKernelArg(k_word_foc, 7, sizeof(cl_mem), &d_extra_out);
    clSetKernelArg(k_word_foc, 8, sizeof(cl_uint), &n_extra);
    gs = ((2 + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_word_foc, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    uint32_t extra_indices[2];
    clEnqueueReadBuffer(queue, d_extra_out, CL_TRUE, 0,
        sizeof(uint32_t) * 2, extra_indices, 0, NULL, NULL);

    uint32_t idx_the = h_word_indices[0];
    uint32_t idx_cat = extra_indices[0];
    uint32_t idx_was = h_word_indices[6];
    uint32_t idx_on  = h_word_indices[19];
    uint32_t idx_mat = extra_indices[1];

    printf("  Word indices: the=%u cat=%u was=%u on=%u mat=%u\n",
           idx_the, idx_cat, idx_was, idx_on, idx_mat);

    /* Create pairs within window=2: (the,cat) (the,was) (cat,was) (cat,on) (was,on) (was,the_2)
     * Plus (on,the_2) (on,mat) (the_2,mat) */
    int num_pairs_in = 9;
    uint32_t h_pair_a[] = {idx_the, idx_the, idx_cat, idx_cat, idx_was, idx_was, idx_on, idx_on, idx_the};
    uint32_t h_pair_b[] = {idx_cat, idx_was, idx_was, idx_on,  idx_on,  idx_the, idx_the,idx_mat,idx_mat};

    /* Add duplicates to test dedup */
    int num_with_dups = 12;
    uint32_t h_pair_a2[12], h_pair_b2[12];
    memcpy(h_pair_a2, h_pair_a, sizeof(uint32_t) * 9);
    memcpy(h_pair_b2, h_pair_b, sizeof(uint32_t) * 9);
    /* Duplicate first 3 pairs */
    h_pair_a2[9]  = idx_the; h_pair_b2[9]  = idx_cat;
    h_pair_a2[10] = idx_the; h_pair_b2[10] = idx_was;
    h_pair_a2[11] = idx_cat; h_pair_b2[11] = idx_was;

    cl_mem d_pair_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * num_with_dups, h_pair_a2, &err);
    cl_mem d_pair_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * num_with_dups, h_pair_b2, &err);
    cl_mem d_pair_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * num_with_dups, NULL, &err);

    cl_uint np = num_with_dups;
    clSetKernelArg(k_pair_foc, 0, sizeof(cl_mem), &pht_keys);
    clSetKernelArg(k_pair_foc, 1, sizeof(cl_mem), &pht_values);
    clSetKernelArg(k_pair_foc, 2, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_pair_foc, 3, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_pair_foc, 4, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_pair_foc, 5, sizeof(cl_mem), &pair_mi);
    clSetKernelArg(k_pair_foc, 6, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_pair_foc, 7, sizeof(cl_mem), &pair_next_free);
    clSetKernelArg(k_pair_foc, 8, sizeof(cl_mem), &d_pair_a);
    clSetKernelArg(k_pair_foc, 9, sizeof(cl_mem), &d_pair_b);
    clSetKernelArg(k_pair_foc, 10, sizeof(cl_mem), &d_pair_out);
    clSetKernelArg(k_pair_foc, 11, sizeof(cl_uint), &np);

    t0 = now_ms();
    gs = ((num_with_dups + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_pair_foc, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();

    uint32_t h_pair_indices[12];
    clEnqueueReadBuffer(queue, d_pair_out, CL_TRUE, 0,
        sizeof(uint32_t) * num_with_dups, h_pair_indices, 0, NULL, NULL);

    /* Pair dedup: indices 0 and 9 should match (the,cat) */
    int pair_dedup = (h_pair_indices[0] == h_pair_indices[9] &&
                      h_pair_indices[1] == h_pair_indices[10] &&
                      h_pair_indices[2] == h_pair_indices[11]);
    printf("  Created %d pairs (3 dups) in %.2f ms\n", num_with_dups, t1-t0);
    printf("  Pair dedup: %s\n", pair_dedup ? "PASS" : "FAIL");

    /* Note: pair(the,was) and pair(was,the) should be the same (canonical order) */
    int canon = (h_pair_indices[1] == h_pair_indices[5]);
    printf("  Canonical order (the,was)==(was,the): idx=%u==%u  %s\n",
           h_pair_indices[1], h_pair_indices[5], canon ? "PASS" : "FAIL");

    clEnqueueNDRangeKernel(queue, k_stats, 1, NULL, &one, &one, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_stats, CL_TRUE, 0, sizeof(stats), stats, 0, NULL, NULL);
    /* 9 pairs but (the,was) and (was,the) deduplicate to same, and
     * (on,the) and (the,on) too. Let's count unique: */
    printf("  Pool: %u words, %u pairs, %u sections\n", stats[0], stats[1], stats[2]);

    int num_unique_pairs = stats[1];
    printf("  %s\n\n", (num_unique_pairs > 0 && num_unique_pairs <= num_pairs_in) ? "PASS" : "FAIL");

    /* ═══ TEST 3: Create sections ═══ */

    printf("--- Test 3: Create sections ---\n");

    /* Simulate sections: word + disjunct hash */
    int num_sections = 8;
    uint32_t h_sec_words[] = {idx_the, idx_the, idx_cat, idx_cat, idx_was, idx_was, idx_on, idx_mat};
    uint64_t h_sec_dhash[] = {
        0x1111111111111111ULL,  /* the: cat+ */
        0x2222222222222222ULL,  /* the: was+ */
        0x3333333333333333ULL,  /* cat: the- was+ */
        0x4444444444444444ULL,  /* cat: was- on+ */
        0x5555555555555555ULL,  /* was: cat- on+ */
        0x2222222222222222ULL,  /* was: same disjunct as "the: was+" — different word, different section */
        0x6666666666666666ULL,  /* on: was- the+ */
        0x7777777777777777ULL,  /* mat: on- the+ */
    };

    cl_mem d_sec_w = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * num_sections, h_sec_words, &err);
    cl_mem d_sec_d = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * num_sections, h_sec_dhash, &err);
    cl_mem d_sec_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * num_sections, NULL, &err);

    cl_uint ns = num_sections;
    clSetKernelArg(k_sec_foc, 0, sizeof(cl_mem), &sht_keys);
    clSetKernelArg(k_sec_foc, 1, sizeof(cl_mem), &sht_values);
    clSetKernelArg(k_sec_foc, 2, sizeof(cl_mem), &sec_word);
    clSetKernelArg(k_sec_foc, 3, sizeof(cl_mem), &sec_disjunct_hash);
    clSetKernelArg(k_sec_foc, 4, sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_sec_foc, 5, sizeof(cl_mem), &sec_next_free);
    clSetKernelArg(k_sec_foc, 6, sizeof(cl_mem), &d_sec_w);
    clSetKernelArg(k_sec_foc, 7, sizeof(cl_mem), &d_sec_d);
    clSetKernelArg(k_sec_foc, 8, sizeof(cl_mem), &d_sec_out);
    clSetKernelArg(k_sec_foc, 9, sizeof(cl_uint), &ns);

    t0 = now_ms();
    gs = ((num_sections + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_sec_foc, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();

    uint32_t h_sec_indices[8];
    clEnqueueReadBuffer(queue, d_sec_out, CL_TRUE, 0,
        sizeof(uint32_t) * num_sections, h_sec_indices, 0, NULL, NULL);

    printf("  Created %d sections in %.2f ms\n", num_sections, t1-t0);

    /* Section(the, 0x222..) != Section(was, 0x222..) — different word, different section */
    int sec_diff = (h_sec_indices[1] != h_sec_indices[5]);
    printf("  Different word same disjunct = different section: %s\n",
           sec_diff ? "PASS" : "FAIL");

    clEnqueueNDRangeKernel(queue, k_stats, 1, NULL, &one, &one, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_stats, CL_TRUE, 0, sizeof(stats), stats, 0, NULL, NULL);
    printf("  Pool: %u words, %u pairs, %u sections\n", stats[0], stats[1], stats[2]);
    printf("  %s\n\n", (stats[2] == (uint32_t)num_sections) ? "PASS" : "FAIL");

    /* ═══ TEST 4: Count pairs ═══ */

    printf("--- Test 4: Count pairs (atomic double increment) ---\n");

    /* Count pair 0 (the,cat) 100 times */
    int count_n = 100;
    uint32_t* h_count_indices = malloc(sizeof(uint32_t) * count_n);
    for (int i = 0; i < count_n; i++) {
        h_count_indices[i] = h_pair_indices[0]; /* (the,cat) */
    }

    cl_mem d_count_idx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * count_n, h_count_indices, &err);

    cl_uint cn = count_n;
    clSetKernelArg(k_count_pairs, 0, sizeof(cl_mem), &pair_count);
    clSetKernelArg(k_count_pairs, 1, sizeof(cl_mem), &pair_word_a);
    clSetKernelArg(k_count_pairs, 2, sizeof(cl_mem), &pair_word_b);
    clSetKernelArg(k_count_pairs, 3, sizeof(cl_mem), &word_count);
    clSetKernelArg(k_count_pairs, 4, sizeof(cl_mem), &pair_flags);
    clSetKernelArg(k_count_pairs, 5, sizeof(cl_mem), &d_count_idx);
    clSetKernelArg(k_count_pairs, 6, sizeof(cl_uint), &cn);

    t0 = now_ms();
    gs = ((count_n + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_count_pairs, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();

    /* Read back the pair count and word marginals */
    double h_pair_c;
    clEnqueueReadBuffer(queue, pair_count, CL_TRUE,
        sizeof(double) * h_pair_indices[0], sizeof(double), &h_pair_c, 0, NULL, NULL);

    double h_wc_the, h_wc_cat;
    clEnqueueReadBuffer(queue, word_count, CL_TRUE,
        sizeof(double) * idx_the, sizeof(double), &h_wc_the, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, word_count, CL_TRUE,
        sizeof(double) * idx_cat, sizeof(double), &h_wc_cat, 0, NULL, NULL);

    printf("  Counted pair(the,cat) %d times in %.2f ms\n", count_n, t1-t0);
    printf("  pair_count = %.1f (expected %.1f)  %s\n",
           h_pair_c, (double)count_n,
           (fabs(h_pair_c - count_n) < 0.5) ? "PASS" : "FAIL");
    printf("  word_count[the] = %.1f  word_count[cat] = %.1f  (both expect %.1f)  %s\n",
           h_wc_the, h_wc_cat, (double)count_n,
           (fabs(h_wc_the - count_n) < 0.5 && fabs(h_wc_cat - count_n) < 0.5)
           ? "PASS" : "FAIL");

    /* Check dirty flag */
    uint32_t h_flag;
    clEnqueueReadBuffer(queue, pair_flags, CL_TRUE,
        sizeof(uint32_t) * h_pair_indices[0], sizeof(uint32_t), &h_flag, 0, NULL, NULL);
    printf("  Dirty flag = %u (expected 1)  %s\n\n", h_flag, (h_flag == 1) ? "PASS" : "FAIL");

    /* ═══ TEST 5: Count sections ═══ */

    printf("--- Test 5: Count sections ---\n");

    /* Count section 0 (the: cat+) 50 times */
    int sec_count_n = 50;
    uint32_t* h_sec_cnt_idx = malloc(sizeof(uint32_t) * sec_count_n);
    for (int i = 0; i < sec_count_n; i++) {
        h_sec_cnt_idx[i] = h_sec_indices[0];
    }

    cl_mem d_sec_cnt_idx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * sec_count_n, h_sec_cnt_idx, &err);

    cl_uint scn = sec_count_n;
    clSetKernelArg(k_count_sec, 0, sizeof(cl_mem), &sec_count);
    clSetKernelArg(k_count_sec, 1, sizeof(cl_mem), &d_sec_cnt_idx);
    clSetKernelArg(k_count_sec, 2, sizeof(cl_uint), &scn);

    t0 = now_ms();
    gs = ((sec_count_n + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_count_sec, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();

    double h_sec_c;
    clEnqueueReadBuffer(queue, sec_count, CL_TRUE,
        sizeof(double) * h_sec_indices[0], sizeof(double), &h_sec_c, 0, NULL, NULL);

    printf("  Counted section 0 %d times in %.2f ms\n", sec_count_n, t1-t0);
    printf("  sec_count = %.1f (expected %.1f)  %s\n\n",
           h_sec_c, (double)sec_count_n,
           (fabs(h_sec_c - sec_count_n) < 0.5) ? "PASS" : "FAIL");

    /* ═══ TEST 6: Bulk performance ═══ */

    printf("--- Test 6: Bulk performance ---\n");

    /* Create 100K words */
    int bulk_words = 100000;
    uint64_t* h_bulk_hashes = malloc(sizeof(uint64_t) * bulk_words);
    uint64_t rng = 0xDEADBEEFCAFEBABEULL;
    for (int i = 0; i < bulk_words; i++) {
        rng += 0x9E3779B97F4A7C15ULL;
        uint64_t z = rng;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        if (z == HT_EMPTY_KEY) z = 0;
        h_bulk_hashes[i] = z;
    }

    cl_mem d_bulk = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * bulk_words, h_bulk_hashes, &err);
    cl_mem d_bulk_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * bulk_words, NULL, &err);

    cl_uint bw = bulk_words;
    clSetKernelArg(k_word_foc, 6, sizeof(cl_mem), &d_bulk);
    clSetKernelArg(k_word_foc, 7, sizeof(cl_mem), &d_bulk_out);
    clSetKernelArg(k_word_foc, 8, sizeof(cl_uint), &bw);

    t0 = now_ms();
    gs = ((bulk_words + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_word_foc, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();
    printf("  100K words created in %.1f ms (%.1f M/sec)\n",
           t1-t0, bulk_words / ((t1-t0) / 1000.0) / 1e6);

    /* Create 1M pairs from random word indices */
    int bulk_pairs = 1000000;
    uint32_t* h_bulk_pa = malloc(sizeof(uint32_t) * bulk_pairs);
    uint32_t* h_bulk_pb = malloc(sizeof(uint32_t) * bulk_pairs);
    for (int i = 0; i < bulk_pairs; i++) {
        rng += 0x9E3779B97F4A7C15ULL;
        h_bulk_pa[i] = (uint32_t)(rng >> 32) % bulk_words;
        h_bulk_pb[i] = (uint32_t)(rng & 0xFFFFFFFF) % bulk_words;
    }

    cl_mem d_bulk_pa = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bulk_pairs, h_bulk_pa, &err);
    cl_mem d_bulk_pb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * bulk_pairs, h_bulk_pb, &err);
    cl_mem d_bulk_po = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * bulk_pairs, NULL, &err);

    cl_uint bp = bulk_pairs;
    clSetKernelArg(k_pair_foc, 8, sizeof(cl_mem), &d_bulk_pa);
    clSetKernelArg(k_pair_foc, 9, sizeof(cl_mem), &d_bulk_pb);
    clSetKernelArg(k_pair_foc, 10, sizeof(cl_mem), &d_bulk_po);
    clSetKernelArg(k_pair_foc, 11, sizeof(cl_uint), &bp);

    t0 = now_ms();
    gs = ((bulk_pairs + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, k_pair_foc, 1, NULL, &gs, &local_size, 0, NULL, NULL);
    clFinish(queue);
    t1 = now_ms();
    printf("  1M pairs created in %.1f ms (%.1f M/sec)\n",
           t1-t0, bulk_pairs / ((t1-t0) / 1000.0) / 1e6);

    /* Final stats */
    clEnqueueNDRangeKernel(queue, k_stats, 1, NULL, &one, &one, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_stats, CL_TRUE, 0, sizeof(stats), stats, 0, NULL, NULL);
    printf("  Final pool: %u words, %u pairs, %u sections\n\n", stats[0], stats[1], stats[2]);

    printf("=== All tests complete ===\n");

    /* ─── Cleanup ─── */

    clReleaseMemObject(wht_keys);
    clReleaseMemObject(wht_values);
    clReleaseMemObject(word_name_hash);
    clReleaseMemObject(word_count);
    clReleaseMemObject(word_class_id);
    clReleaseMemObject(word_next_free);
    clReleaseMemObject(pht_keys);
    clReleaseMemObject(pht_values);
    clReleaseMemObject(pair_word_a);
    clReleaseMemObject(pair_word_b);
    clReleaseMemObject(pair_count);
    clReleaseMemObject(pair_mi);
    clReleaseMemObject(pair_flags);
    clReleaseMemObject(pair_next_free);
    clReleaseMemObject(sht_keys);
    clReleaseMemObject(sht_values);
    clReleaseMemObject(sec_word);
    clReleaseMemObject(sec_disjunct_hash);
    clReleaseMemObject(sec_count);
    clReleaseMemObject(sec_next_free);
    clReleaseMemObject(d_stats);
    /* ... remaining temp buffers */
    clReleaseKernel(k_word_foc);
    clReleaseKernel(k_pair_foc);
    clReleaseKernel(k_sec_foc);
    clReleaseKernel(k_count_pairs);
    clReleaseKernel(k_count_sec);
    clReleaseKernel(k_stats);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(h_word_hashes);
    free(h_word_indices);
    free(h_count_indices);
    free(h_sec_cnt_idx);
    free(h_bulk_hashes);
    free(h_bulk_pa);
    free(h_bulk_pb);
    free(combined);
    free(ht_src);
    free(as_src);

    return 0;
}
