/*
 * test-hashtable.c -- Standalone test for GPU hash table
 *
 * Compile: gcc -O2 -o test-hashtable test-hashtable.c -lOpenCL
 * Run:     ./test-hashtable
 *
 * Tests:
 *   1. Bulk insert 1M keys
 *   2. Lookup all inserted keys (verify 100% hit)
 *   3. Lookup non-existent keys (verify 100% miss)
 *   4. Delete some keys, verify they're gone
 *   5. Insert-or-increment (counting)
 *   6. Iterate and verify count
 *   7. Performance: inserts/sec and lookups/sec
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

/* ─── Configuration ─── */

/* Table capacity — must be power of 2.
 * At 50% load factor, 4M slots supports 2M entries. */
#define TABLE_CAPACITY  (4 * 1024 * 1024)  /* 4M slots */
#define NUM_TEST_ITEMS  (1 * 1024 * 1024)  /* 1M test entries */

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

/* ─── Read kernel source ─── */

char* read_file(const char* path, size_t* len)
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(*len + 1);
    fread(buf, 1, *len, f);
    buf[*len] = '\0';
    fclose(f);
    return buf;
}

/* ─── Simple PRNG (splitmix64) ─── */

static uint64_t rng_state = 0x12345678DEADBEEFULL;

uint64_t next_random(void)
{
    rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
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

    printf("=== GPU Hash Table Test ===\n");
    printf("Table capacity: %d slots (%lu MB)\n",
           TABLE_CAPACITY,
           (unsigned long)(TABLE_CAPACITY * (sizeof(uint64_t) + sizeof(uint32_t))) / (1024*1024));
    printf("Test items:     %d\n\n", NUM_TEST_ITEMS);

    /* ─── OpenCL setup ─── */

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CL_CHECK(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CL_CHECK(err, "clGetDeviceIDs");

    /* Print device name */
    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("GPU: %s\n", dev_name);

    /* Check for int64 atomics */
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    if (!strstr(extensions, "cl_khr_int64_base_atomics")) {
        fprintf(stderr, "ERROR: Device does not support cl_khr_int64_base_atomics\n");
        return 1;
    }
    printf("cl_khr_int64_base_atomics: supported\n\n");

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    /* ─── Build kernel ─── */

    /* Find kernel source relative to this test file or in same dir */
    const char* kernel_paths[] = {
        "gpu-hashtable.cl",
        "opencog/opencl/atomspace/gpu-hashtable.cl",
        NULL
    };
    char* src = NULL;
    size_t src_len = 0;
    for (int i = 0; kernel_paths[i]; i++) {
        FILE* f = fopen(kernel_paths[i], "r");
        if (f) {
            fclose(f);
            src = read_file(kernel_paths[i], &src_len);
            printf("Kernel source: %s\n", kernel_paths[i]);
            break;
        }
    }
    if (!src) {
        fprintf(stderr, "Cannot find gpu-hashtable.cl\n");
        return 1;
    }

    cl_program program = clCreateProgramWithSource(ctx, 1,
        (const char**)&src, &src_len, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        return 1;
    }
    printf("Kernel compiled successfully\n\n");

    /* Create kernels */
    cl_kernel k_insert = clCreateKernel(program, "ht_insert", &err);
    CL_CHECK(err, "create ht_insert");
    cl_kernel k_lookup = clCreateKernel(program, "ht_lookup", &err);
    CL_CHECK(err, "create ht_lookup");
    cl_kernel k_delete = clCreateKernel(program, "ht_delete", &err);
    CL_CHECK(err, "create ht_delete");
    cl_kernel k_inc = clCreateKernel(program, "ht_insert_or_increment", &err);
    CL_CHECK(err, "create ht_insert_or_increment");
    cl_kernel k_iter = clCreateKernel(program, "ht_iterate", &err);
    CL_CHECK(err, "create ht_iterate");

    /* ─── Allocate table on GPU ─── */

    cl_ulong capacity = TABLE_CAPACITY;

    cl_mem d_keys = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint64_t) * TABLE_CAPACITY, NULL, &err);
    CL_CHECK(err, "alloc table keys");

    cl_mem d_values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(uint32_t) * TABLE_CAPACITY, NULL, &err);
    CL_CHECK(err, "alloc table values");

    /* Initialize table to empty (all 0xFF bytes) */
    uint8_t pattern_ff = 0xFF;
    err = clEnqueueFillBuffer(queue, d_keys, &pattern_ff, 1,
        0, sizeof(uint64_t) * TABLE_CAPACITY, 0, NULL, NULL);
    CL_CHECK(err, "fill table keys");
    err = clEnqueueFillBuffer(queue, d_values, &pattern_ff, 1,
        0, sizeof(uint32_t) * TABLE_CAPACITY, 0, NULL, NULL);
    CL_CHECK(err, "fill table values");
    clFinish(queue);

    /* ─── Generate test data ─── */

    uint64_t* h_keys   = malloc(sizeof(uint64_t) * NUM_TEST_ITEMS);
    uint32_t* h_values = malloc(sizeof(uint32_t) * NUM_TEST_ITEMS);
    uint32_t* h_results = malloc(sizeof(uint32_t) * NUM_TEST_ITEMS);

    rng_state = 0x12345678DEADBEEFULL;
    for (int i = 0; i < NUM_TEST_ITEMS; i++) {
        uint64_t k = next_random();
        /* Avoid sentinel key */
        if (k == HT_EMPTY_KEY) k = 0;
        h_keys[i]   = k;
        h_values[i] = (uint32_t)(i & 0xFFFFFFFF);
    }

    /* Upload test data to GPU */
    cl_mem d_in_keys = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * NUM_TEST_ITEMS, h_keys, &err);
    CL_CHECK(err, "alloc input keys");
    cl_mem d_in_values = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * NUM_TEST_ITEMS, h_values, &err);
    CL_CHECK(err, "alloc input values");
    cl_mem d_out_values = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * NUM_TEST_ITEMS, NULL, &err);
    CL_CHECK(err, "alloc output values");

    size_t global_size = NUM_TEST_ITEMS;
    size_t local_size = 256;
    /* Round up global size to multiple of local size */
    global_size = ((global_size + local_size - 1) / local_size) * local_size;

    cl_uint num_items = NUM_TEST_ITEMS;

    /* ═══ TEST 1: Bulk Insert ═══ */

    printf("--- Test 1: Insert %d items ---\n", NUM_TEST_ITEMS);

    clSetKernelArg(k_insert, 0, sizeof(cl_mem), &d_keys);
    clSetKernelArg(k_insert, 1, sizeof(cl_mem), &d_values);
    clSetKernelArg(k_insert, 2, sizeof(cl_ulong), &capacity);
    clSetKernelArg(k_insert, 3, sizeof(cl_mem), &d_in_keys);
    clSetKernelArg(k_insert, 4, sizeof(cl_mem), &d_in_values);
    clSetKernelArg(k_insert, 5, sizeof(cl_uint), &num_items);

    double t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_insert, 1, NULL,
        &global_size, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue insert");
    clFinish(queue);
    double t1 = now_ms();

    printf("  Inserted %d items in %.1f ms (%.1f M keys/sec)\n",
           NUM_TEST_ITEMS, t1-t0, NUM_TEST_ITEMS / ((t1-t0) / 1000.0) / 1e6);

    /* ═══ TEST 2: Lookup all (should all hit) ═══ */

    printf("--- Test 2: Lookup %d items (expect all found) ---\n", NUM_TEST_ITEMS);

    clSetKernelArg(k_lookup, 0, sizeof(cl_mem), &d_keys);
    clSetKernelArg(k_lookup, 1, sizeof(cl_mem), &d_values);
    clSetKernelArg(k_lookup, 2, sizeof(cl_ulong), &capacity);
    clSetKernelArg(k_lookup, 3, sizeof(cl_mem), &d_in_keys);
    clSetKernelArg(k_lookup, 4, sizeof(cl_mem), &d_out_values);
    clSetKernelArg(k_lookup, 5, sizeof(cl_uint), &num_items);

    t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_lookup, 1, NULL,
        &global_size, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue lookup");
    clFinish(queue);
    t1 = now_ms();

    /* Read back results */
    err = clEnqueueReadBuffer(queue, d_out_values, CL_TRUE, 0,
        sizeof(uint32_t) * NUM_TEST_ITEMS, h_results, 0, NULL, NULL);
    CL_CHECK(err, "read lookup results");

    int hits = 0, misses = 0, wrong = 0;
    for (int i = 0; i < NUM_TEST_ITEMS; i++) {
        if (h_results[i] == HT_EMPTY_VALUE)
            misses++;
        else if (h_results[i] == h_values[i])
            hits++;
        else
            wrong++;
    }

    printf("  Lookup in %.1f ms (%.1f M keys/sec)\n",
           t1-t0, NUM_TEST_ITEMS / ((t1-t0) / 1000.0) / 1e6);
    printf("  Hits: %d  Misses: %d  Wrong: %d\n", hits, misses, wrong);
    printf("  %s\n\n", (hits == NUM_TEST_ITEMS && misses == 0 && wrong == 0)
           ? "PASS" : "FAIL");

    /* ═══ TEST 3: Lookup non-existent keys (should all miss) ═══ */

    printf("--- Test 3: Lookup %d non-existent keys ---\n", NUM_TEST_ITEMS);

    /* Generate different keys */
    rng_state = 0xABCDABCDABCDABCDULL;
    uint64_t* h_miss_keys = malloc(sizeof(uint64_t) * NUM_TEST_ITEMS);
    for (int i = 0; i < NUM_TEST_ITEMS; i++) {
        uint64_t k = next_random();
        if (k == HT_EMPTY_KEY) k = 0;
        h_miss_keys[i] = k;
    }

    cl_mem d_miss_keys = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * NUM_TEST_ITEMS, h_miss_keys, &err);
    CL_CHECK(err, "alloc miss keys");

    clSetKernelArg(k_lookup, 3, sizeof(cl_mem), &d_miss_keys);

    t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_lookup, 1, NULL,
        &global_size, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue miss lookup");
    clFinish(queue);
    t1 = now_ms();

    err = clEnqueueReadBuffer(queue, d_out_values, CL_TRUE, 0,
        sizeof(uint32_t) * NUM_TEST_ITEMS, h_results, 0, NULL, NULL);
    CL_CHECK(err, "read miss results");

    misses = 0;
    for (int i = 0; i < NUM_TEST_ITEMS; i++) {
        if (h_results[i] == HT_EMPTY_VALUE) misses++;
    }

    printf("  Lookup in %.1f ms\n", t1-t0);
    printf("  Misses: %d / %d\n", misses, NUM_TEST_ITEMS);
    printf("  %s\n\n", (misses == NUM_TEST_ITEMS) ? "PASS" : "FAIL");

    /* ═══ TEST 4: Delete first 1000 keys, verify ═══ */

    int num_delete = 1000;
    printf("--- Test 4: Delete %d keys ---\n", num_delete);

    cl_mem d_del_keys = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * num_delete, h_keys, &err);
    CL_CHECK(err, "alloc del keys");

    cl_uint nd = num_delete;
    clSetKernelArg(k_delete, 0, sizeof(cl_mem), &d_keys);
    clSetKernelArg(k_delete, 1, sizeof(cl_mem), &d_values);
    clSetKernelArg(k_delete, 2, sizeof(cl_ulong), &capacity);
    clSetKernelArg(k_delete, 3, sizeof(cl_mem), &d_del_keys);
    clSetKernelArg(k_delete, 4, sizeof(cl_uint), &nd);

    size_t del_global = ((num_delete + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_delete, 1, NULL,
        &del_global, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue delete");
    clFinish(queue);

    /* Lookup deleted keys — should miss */
    cl_mem d_del_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * num_delete, h_keys, &err);
    cl_mem d_del_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * num_delete, NULL, &err);

    clSetKernelArg(k_lookup, 3, sizeof(cl_mem), &d_del_query);
    clSetKernelArg(k_lookup, 4, sizeof(cl_mem), &d_del_out);
    clSetKernelArg(k_lookup, 5, sizeof(cl_uint), &nd);

    err = clEnqueueNDRangeKernel(queue, k_lookup, 1, NULL,
        &del_global, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue del verify");
    clFinish(queue);

    uint32_t* h_del_results = malloc(sizeof(uint32_t) * num_delete);
    err = clEnqueueReadBuffer(queue, d_del_out, CL_TRUE, 0,
        sizeof(uint32_t) * num_delete, h_del_results, 0, NULL, NULL);

    int del_gone = 0;
    for (int i = 0; i < num_delete; i++) {
        if (h_del_results[i] == HT_EMPTY_VALUE) del_gone++;
    }
    printf("  Deleted keys returning empty: %d / %d\n", del_gone, num_delete);
    printf("  %s\n\n", (del_gone == num_delete) ? "PASS" : "FAIL");

    /* ═══ TEST 5: Insert-or-increment ═══ */

    printf("--- Test 5: Insert-or-increment (counting) ---\n");

    /* Re-initialize table: keys to 0xFF (empty), values to 0 (for counting) */
    err = clEnqueueFillBuffer(queue, d_keys, &pattern_ff, 1,
        0, sizeof(uint64_t) * TABLE_CAPACITY, 0, NULL, NULL);
    CL_CHECK(err, "refill keys");
    uint8_t pattern_00 = 0x00;
    err = clEnqueueFillBuffer(queue, d_values, &pattern_00, 1,
        0, sizeof(uint32_t) * TABLE_CAPACITY, 0, NULL, NULL);
    CL_CHECK(err, "refill values to 0");
    clFinish(queue);

    /* Insert same 100 keys 1000 times each = 100K operations */
    int inc_unique = 100;
    int inc_repeats = 1000;
    int inc_total = inc_unique * inc_repeats;
    uint64_t* h_inc_keys = malloc(sizeof(uint64_t) * inc_total);

    rng_state = 0xFEEDFACECAFEBABEULL;
    uint64_t base_keys[100];
    for (int i = 0; i < inc_unique; i++) {
        base_keys[i] = next_random();
        if (base_keys[i] == HT_EMPTY_KEY) base_keys[i] = 0;
    }
    for (int r = 0; r < inc_repeats; r++) {
        for (int i = 0; i < inc_unique; i++) {
            h_inc_keys[r * inc_unique + i] = base_keys[i];
        }
    }

    cl_mem d_inc_keys = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * inc_total, h_inc_keys, &err);
    CL_CHECK(err, "alloc inc keys");

    cl_uint inc_n = inc_total;
    clSetKernelArg(k_inc, 0, sizeof(cl_mem), &d_keys);
    clSetKernelArg(k_inc, 1, sizeof(cl_mem), &d_values);
    clSetKernelArg(k_inc, 2, sizeof(cl_ulong), &capacity);
    clSetKernelArg(k_inc, 3, sizeof(cl_mem), &d_inc_keys);
    clSetKernelArg(k_inc, 4, sizeof(cl_uint), &inc_n);

    size_t inc_global = ((inc_total + local_size - 1) / local_size) * local_size;
    t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_inc, 1, NULL,
        &inc_global, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue increment");
    clFinish(queue);
    t1 = now_ms();
    printf("  %d increments in %.1f ms\n", inc_total, t1-t0);

    /* Lookup the 100 keys, verify counts = 1000 (first insert = 1, then 999 increments) */
    cl_mem d_inc_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint64_t) * inc_unique, base_keys, &err);
    cl_mem d_inc_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * inc_unique, NULL, &err);
    cl_uint inc_u = inc_unique;

    clSetKernelArg(k_lookup, 3, sizeof(cl_mem), &d_inc_query);
    clSetKernelArg(k_lookup, 4, sizeof(cl_mem), &d_inc_out);
    clSetKernelArg(k_lookup, 5, sizeof(cl_uint), &inc_u);

    size_t inc_q_global = ((inc_unique + local_size - 1) / local_size) * local_size;
    err = clEnqueueNDRangeKernel(queue, k_lookup, 1, NULL,
        &inc_q_global, &local_size, 0, NULL, NULL);
    clFinish(queue);

    uint32_t* h_inc_results = malloc(sizeof(uint32_t) * inc_unique);
    clEnqueueReadBuffer(queue, d_inc_out, CL_TRUE, 0,
        sizeof(uint32_t) * inc_unique, h_inc_results, 0, NULL, NULL);

    int correct_counts = 0;
    for (int i = 0; i < inc_unique; i++) {
        if (h_inc_results[i] == (uint32_t)inc_repeats) correct_counts++;
        else if (i < 5) printf("  key %d: expected %d got %u\n", i, inc_repeats, h_inc_results[i]);
    }
    printf("  Correct counts (%d): %d / %d\n", inc_repeats, correct_counts, inc_unique);
    printf("  %s\n\n", (correct_counts == inc_unique) ? "PASS" : "FAIL");

    /* ═══ TEST 6: Iterate ═══ */

    printf("--- Test 6: Iterate (collect non-empty entries) ---\n");

    cl_mem d_iter_keys = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint64_t) * inc_unique * 2, NULL, &err);
    cl_mem d_iter_values = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        sizeof(uint32_t) * inc_unique * 2, NULL, &err);
    uint32_t zero = 0;
    cl_mem d_iter_count = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t), &zero, &err);
    cl_uint max_out = inc_unique * 2;

    clSetKernelArg(k_iter, 0, sizeof(cl_mem), &d_keys);
    clSetKernelArg(k_iter, 1, sizeof(cl_mem), &d_values);
    clSetKernelArg(k_iter, 2, sizeof(cl_ulong), &capacity);
    clSetKernelArg(k_iter, 3, sizeof(cl_mem), &d_iter_keys);
    clSetKernelArg(k_iter, 4, sizeof(cl_mem), &d_iter_values);
    clSetKernelArg(k_iter, 5, sizeof(cl_mem), &d_iter_count);
    clSetKernelArg(k_iter, 6, sizeof(cl_uint), &max_out);

    size_t iter_global = ((TABLE_CAPACITY + local_size - 1) / local_size) * local_size;
    t0 = now_ms();
    err = clEnqueueNDRangeKernel(queue, k_iter, 1, NULL,
        &iter_global, &local_size, 0, NULL, NULL);
    CL_CHECK(err, "enqueue iterate");
    clFinish(queue);
    t1 = now_ms();

    uint32_t iter_count;
    clEnqueueReadBuffer(queue, d_iter_count, CL_TRUE, 0,
        sizeof(uint32_t), &iter_count, 0, NULL, NULL);
    printf("  Iterated %u M slots in %.1f ms, found %u entries\n",
           TABLE_CAPACITY / (1024*1024), t1-t0, iter_count);
    printf("  Expected: %d entries\n", inc_unique);
    printf("  %s\n\n", (iter_count == (uint32_t)inc_unique) ? "PASS" : "FAIL");

    /* ═══ Summary ═══ */

    printf("=== All tests complete ===\n");

    /* ─── Cleanup ─── */

    clReleaseMemObject(d_keys);
    clReleaseMemObject(d_values);
    clReleaseMemObject(d_in_keys);
    clReleaseMemObject(d_in_values);
    clReleaseMemObject(d_out_values);
    clReleaseMemObject(d_miss_keys);
    clReleaseMemObject(d_del_keys);
    clReleaseMemObject(d_del_query);
    clReleaseMemObject(d_del_out);
    clReleaseMemObject(d_inc_keys);
    clReleaseMemObject(d_inc_query);
    clReleaseMemObject(d_inc_out);
    clReleaseMemObject(d_iter_keys);
    clReleaseMemObject(d_iter_values);
    clReleaseMemObject(d_iter_count);
    clReleaseKernel(k_insert);
    clReleaseKernel(k_lookup);
    clReleaseKernel(k_delete);
    clReleaseKernel(k_inc);
    clReleaseKernel(k_iter);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_keys);
    free(h_values);
    free(h_results);
    free(h_miss_keys);
    free(h_del_results);
    free(h_inc_keys);
    free(h_inc_results);
    free(src);

    return 0;
}
