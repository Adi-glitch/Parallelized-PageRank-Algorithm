#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <sys/time.h>

#define MAX_LINE 1024
#define THREADS_PER_BLOCK_X 256 

int N_nodes;
double D_damping_factor; 
int NUM_iters;   

// Host arrays
int** incoming_links_host;
int* incoming_counts_host;
int* out_degree_host;
double* pr_new_host;

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void allocate_host_memory() {
    incoming_links_host = (int**)malloc(N_nodes * sizeof(int*));
    assert(incoming_links_host != NULL);
    incoming_counts_host = (int*)calloc(N_nodes, sizeof(int));
    assert(incoming_counts_host != NULL);
    out_degree_host = (int*)calloc(N_nodes, sizeof(int));
    assert(out_degree_host != NULL);
    pr_new_host = (double*)calloc(N_nodes, sizeof(double));
    assert(pr_new_host != NULL);
}

void read_graph_data(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening graph file");
        exit(EXIT_FAILURE);
    }
    char line[MAX_LINE];
    int from_node, to_node;

    // First pass: count degrees and incoming links
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%d\t%d", &from_node, &to_node) == 2) {
            // Ensure nodes are within bounds if N_nodes is pre-determined and fixed
            // For this code, we assume N_nodes is the max node ID + 1, determined by input or a large enough allocation
            if (from_node >= N_nodes || to_node >= N_nodes || from_node < 0 || to_node < 0) {
                fprintf(stderr, "Error: Node ID out of bounds (max %d): %d or %d\n", N_nodes-1, from_node, to_node);
                // Potentially resize or exit, depending on how N_nodes is determined.
                // If N_nodes is from argv, this is a hard error.
                // If N_nodes is dynamically determined, this might involve reallocating.
                // For simplicity here, assuming N_nodes is large enough or input graph is valid.
                // exit(EXIT_FAILURE); // Or handle more gracefully
                continue; // Skip this edge if N is strictly defined
            }
            out_degree_host[from_node]++;
            incoming_counts_host[to_node]++;
        }
    }

    // Allocate memory for each node's incoming links list
    for (int i = 0; i < N_nodes; i++) {
        if (incoming_counts_host[i] > 0) {
            incoming_links_host[i] = (int*)malloc(incoming_counts_host[i] * sizeof(int));
            assert(incoming_links_host[i] != NULL);
        } else {
            incoming_links_host[i] = NULL;
        }
        incoming_counts_host[i] = 0; // Reset to use as an index in the next pass
    }

    rewind(fp); // Go back to the beginning of the file

    // Second pass: populate incoming links
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%d\t%d", &from_node, &to_node) == 2) {
             if (from_node >= N_nodes || to_node >= N_nodes || from_node < 0 || to_node < 0) {
                continue; // Skip this edge if N is strictly defined
            }
            // incoming_links_host[to_node] stores a list of nodes that link TO to_node
            incoming_links_host[to_node][incoming_counts_host[to_node]++] = from_node;
        }
    }
    fclose(fp);
}

// Kernel to calculate sum of PageRanks from dangling nodes for the current iteration
__global__ void sum_dangling_pageranks_kernel(const double* d_pr_old,
                                             const int* d_out_degree,
                                             double* d_total_dangling_sum, // Device pointer to a single double
                                             int num_nodes) {
    __shared__ double sdata[THREADS_PER_BLOCK_X]; // Shared memory for reduction within a block

    unsigned int tid_in_block = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    double current_val = 0.0;
    if (global_idx < num_nodes && d_out_degree[global_idx] == 0) {
        current_val = d_pr_old[global_idx] / (double)num_nodes;
    }
    sdata[tid_in_block] = current_val;

    __syncthreads(); // Ensure all threads in the block have loaded their value into sdata

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads(); // Synchronize after each step of reduction
    }

    // First thread in each block adds its block's sum to the global sum atomically
    if (tid_in_block == 0) {
        atomicAdd(d_total_dangling_sum, sdata[0]);
    }
}


// Main PageRank calculation kernel
__global__ void pagerank_kernel(const double* d_pr_old,      // Old PageRank values
                               double* d_pr_new,             // New PageRank values (output)
                               const int* d_flat_incoming_links, // Flattened list of source nodes for incoming links
                               const int* d_incoming_counts,   // Number of incoming links for each node
                               const int* d_out_degree,        // Out-degree of each node
                               const int* d_offsets,           // Offsets into d_flat_incoming_links for each node
                               const double* d_teleport_values, // Teleportation contribution for each node
                               double damping_factor,         // Damping factor
                               int num_nodes,                  // Total number of nodes
                               const double* d_iter_dangling_sum) { // Dangling sum for this iteration (device pointer)
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID, represents the current node to process

    if (i < num_nodes) {
        double sum_from_links = 0.0;
        // Iterate over all nodes J that link to node I
        for (int k = 0; k < d_incoming_counts[i]; k++) {
            int source_node_index_in_flat_array = d_offsets[i] + k;
            int source_node_J = d_flat_incoming_links[source_node_index_in_flat_array];

            if (d_out_degree[source_node_J] > 0) {
                sum_from_links += d_pr_old[source_node_J] / (double)d_out_degree[source_node_J];
            }
        }
        // PageRank formula: PR(I) = damping * (sum(PR(J)/OutDegree(J)) + DanglingSum) + (1-damping) * Teleport(I)
        d_pr_new[i] = damping_factor * (sum_from_links + (*d_iter_dangling_sum)) + (1.0 - damping_factor) * d_teleport_values[i];
    }
}

void run_pagerank_cuda(double* d_pr_old, double* d_pr_new,
                       int* d_flat_incoming_links, int* d_incoming_counts,
                       int* d_out_degree, int* d_offsets,
                       double* d_teleport_values, double damping_factor,
                       int num_nodes, int num_iterations) {

    // Define thread block and grid dimensions (1D)
    dim3 threadsPerBlock(THREADS_PER_BLOCK_X);
    dim3 numBlocks((num_nodes + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Allocate device memory for the dangling sum for each iteration
    double *d_iter_dangling_sum;
    CUDA_CHECK(cudaMalloc(&d_iter_dangling_sum, sizeof(double)));

    for (int iter = 0; iter < num_iterations; iter++) {
        // Copy current pr_new to pr_old for next iteration
        // pr_new from previous iteration (or initial) becomes pr_old for current
        CUDA_CHECK(cudaMemcpy(d_pr_old, d_pr_new, num_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

        // Reset dangling sum for this iteration to 0.0 on device
        CUDA_CHECK(cudaMemset(d_iter_dangling_sum, 0, sizeof(double)));

        // Calculate dangling_sum for the current iteration using d_pr_old
        sum_dangling_pageranks_kernel<<<numBlocks, threadsPerBlock>>>(
            d_pr_old, d_out_degree, d_iter_dangling_sum, num_nodes);
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure dangling sum calculation is complete

        // Calculate new PageRank values
        pagerank_kernel<<<numBlocks, threadsPerBlock>>>(
            d_pr_old, d_pr_new,
            d_flat_incoming_links, d_incoming_counts, d_out_degree, d_offsets,
            d_teleport_values, damping_factor, num_nodes, d_iter_dangling_sum);
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize()); 
        
    }

    // Free the iteration-specific dangling sum memory
    CUDA_CHECK(cudaFree(d_iter_dangling_sum));
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <graph_file> <num_nodes> <damping_factor> <num_iterations>\n", argv[0]);
        return 1;
    }
    char* graph_filename = argv[1];
    N_nodes = atoi(argv[2]);
    D_damping_factor = atof(argv[3]);
    NUM_iters = atoi(argv[4]);

    if (N_nodes <= 0 || D_damping_factor < 0.0 || D_damping_factor > 1.0 || NUM_iters <= 0) {
        fprintf(stderr, "Invalid input parameters.\n");
        return 1;
    }

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // 1. Allocate host memory
    allocate_host_memory();

    // 2. Read graph from file
    read_graph_data(graph_filename);
    printf("Graph reading complete. N_nodes = %d\n", N_nodes);


    // 3. Flatten incoming_links_host for GPU transfer
    // Calculate total number of incoming links across all nodes
    long long total_incoming_links = 0; // Use long long for very large graphs
    for (int i = 0; i < N_nodes; i++) {
        total_incoming_links += incoming_counts_host[i];
    }
    printf("Total incoming links: %lld\n", total_incoming_links);


    int* flat_incoming_links_host = (int*)malloc(total_incoming_links * sizeof(int));
    assert(flat_incoming_links_host != NULL);
    int* offsets_host = (int*)malloc(N_nodes * sizeof(int)); // Offsets for each node in the flattened array
    assert(offsets_host != NULL);

    int current_pos = 0;
    for (int i = 0; i < N_nodes; i++) {
        offsets_host[i] = current_pos;
        memcpy(flat_incoming_links_host + current_pos, incoming_links_host[i], incoming_counts_host[i] * sizeof(int));
        current_pos += incoming_counts_host[i];
    }
    assert(current_pos == total_incoming_links);


    // 4. Allocate memory on CUDA device
    double *d_pr_old, *d_pr_new_gpu, *d_teleport_values;
    int *d_flat_incoming_links, *d_incoming_counts, *d_out_degree, *d_offsets;

    CUDA_CHECK(cudaMalloc(&d_pr_old, N_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pr_new_gpu, N_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_teleport_values, N_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flat_incoming_links, total_incoming_links * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_incoming_counts, N_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_degree, N_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, N_nodes * sizeof(int)));

    // 5. Initialize PageRank and teleport values on host and copy to device
    double* teleport_host = (double*)malloc(N_nodes * sizeof(double));
    assert(teleport_host != NULL);
    double initial_pr_value = 1.0 / (double)N_nodes;
    for (int i = 0; i < N_nodes; i++) {
        pr_new_host[i] = initial_pr_value; 
        teleport_host[i] = initial_pr_value;   
    }

    CUDA_CHECK(cudaMemcpy(d_pr_new_gpu, pr_new_host, N_nodes * sizeof(double), cudaMemcpyHostToDevice)); // Initial PRs go to d_pr_new_gpu
    CUDA_CHECK(cudaMemcpy(d_teleport_values, teleport_host, N_nodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flat_incoming_links, flat_incoming_links_host, total_incoming_links * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_incoming_counts, incoming_counts_host, N_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_degree, out_degree_host, N_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets_host, N_nodes * sizeof(int), cudaMemcpyHostToDevice));

    // 6. Run PageRank calculation on CUDA
    printf("Starting PageRank iterations on GPU...\n");
    run_pagerank_cuda(d_pr_old, d_pr_new_gpu,
                      d_flat_incoming_links, d_incoming_counts, d_out_degree, d_offsets,
                      d_teleport_values, D_damping_factor, N_nodes, NUM_iters);
    printf("PageRank iterations complete.\n");

    // 7. Copy final PageRank results from device to host
    CUDA_CHECK(cudaMemcpy(pr_new_host, d_pr_new_gpu, N_nodes * sizeof(double), cudaMemcpyDeviceToHost));
    
    gettimeofday(&end_time, NULL);
    double total_time_sec = ((end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0);

    // 8. Print results (e.g., top PageRank, sum)
    printf("\nFinal PageRank (first few nodes and max):\n");

    int max_pagerank_node = 0;
    double max_pagerank_value = (N_nodes > 0) ? pr_new_host[0] : 0.0;
    double sum_pagerank = 0.0;

    if (N_nodes > 0) {
        for (int i = 0; i < N_nodes; i++) {
            if (pr_new_host[i] > max_pagerank_value) {
                max_pagerank_value = pr_new_host[i];
                max_pagerank_node = i;
            }
            sum_pagerank += pr_new_host[i];
        }
        printf("Max PageRank = %.8f at Node %d\n", max_pagerank_value, max_pagerank_node);
        printf("Sum of all PageRanks: %.8f (should be close to 1.0)\n", sum_pagerank);
    }


    printf("\nTotal execution time: %.4f seconds\n", total_time_sec);

    // 9. Cleanup host memory
    for (int i = 0; i < N_nodes; i++) {
        if (incoming_links_host[i] != NULL) {
            free(incoming_links_host[i]);
        }
    }
    free(incoming_links_host);
    free(incoming_counts_host);
    free(out_degree_host);
    free(pr_new_host);
    free(teleport_host);
    free(flat_incoming_links_host);
    free(offsets_host);

    // 10. Cleanup device memory
    CUDA_CHECK(cudaFree(d_pr_old));
    CUDA_CHECK(cudaFree(d_pr_new_gpu));
    CUDA_CHECK(cudaFree(d_teleport_values));
    CUDA_CHECK(cudaFree(d_flat_incoming_links));
    CUDA_CHECK(cudaFree(d_incoming_counts));
    CUDA_CHECK(cudaFree(d_out_degree));
    CUDA_CHECK(cudaFree(d_offsets));

    return 0;
}
