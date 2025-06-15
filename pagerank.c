#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>

#define MAX_LINE 1024

// Global graph + PageRank data
int N;                // Number of nodes
int** incoming_links; // incoming_links[i] = array of nodes pointing to i
int* incoming_counts; // number of incoming links per node
int* out_degree;      // number of outgoing links per node

double* pr_old;       // PR at t
double* pr_new;       // PR at t+1
double* teleport;     // teleportation vector (usually uniform)

int num_iters;
double d; // damping factor

// Allocate memory
void allocate_memory() {
    incoming_links = (int**)malloc(N * sizeof(int*));
    incoming_counts = (int*)calloc(N, sizeof(int));
    out_degree = (int*)calloc(N, sizeof(int));

    pr_old = (double*)calloc(N, sizeof(double));
    pr_new = (double*)calloc(N, sizeof(double));
    teleport = (double*)calloc(N, sizeof(double));
}

// Read the graph from file in (from -> to) format
void read_graph(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening graph file");
        exit(1);
    }

    int from, to;
    char line[MAX_LINE];

    // First pass: count in-degree and out-degree
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%d\t%d", &from, &to) == 2) {
            out_degree[from]++;
            incoming_counts[to]++;
        }
    }

    // Allocate incoming link arrays
    for (int i = 0; i < N; i++) {
        incoming_links[i] = (int*)malloc(incoming_counts[i] * sizeof(int));
        incoming_counts[i] = 0; 
    }

    // Second pass: populate incoming links
    rewind(fp);
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%d\t%d", &from, &to) == 2) {
            incoming_links[to][incoming_counts[to]++] = from;
        }
    }

    fclose(fp);
    printf("Finished reading graph.\n");
}

// Initialize PR vectors
void initialize_pagerank() {
    for (int i = 0; i < N; i++) {
        pr_new[i] = 1.0 / N;
        teleport[i] = 1.0 / N;
    }
}

// Main PageRank algorithm
void pagerank() {
    for (int iter = 0; iter < num_iters; iter++) {
        double dangling_sum = 0.0;

        // Swap and reset
        for (int i = 0; i < N; i++) {
            pr_old[i] = pr_new[i];
            pr_new[i] = 0.0;
        }

        // Sum of PageRank from dangling nodes
        for (int i = 0; i < N; i++) {
            if (out_degree[i] == 0)
                dangling_sum += pr_old[i] / N;
        }

        // Contribution from incoming links
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < incoming_counts[i]; j++) {
                int src = incoming_links[i][j];
                pr_new[i] += pr_old[src] / out_degree[src];
            }
        }

        // Final PR update with damping and teleport
        for (int i = 0; i < N; i++) {
            pr_new[i] = d * (pr_new[i] + dangling_sum) + (1 - d) * teleport[i];
        }

        //printf("Iteration %d complete.\n", iter + 1);
    }
}

void free_memory() {
    for (int i = 0; i < N; i++) {
        free(incoming_links[i]);
    }
    free(incoming_links);
    free(incoming_counts);
    free(out_degree);
    free(pr_old);
    free(pr_new);
    free(teleport);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <graph_file> <num_nodes> <damping> <num_iterations>\n", argv[0]);
        return 1;
    }

    char* filename = argv[1];
    N = atoi(argv[2]);
    d = atof(argv[3]);
    num_iters = atoi(argv[4]);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    allocate_memory();
    read_graph(filename);
    initialize_pagerank();
    pagerank();

    gettimeofday(&end, NULL);
    double total_time = ((end.tv_sec - start.tv_sec) * 1000 +
                         (end.tv_usec - start.tv_usec) / 1000.0) / 1000.0;


    printf("\nTotal time: %.4f seconds\n", total_time);
    printf("Total iterations: %d\n", num_iters);
    printf("Node 1: %.6f\n", pr_new[1]);

    int max_node = 0;
    double max_value = pr_new[0];
    for (int i = 1; i < N; i++) {
        if (pr_new[i] > max_value) {
            max_value = pr_new[i];
            max_node = i;
        }
    }
    printf("Max PageRank = %.6f at Node %d\n", max_value, max_node);

    double sum = 0;
    for (int i = 0; i < N; i++) sum += pr_new[i];
    printf("Sum of PageRank: %.6f\n", sum);


    free_memory();
    return 0;
}