# Parallelized-PageRank-Algorithm

This repository contains multiple implementations of the PageRank algorithm with a focus on performance optimization using various parallel programming models.

## Overview

PageRank is an algorithm used by search engines to rank web pages based on their importance. This project includes:

- A **serial** C implementation
- A **multi-threaded** version using OpenMP
- A **GPU-accelerated** version using CUDA

These implementations help benchmark the performance benefits and trade-offs of CPU and GPU parallelism.

## 📁 File Structure

| File                 | Description |
|----------------------|-------------|
| `pagerank.c`         | Serial C implementation of PageRank |
| `pagerank_openmp.c`  | OpenMP-based parallel CPU implementation |
| `pagerank_cuda.cu`   | CUDA implementation for GPU acceleration |

## 🛠️ Build Instructions

### 🔧 Requirements

- **For all**: GCC or Clang, Make
- **OpenMP version**: OpenMP-supporting compiler (`gcc` with `-fopenmp`)
- **CUDA version**: NVIDIA GPU with CUDA Toolkit

## 🧪 Compilation & Run

### 1. Serial Version

```bash
gcc -o pagerank pagerank.c -lm
./pagerank_serial input.txt num_nodes damping_factor iterations
```

### 2. OpenMP Version

```bash
gcc -fopenmp -o pagerank_openmp pagerank_openmp.c -lm  
./pagerank_omp input.txt num_nodes damping_factor iterations
```

### 3. CUDA Version

```bash
nvcc pagerank_cuda.cu -o pagerank_cuda
./pagerank_cuda input.txt num_nodes damping_factor iterations
```

> Make sure `input.txt` follows the expected graph input format. Typically:  
```
<src_node dst_node>
<src_node dst_node>
...
```

## 📈 Performance Metrics

| Implementation | Parallelism Model | Expected Speedup |
|----------------|-------------------|------------------|
| Serial         | None              | 1×               |
| OpenMP         | Shared Memory     | 3–6× (CPU-core dependent) |
| CUDA           | GPU (SIMT model)  | 10–100× (hardware dependent) |

> Benchmarks were performed on representative hardware with appropriately scaled graphs.

## 🧵 Notes

- Damping factor is usually set to **0.85**
- Convergence threshold can be adjusted in code
- CUDA kernel is optimized for coalesced memory access and parallel summation
- All versions use sparse graph representation via adjacency lists or edge lists

## 📚 References

- [PageRank - Wikipedia](https://en.wikipedia.org/wiki/PageRank)
- NVIDIA CUDA Documentation
- OpenMP Specification
