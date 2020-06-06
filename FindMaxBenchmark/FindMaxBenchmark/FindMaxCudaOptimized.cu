// Slight optimization of FindMaxCudaNaive.cu; uses pinned memory as well as sequential adressing

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

// Shared memory amount is configured in execution configuration
__global__ void reduceMaxOptimized(int* data) {
  // Shared mem size allocated dynamically using execution parameter
  extern __shared__ int partial_max[];

  // Calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load current element into shared memory
  partial_max[threadIdx.x] = data[tid];
  __syncthreads(); // Wait for each thread to load its element

  // Sequential adressing
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) { // No more modulo operations
      // Write higher value back
      partial_max[threadIdx.x] = partial_max[threadIdx.x] > partial_max[threadIdx.x + stride] ? partial_max[threadIdx.x] : partial_max[threadIdx.x + stride];
    }
    __syncthreads(); // Wait for other threads
  }

  // Thread 0 of each block write the result back
  if (threadIdx.x == 0) {
    data[blockIdx.x] = partial_max[0];
  }
}

extern "C" int launchMaxCudaOptimized(std::vector<int> data) {
  // Parameters
  int size = data.size();
  size_t n_bytes = size * sizeof(int);

  // Map vector data to cuda storage ->  + 50% performance gains
  int* cuda_data;
  cudaHostRegister(data.data(), n_bytes, cudaHostRegisterMapped);
  cudaHostGetDevicePointer(&cuda_data, data.data(), 0);

  int n_remaining = size;
  do {
    int cta_size = n_remaining < 1024 ? n_remaining : 1024; // Set cta as high as possible, max 1024
    int block_size = n_remaining / cta_size; // Calculate required blocks
    reduceMaxOptimized << <block_size, cta_size, cta_size * sizeof(int) >> > (cuda_data);
    n_remaining = block_size; // Each block equals one operation that still needs to be reduced
  } while (n_remaining > 1);

  // Copy result back
  int result;
  cudaMemcpy(&result, cuda_data, sizeof(int), cudaMemcpyDeviceToHost);

  return result;
}