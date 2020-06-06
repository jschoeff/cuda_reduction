// Naive implementation of parallel reduction using CUDA using interleaved adressing; memory gets copied to the device

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

// Shared memory amount is configured in execution configuration
__global__ void reduceMaxNaive(int* data) {
  // Shared mem size allocated dynamically using execution parameter
  extern __shared__ int partial_max[];

  // Calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load current element into shared memory
  partial_max[threadIdx.x] = data[tid];
  __syncthreads(); // Wait for each thread to load its element

  // Interleaved adressing
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (threadIdx.x % (2 * stride) == 0) { // Modulo operation is quite slow on the GPU
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

extern "C" int launchMaxCudaNaive(std::vector<int> data) {
  // Parameters
  int size = data.size();
  size_t n_bytes = size * sizeof(int);

  // Old implementation; allocate cuda memory, then copy vector data over
  int* cuda_data;
  cudaMalloc(&cuda_data, n_bytes);
  cudaMemcpy(cuda_data, data.data(), n_bytes, cudaMemcpyHostToDevice);

  int n_remaining = size;
  do {
    int cta_size = n_remaining < 1024 ? n_remaining : 1024; // Set cta as high as possible, max 1024
    int block_size = n_remaining / cta_size; // Calculate required blocks
    reduceMaxNaive << <block_size, cta_size, cta_size * sizeof(int) >> > (cuda_data);
    n_remaining = block_size; // Each block equals one operation that still needs to be reduced
  } while (n_remaining > 1);

  // Copy result back
  int result;
  cudaMemcpy(&result, cuda_data, sizeof(int), cudaMemcpyDeviceToHost);

  return result;
}