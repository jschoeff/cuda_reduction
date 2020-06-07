#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_runtime.h"

#include "FindMaxSequential.h"
#include "FindMaxParallelCpu.h"
#include "FindMaxCudaNaive.h"
#include "FindMaxCudaOptimized.h"


int main() {
  // Define size
  int size = 1 << 24;

  // Init rng
  std::random_device rd;
  std::default_random_engine rand_gen(rd());
  std::uniform_int_distribution<long long unsigned> dist(0, 0xFFFFFFFFFFFFFFFF);

  // Generate vector data
  std::vector<int> data(size);
  std::generate(begin(data), end(data), [&dist, &rand_gen]() { return dist(rand_gen); }); // This is quite slow, alternatively use rand()

  // Create all implementations
  std::vector<std::unique_ptr<FindMaxBase>> implementations;
  implementations.push_back(std::make_unique<FindMaxSequential>(data, "Sequenziell"));
  implementations.push_back(std::make_unique<FindMaxParallelCpu>(data, "Parallel mit CPU"));
  implementations.push_back(std::make_unique<FindMaxCudaNaive>(data, "Parallel mit CUDA - nicht initialisiert"));
  implementations.push_back(std::make_unique<FindMaxCudaNaive>(data, "Parallel mit CUDA"));
  implementations.push_back(std::make_unique<FindMaxCudaOptimized>(data, "Parallele CUDA - optimiert"));

  // Optianllay inialize CUDA context with arbitrary CUDA call
  //cudaFree(0);

  // Run each implementation and measure runtime
  std::cout << "Starte Benchmark mit size = " << size << "\n";
  for (auto& implementation : implementations) {
    auto begin = std::chrono::high_resolution_clock::now();

    int max = implementation->run();

    // Measure time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = end - begin;

    // Output
    std::cout << implementation->getName() << ": " << runtime.count() << "ms (max=" << max << ")\n";
  }
  return 0;
}