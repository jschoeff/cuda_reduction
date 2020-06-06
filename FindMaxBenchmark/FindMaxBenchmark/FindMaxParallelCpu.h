#pragma once
#include "FindMaxBase.h"
#include <algorithm>
#include <numeric>
#include <execution>

class FindMaxParallelCpu : public FindMaxBase
{
public:
  FindMaxParallelCpu(std::vector<int> data, std::string name)
    :FindMaxBase(data, name)
  {}

  virtual int run() override {
    // execution::par does not work with std::max_element in msvc
    return std::reduce(
      std::execution::par,
      _data.begin(),
      _data.end(),
      0,
      [](int a, int b) { return std::max(a, b); }
    );
  };
};

