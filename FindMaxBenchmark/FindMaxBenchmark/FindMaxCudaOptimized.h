#pragma once

#include "FindMaxBase.h"
#include <algorithm>
#include <numeric>
#include <execution>



extern "C" int launchMaxCudaOptimized(std::vector<int> data);

class FindMaxCudaOptimized : public FindMaxBase
{
public:
  FindMaxCudaOptimized(std::vector<int> data, std::string name)
    :FindMaxBase(data, name)
  {}

  virtual int run() override {
    return launchMaxCudaOptimized(_data);
  };
};

