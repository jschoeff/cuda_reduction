#pragma once

#include "FindMaxBase.h"
#include <algorithm>
#include <numeric>
#include <execution>



extern "C" int launchMaxCudaNaive(std::vector<int> data);

class FindMaxCudaNaive : public FindMaxBase
{
public:
	FindMaxCudaNaive(std::vector<int> data, std::string name)
    :FindMaxBase(data, name)
  {}

  virtual int run() override {
    return launchMaxCudaNaive(_data);
  };
};

