#pragma once
#include "FindMaxBase.h"
#include <algorithm>
#include <execution>

class FindMaxSequential : public FindMaxBase
{
public:
  FindMaxSequential(std::vector<int> data, std::string name)
    :FindMaxBase(data, name)
  {}

  virtual int run() override { return *std::max_element(std::begin(_data), std::end(_data)); };
};

