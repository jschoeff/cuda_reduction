#pragma once
#include <vector>
#include <string>

class FindMaxBase {
public:
  FindMaxBase(std::vector<int> data, std::string name)
    :_data(data), _name(name)
  {}

  virtual int run() = 0; // Abstract implementation
  std::string getName() { return _name; };

protected:
  std::vector<int> _data;
  std::string _name;
};