#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <string>

struct DatasetPair {
    torch::Tensor X;
    torch::Tensor Y;
};

int getInputsLabels(const std::string &dataset_path, DatasetPair &ds);

#endif