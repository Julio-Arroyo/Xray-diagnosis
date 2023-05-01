#ifndef DATASET_HPP
#define DATASET_HPP

#include <torch/torch.h>

class CheXpert : public torch::data::datasets::Dataset<CheXpert> {
private:
    torch::Tensor inputs, labels;

public:
    explicit CheXpert(const std::string& datapath);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

#endif
