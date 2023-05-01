#include "dataset.hpp"
#include "utils.hpp"

CheXpert::CheXpert(const std::string &datapath, torch::Device device) {
    DatasetPair data;
    getData(datapath, data);

    data.X.to(device);
    data.Y.to(device);

    inputs = data.X;
    labels = data.Y;
}

torch::data::Example<> CheXpert::get(size_t index) {
    return {inputs[index], labels[index]};
}

torch::optional<size_t> CheXpert::size() const {
    return inputs.size(0);
}
