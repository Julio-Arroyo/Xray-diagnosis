#include "dataset.hpp"
#include "utils.hpp"

CheXpert::CheXpert(const std::string &datapath, torch::Device device) {
    DatasetPair data;
    getData(datapath, data);

    std::cout << "Got data" << std::endl;
    std::cout << "- Inputs: " << std::endl;
    std::cout << "    - Sizes: " << data.X.sizes() << std::endl;
    std::cout << "    - Options: " << data.X.options() << std::endl;
    std::cout << std::endl;

    std::cout << "- Labels: " << std::endl;
    std::cout << "    - Sizes: " << data.Y.sizes() << std::endl;
    std::cout << "    - Options: " << data.Y.options() << std::endl;
    std::cout << std::endl;

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
