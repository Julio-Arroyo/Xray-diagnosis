#include "dataset.hpp"
#include "utils.hpp"

CheXpert::CheXpert(const std::string &datapath, torch::Device device) {
    DatasetPair data;
    getData(datapath, data);

    data.X.to(device);
    data.Y.to(device);

    std::cout << "Got data" << std::endl;
    std::cout << "- Inputs: " << std::endl;
    std::cout << "    - Sizes: " << data.X.sizes() << std::endl;
    std::cout << "    - Options: " << data.X.options() << std::endl;
    std::cout << std::endl;

    std::cout << "- Labels: " << std::endl;
    std::cout << "    - Sizes: " << data.Y.sizes() << std::endl;
    std::cout << "    - Options: " << data.Y.options() << std::endl;
    std::cout << std::endl;

    inputs = data.X;
    labels = data.Y;
}

torch::data::Example<> CheXpert::get(size_t index) {
    return {inputs[index], labels[index]};
}

torch::optional<size_t> CheXpert::size() const {
    std::cout << "in CheXpert::size():" << inputs.sizes()[0] << std::endl;
    return inputs.sizes()[0];  // BUG: not sure this does what I intend it to
}
