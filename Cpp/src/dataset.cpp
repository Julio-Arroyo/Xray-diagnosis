#include "dataset.hpp"

CheXpert::CheXpert(const std::string &datapath) {
    DatasetPair data;
    getData(datapath, data);

    std::cout << "In dataset constructor:" << std::endl;
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
    return inputs.sizes()[0];
}
