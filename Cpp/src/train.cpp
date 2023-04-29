#include "utils.hpp"

const std::string DataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/first60k.pt";

int main() {
    DatasetPair dataset;
    if (getInputsLabels(DataPath, dataset) == -1) {
        return -1;
    }
    torch::Tensor inputs = dataset.X;
    torch::Tensor labels = dataset.Y;

    std::cout << "Got dataset" << std::endl;
    std::cout << "- Inputs: " << std::endl;
    std::cout << "    - Sizes: " << inputs.sizes() << std::endl;
    std::cout << "    - Options: " << inputs.options() << std::endl;

    std::cout << "- Labels: " << std::endl;
    std::cout << "    - Sizes: " << labels.sizes() << std::endl;
    std::cout << "    - Options: " << labels.options() << std::endl;
}
