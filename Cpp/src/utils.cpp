#include "utils.hpp"


int getInputsLabels(const std::string &dataset_path, DatasetPair &ds) {
    torch::jit::script::Module dataset;
    try {
        dataset = torch::jit::load(dataset_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    for (const auto& buff : dataset.named_buffers()) {
        std::string name = buff.name;
        torch::Tensor t = buff.value;
        if (name == "inputs") {
            std::cout << "FOUND INPUTS" << std::endl;
            ds.X = t;
        } else if (name == "labels") {
            std::cout << "FOUND LABELS" << std::endl;
            ds.Y = t;
        }
        std::cout << name << "," << std::endl;
        std::cout << "Sizes: " << t.sizes() << std::endl;
        std::cout << "Options: " << t.options() << std::endl;
    }

    return 0;
}
