#include "scalarCNN.hpp"
#include "dataset.hpp"

const std::string DataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/first60k.pt";
const int BatchSize = 64;
const int NumEpochs = 1;

int main() {
    torch::Device device(torch::kCUDA);

    auto dataset = CheXpert(DataPath, device).map(torch::data::transforms::Stack<>());

    std::cout << "dataset.size(): " << dataset.size() << std::endl;
    // torch::Tensor inputs = dataset.X;
    // torch::Tensor labels = dataset.Y;

    // std::cout << "Got dataset" << std::endl;
    // std::cout << "- Inputs: " << std::endl;
    // std::cout << "    - Sizes: " << inputs.sizes() << std::endl;
    // std::cout << "    - Options: " << inputs.options() << std::endl;
    // std::cout << std::endl;

    // std::cout << "- Labels: " << std::endl;
    // std::cout << "    - Sizes: " << labels.sizes() << std::endl;
    // std::cout << "    - Options: " << labels.options() << std::endl;
    // std::cout << std::endl;

    ScalarCNN model;

    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                            torch::data::DataLoaderOptions()
                                                        .batch_size(BatchSize)
                                                        .workers(2));

    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(2e-4)
                                                    .betas(std::make_tuple(0.5, 0.5)));

    model->to(device);

    for (int epoch = 1; epoch < NumEpochs; ++epoch) {
        for (torch::data::Example<>& batch : *data_loader) {
            std::cout << "Batch sizes: " << batch.data.sizes() << std::endl;
            std::cout << "Labels sizes: " << batch.target.sizes() << std::endl;
            std::cout << std::endl;
        }
    }
}
