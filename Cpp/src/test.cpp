#include <torch/torch.h>
#include <string>
#include <fstream>
#include "resnet.hpp"

const std::string TrainedModelPath = "../weights/resnet18.pt";
const std::string TestImagesPath = "TODO";

int main() {
    std::ofstream preds_file;
    preds_file.open("../TODO_NAME_ME.txt");

    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;
    
    std::vector<int> n_blocks = {2, 2, 2, 2};
    std::vector<int> n_filters = {64, 128, 256, 512};
    ResNet resnet18(n_blocks,
                    n_filters,
                    true);
    torch::load(resnet18, TrainedModelPath);

    resnet18->to(device);
    
    auto test_dataset = CheXpert(TestImagesPath).map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(std::move(val_dataset),
                                                     torch::data::DataLoaderOptions()
                                                                    .batch_size(BatchSize)
                                                                    .workers(2));

    model->eval();
    for (torch::data::Example<>& batch : *test_loader) {
        torch::Tensor inputs = batch.data.to(device);

        torch::Tensor preds = model->forward(inputs);

        for (int i = 0; i < preds.sizes()[0]; i++) {
            preds_file << preds[i].item() << "\n";
        }
    }

    preds_file.close();

    return 0;
}
