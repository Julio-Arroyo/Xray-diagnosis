#include <torch/torch.h>
#include <string>
#include <fstream>
#include "resnet.hpp"
#include "dataset.hpp"

const std::string TrainedModelPath = "../weights/Big_Resnet18_best_val.pt";
const std::string TestImagesPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/test_images.pt";
const int BatchSize = 128;
const int NumWorkers = 4;

int main() {
    std::ofstream preds_file;
    preds_file.open("/groups/CS156b/2023/Xray-diagnosis/Cpp/logs/Big_Resnet18_FindingNoFinding.txt");

    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;
    
    std::vector<int> n_blocks = {2, 2, 2, 2};
    std::vector<int> n_filters = {64, 128, 256, 512};
    ResNet model(n_blocks,
                    n_filters,
                    true);

    torch::Tensor x = torch::randn({BatchSize, 1, 224, 224});
    model(x);

    torch::load(model, TrainedModelPath);

    model->to(device);
    
    auto test_dataset = CheXpert(TestImagesPath).map(torch::data::transforms::Stack<>());
    const int test_set_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset),
                                                    torch::data::samplers::SequentialSampler(test_set_size),
                                                    torch::data::DataLoaderOptions()
                                                                .batch_size(BatchSize)
                                                                .workers(NumWorkers));

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
