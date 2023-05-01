#include "scalarCNN.hpp"
#include "dataset.hpp"

const std::string DataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/PY_first60k.pt";
const int BatchSize = 64;
const int NumEpochs = 1;

int main() {
    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;

    auto dataset = CheXpert(DataPath, device).map(torch::data::transforms::Stack<>());

    std::cout << "dataset.size(): " << dataset.size() << std::endl;
    // torch::Tensor inputs = dataset.X;
    // torch::Tensor labels = dataset.Y;

    ScalarCNN model;

    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                            torch::data::DataLoaderOptions()
                                                        .batch_size(BatchSize)
                                                        .workers(2));

    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(2e-4)
                                                    .betas(std::make_tuple(0.5, 0.5)));

    // torch::nn::MSELoss criterion(torch::nn::MSELossOptions(torch::enumtype::kSum));

    model->to(device);

    // Todo restore from checkpoint

    for (int epoch = 1; epoch <= NumEpochs; ++epoch) {
        double running_loss = 0.0;
        int batch_index = 0;
        for (torch::data::Example<>& batch : *data_loader) {
            std::cout << "batch_index: " << ++batch_index << std::endl;

            torch::Tensor inputs = batch.data;
            torch::Tensor labels = batch.target;

            std::cout << "inputs options: " << inputs.options() << std::endl;
            torch::Tensor preds = model->forward(inputs);

            torch::nn::functional::MSELossFuncOptions MSEoptions(torch::kSum);
            auto loss = torch::nn::functional::mse_loss(preds, labels, MSEoptions);
            double batch_loss = loss.item<double>();
            running_loss += batch_loss;

            optim.zero_grad();
            loss.backward();
            optim.step();
        }

        std::printf(
            "\r[Epoch: %2ld/%2ld] Loss: %.4f",
            epoch,
            NumEpochs,
            ++batch_index,
            running_loss);
    }
}
