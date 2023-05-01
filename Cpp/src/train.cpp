#include "scalarCNN.hpp"
#include "dataset.hpp"

const std::string TrainDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/PY_first60k_train.pt";
const std::string ValDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/PY_first60k_val.pt";
const int BatchSize = 64;
const int NumEpochs = 10;

int main() {
    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;

    auto train_dataset = CheXpert(TrainDataPath).map(torch::data::transforms::Stack<>());
    auto val_dataset = CheXpert(ValDataPath).map(torch::data::transforms::Stack<>());
    const int TrainSetSize = train_dataset.size().value();
    const int ValSetSize = val_dataset.size().value();
    std::cout << "Dataset sizes train/val: " << TrainSetSize << ", " << ValSetSize << std::endl;

    ScalarCNN model;

    auto train_loader = torch::data::make_data_loader(std::move(train_dataset),
                                                      torch::data::DataLoaderOptions()
                                                                  .batch_size(BatchSize)
                                                                  .workers(2));
    auto val_loader = torch::data::make_data_loader(std::move(val_dataset),
                                                    torch::data::DataLoaderOptions()
                                                                .batch_size(BatchSize)
                                                                .workers(2));
  
    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(2e-4)
                                                    .betas(std::make_tuple(0.5, 0.5)));

    model->to(device);

    // Todo restore from checkpoint

    for (int epoch = 1; epoch <= NumEpochs; ++epoch) {
        model->train();
        double running_loss = 0.0;
        for (torch::data::Example<>& batch : *train_loader) {
            torch::Tensor inputs = batch.data.to(device);
            torch::Tensor labels = batch.target.to(device);

            torch::Tensor preds = model->forward(inputs);

            torch::nn::functional::MSELossFuncOptions MSEoptions(torch::kSum);
            auto loss = torch::nn::functional::mse_loss(preds, labels, MSEoptions);
            double batch_loss = loss.item<double>();
            running_loss += batch_loss;

            optim.zero_grad();
            loss.backward();
            optim.step();
        }

        model->eval();
        double val_loss = 0.0;
        for (torch::data::Example<>& batch : *val_loader) {
            torch::Tensor inputs = batch.data.to(device);
            torch::Tensor labels = batch.target.to(device);

            torch::Tensor preds = model->forward(inputs);

            torch::nn::functional::MSELossFuncOptions MSEoptions(torch::kSum);
            auto loss = torch::nn::functional::mse_loss(preds, labels, MSEoptions);
            val_loss += loss.item<double>();
        }

        std::printf(
            "\r[Epoch: %2ld/%2ld] Train Loss: %.4f",
            epoch,
            NumEpochs,
            running_loss/TrainSetSize,
            val_loss/ValSetSize);
    }
}
