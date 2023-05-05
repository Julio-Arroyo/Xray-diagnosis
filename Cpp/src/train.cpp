#include <cassert>
#include <fstream>

// #include "scalarCNN.hpp"
#include "dataset.hpp"
#include "resnet.hpp"

const std::string TrainDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_train.pt";
const std::string ValDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_val.pt";
const int BatchSize = 128;
const int NumEpochs = 10;
const int NumResidualBlocksPerStack = 6;
const bool SkipConnections = true;
const int LoggingInterval = 100;  // record loss every LoggingInterval iterations
const std::string ExperimentName = "First_Resnet18";
const int NumWorkers = 4;

int main() {
    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;

    std::ofstream loss_history_file;
    loss_history_file.open("../logs/LOSS_" + ExperimentName + ".txt");
    loss_history_file << "Train,Val\n";

    // Resnet18
    std::vector<int> n_blocks = {2, 2, 2, 2};
    std::vector<int> n_filters = {64, 128, 256, 512};
    ResNet model(n_blocks,
                    n_filters,
                    true);
    torch::Tensor x = torch::randn({BatchSize, 1, 224, 224});
    model(x);

    auto train_dataset = CheXpert(TrainDataPath).map(torch::data::transforms::Stack<>());
    auto val_dataset = CheXpert(ValDataPath).map(torch::data::transforms::Stack<>());
    const int TrainSetSize = train_dataset.size().value();
    const int ValSetSize = val_dataset.size().value();
    std::cout << "Dataset sizes train/val: " << TrainSetSize << ", " << ValSetSize << std::endl;

    // ScalarCNN model;
    // ResNet model(NumResidualBlocksPerStack, SkipConnections);

    auto train_loader = torch::data::make_data_loader(std::move(train_dataset),
                                                      torch::data::DataLoaderOptions()
                                                                  .batch_size(BatchSize)
                                                                  .workers(NumWorkers));
    auto val_loader = torch::data::make_data_loader(std::move(val_dataset),
                                                    torch::data::samplers::SequentialSampler(ValSetSize),
                                                    torch::data::DataLoaderOptions()
                                                                .batch_size(BatchSize)
                                                                .workers(NumWorkers));
  
    torch::optim::Adam optim(model->parameters(),
                             torch::optim::AdamOptions(2e-4)
                                            .betas(std::make_tuple(0.5, 0.5)));

    model->to(device);

    torch::nn::functional::MSELossFuncOptions MSEoptions(torch::kMean);

    // Todo restore from checkpoint

    // model->eval();
    // double val_loss = 0.0;
    // for (torch::data::Example<>& batch : *val_loader) {
    //     torch::Tensor inputs = batch.data.to(device);
    //     torch::Tensor labels = batch.target.to(device);

    //     torch::Tensor preds = model->forward(inputs);

    //     auto loss = torch::nn::functional::mse_loss(preds, labels, MSEoptions);
    //     val_loss += loss.item<double>();
    // }
    // std::printf("Before training: Val Loss %.4f", val_loss/ValSetSize);
    // assert(false);

    int iter = 0;
    double train_batch_loss = 0;
    double val_batch_loss = 0;
    for (int epoch = 1; epoch <= NumEpochs; ++epoch) {
        model->train();
        for (torch::data::Example<>& batch : *train_loader) {
            torch::Tensor inputs = batch.data.to(device);
            torch::Tensor labels = batch.target.to(device);

            torch::Tensor preds = model->forward(inputs);

            auto loss = torch::nn::functional::mse_loss(preds, labels, MSEoptions);
            train_batch_loss += loss.item<double>();

            optim.zero_grad();
            loss.backward();
            optim.step();
            iter++;
        }

        if (iter % LoggingInterval == 0) {
            model->eval();
            int n_batches_valset = 0;
            for (torch::data::Example<>& batch : *val_loader) {
                torch::Tensor inputs = batch.data.to(device);
                torch::Tensor labels = batch.target.to(device);

                torch::Tensor preds = model->forward(inputs);

                auto loss = torch::nn::functional::mse_loss(preds, labels, MSEoptions);
                val_batch_loss += loss.item<double>();
                n_batches_valset++;
            }

            std::cout << "DEBUG: n_batches_valset " << n_batches_valset << std::endl;
            std::printf(
                "\r[Iter: %2ld] Train Loss: %.4f | Val Loss: %.4f",
                iter,
                train_batch_loss/LoggingInterval,
                val_batch_loss/n_batches_valset);
            loss_history_file << train_batch_loss/LoggingInterval << ","<< val_batch_loss/n_batches_valset << "\n";

            train_batch_loss = 0;
            val_batch_loss = 0;
        }

        std::string checkpoint_path = "../weights/resnet18_epoch" + std::to_string(epoch) + ".pt";
        torch::save(model, checkpoint_path);
    }
}
