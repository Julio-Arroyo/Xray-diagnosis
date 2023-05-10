#include <cassert>
#include <fstream>

// #include "scalarCNN.hpp"
#include "dataset.hpp"
#include "resnet.hpp"

const std::string TrainDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_train_multiclass.pt";
const std::string ValDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/entire_val_multiclass.pt";
const int BatchSize = 128;
const int NumEpochs = 10;
// const int NumResidualBlocksPerStack = 6;
const bool SkipConnections = true;
const int LoggingInterval = 100;  // record loss every LoggingInterval iterations
const std::string ExperimentName = "CustomResnet18_lr1e-4";
const int NumWorkers = 4;

// double evaluate(ResNet &model, auto &val_loader, const torch::Device &device,
//                 const int ValSetSize, const double train_loss, const int iter) {

// }

int main() {
    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;

    std::ofstream loss_history_file;
    loss_history_file.open("../logs/LOSS_" + ExperimentName + ".txt");
    loss_history_file << "Train,Val\n";

    // Resnet18
    std::vector<int> n_blocks = {2, 2, 2, 2};
    std::vector<int> n_filters = {CONV1_FILTERS, 32, 64, LAST_STACK_FILTERS};  // {16, 32, 64, 128};
    ResNet model(n_blocks,
                 n_filters,
                 true);
    torch::Tensor x = torch::randn({BatchSize, 1, 224, 224});
    model(x);
    int total_params = 0;
    for (const auto &p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "TOTAL PARAMETERS:" << total_params << std::endl;

    auto train_dataset = CheXpert(TrainDataPath).map(torch::data::transforms::Stack<>());
    auto val_dataset = CheXpert(ValDataPath).map(torch::data::transforms::Stack<>());
    const int TrainSetSize = train_dataset.size().value();
    const int ValSetSize = val_dataset.size().value();
    std::cout << "Dataset sizes train/val: " << TrainSetSize << ", " << ValSetSize << std::endl;

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
                             torch::optim::AdamOptions(1e-4));

    model->to(device);

    // TODO: use cross entropy
    auto criterion = torch::nn::CrossEntropyLoss();
    // torch::nn::functional::MSELossFuncOptions MSEoptions(torch::kMean);

    // Todo restore from checkpoint

    int iter = 0;
    double train_loss = 0;
    double best_val_loss = 9999999;
    for (int epoch = 1; epoch <= NumEpochs; ++epoch) {
        std::cout << "EPOCH #" << epoch << std::endl;
        for (torch::data::Example<>& batch : *train_loader) {
            if (((iter % LoggingInterval) == 0)) {
                model->eval();
                double val_loss = 0;
                int num_val_batches = 0;
                for (torch::data::Example<>& batch : *val_loader) {
                    torch::Tensor inputs = batch.data.to(device);
                    torch::Tensor labels = batch.target.view({labels.size(0), 1}).to(device);

                    torch::Tensor preds = model->forward(inputs);

                    // // Round preds
                    // torch::Tensor pos_preds = torch::where(preds >= 0.5, 1, 0);
                    // torch::Tensor neg_preds = torch::where(preds <= -0.5, -1, 0);
                    // torch::Tensor rounded_preds = pos_preds + neg_preds;

                    auto loss = criterion(preds, labels);
                    val_loss += loss.item<double>();
                    num_val_batches++;
                }

                train_loss = train_loss / LoggingInterval;
                val_loss = val_loss / num_val_batches;

                std::printf(
                    "\r[Iter: %2ld] Train Loss: %.4f | Val Loss: %.4f\n",
                    iter,
                    train_loss,
                    val_loss);
                loss_history_file << train_loss << "," << val_loss << "\n";

                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    std::cout << "New best at epoch " << epoch << " and iter " << iter << std::endl;
                    std::string checkpoint_path = "../weights/" + ExperimentName + "_best_val.pt";
                    torch::save(model, checkpoint_path);
                }

                train_loss = 0; // reset
            }
            
            model->train();
            torch::Tensor inputs = batch.data.to(device);
            torch::Tensor labels = batch.target.to(device);

            torch::Tensor preds = model->forward(inputs);

            auto loss = torch::nn::functional::mse_loss(preds,
                                                        labels.view({labels.size(0), 1}),
                                                        MSEoptions);
            train_loss += ((loss.item<double>()) * inputs.size(0));

            optim.zero_grad();
            loss.backward();
            optim.step();
            iter++;
        }

        std::string checkpoint_path = "../weights/" + ExperimentName + "_epoch" + std::to_string(epoch) + ".pt";
        torch::save(model, checkpoint_path);
    }
    loss_history_file.close();
}
