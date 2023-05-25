#include <cassert>
#include <fstream>

// #include "scalarCNN.hpp"
#include "dataset.hpp"
#include "resnet.hpp"

const std::string TrainDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/train_classweighted.pt";
const std::string ValDataPath = "/groups/CS156b/2023/Xray-diagnosis/Cpp/data/val_classweighted.pt";
const int BatchSize = 128;
const int NumEpochs = 10;
const int NumResidualBlocksPerStack = 9;
const bool SkipConnections = true;
const int NumWorkers = 4;


int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./train <EXPERIMENT_NAME>" << std::endl;
        return 1;
    }

    const std::string ExperimentName = argv[1];

    torch::Device device(torch::kCUDA);
    std::cout << "Got device: " << device << std::endl;

    std::ofstream loss_history_file;
    loss_history_file.open("../logs/LOSS_" + ExperimentName + ".txt");
    loss_history_file << "TrainLoss,ValLoss,ValAcc,ValMSE\n";

    // Resnet18
    std::cout << "NumResidualBlocksPerStack: " << NumResidualBlocksPerStack << std::endl;
    std::vector<int> n_blocks = {NumResidualBlocksPerStack, NumResidualBlocksPerStack, NumResidualBlocksPerStack, NumResidualBlocksPerStack};
    std::vector<int> n_filters = {CONV1_FILTERS, 16, 32, LAST_STACK_FILTERS};  // {16, 32, 64};
    ResNet model(n_blocks,
                 n_filters,
                 true);

    int total_params = 0;
    for (const auto &p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "TOTAL PARAMETERS:" << total_params << std::endl;

    auto train_dataset = CheXpert(TrainDataPath).map(torch::data::transforms::Stack<>());
    auto val_dataset = CheXpert(ValDataPath).map(torch::data::transforms::Stack<>());

    const int TrainSetSize = train_dataset.size().value();
    const int NumTrainBatches = TrainSetSize/BatchSize + 1;
    const int ValSetSize = val_dataset.size().value();
    const int NumValBatches = ValSetSize/BatchSize + 1;
    std::cout << "Dataset sizes train/val: " << TrainSetSize << ", " << ValSetSize << std::endl;
    std::cout << "Number batches train/val: " << NumTrainBatches << ", " << NumValBatches << std::endl;

    auto train_loader = torch::data::make_data_loader(std::move(train_dataset),
                                                      torch::data::DataLoaderOptions()
                                                                  .batch_size(BatchSize)
                                                                  .workers(NumWorkers));
    auto val_loader = torch::data::make_data_loader(std::move(val_dataset),
                                                    torch::data::samplers::SequentialSampler(ValSetSize),
                                                    torch::data::DataLoaderOptions()
                                                                .batch_size(BatchSize)
                                                                .workers(NumWorkers));
  
    torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-6));

    model->to(device);

    auto criterion = torch::nn::CrossEntropyLoss();
    torch::nn::MSELossOptions MSEoptions(torch::kMean);
    auto MSEcriterion = torch::nn::MSELoss(MSEoptions);

    // Todo restore from checkpoint

    double train_loss = 0;
    double best_val_acc = 0;
    for (int epoch = 1; epoch <= NumEpochs; ++epoch) {
        for (torch::data::Example<>& train_batch : *train_loader) {            
            model->train();
            torch::Tensor train_inputs = train_batch.data.to(torch::kFloat32);
            torch::Tensor train_labels = train_batch.target.to(torch::kFloat32);

            train_inputs = train_inputs.to(device);
            train_labels = train_labels.to(device);

            torch::Tensor train_preds = model->forward(train_inputs);

            auto curr_train_loss = criterion(train_preds, train_labels);
            train_loss += curr_train_loss.item<double>();

            optim.zero_grad();
            curr_train_loss.backward();
            optim.step();
        }

        // torch::InferenceMode guard(true);  // DEBUG
        torch::NoGradGuard no_grad;
        model->eval();
        double val_loss = 0;
        double val_mse = 0;
        int val_num_correct = 0;
        for (torch::data::Example<>& val_batch : *val_loader) {
            torch::Tensor val_inputs = val_batch.data.to(torch::kFloat32);
            torch::Tensor val_labels = val_batch.target.to(torch::kFloat32);
            torch::Tensor val_classes = val_labels.argmax(1);

            val_inputs = val_inputs.to(device);
            val_labels = val_labels.to(device);
            val_classes = val_classes.to(device);

            torch::Tensor val_output = model->forward(val_inputs);
            torch::Tensor val_preds = val_output.argmax(1);

            // // Round preds
            // torch::Tensor pos_preds = torch::where(preds >= 0.5, 1, 0);
            // torch::Tensor neg_preds = torch::where(preds <= -0.5, -1, 0);
            // torch::Tensor rounded_preds = pos_preds + neg_preds;

            val_num_correct += val_preds.eq(val_classes).sum().item<int64_t>();

            auto curr_val_loss = criterion(val_output, val_labels);
            auto curr_val_mse = MSEcriterion(val_preds, val_classes);
            val_loss += curr_val_loss.item<double>();
            val_mse += curr_val_mse.item<double>();
        }

        train_loss = train_loss / NumTrainBatches;
        val_loss = val_loss / NumValBatches;
        val_mse = val_mse / NumValBatches;
        double val_accuracy = static_cast<double>(val_num_correct) / ValSetSize;

        std::printf(
            "\r[Epoch: %2ld] Train Loss: %.8f | Val Loss: %.8f | Val Acc: %.8f | Val MSE: %.8f\n",
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
            val_mse);
        loss_history_file << train_loss << "," << val_loss << "," << val_accuracy << "," << val_mse << "\n";

        if (val_accuracy > best_val_acc) {
            best_val_acc = val_accuracy;
            std::cout << "New best at epoch " << epoch << std::endl;
            std::string checkpoint_path = "../weights/" + ExperimentName + "_best_val.pt";
            torch::save(model, checkpoint_path);
        }

        std::string checkpoint_path = "../weights/" + ExperimentName + "_epoch" + std::to_string(epoch) + ".pt";
        torch::save(model, checkpoint_path);
    }
    loss_history_file.close();
}
