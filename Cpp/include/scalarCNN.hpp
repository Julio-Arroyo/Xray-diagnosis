#ifndef CNN_SCALAR_HPP
#define CNN_SCALAR_HPP

#include <torch/torch.h>

class ScalarCNNImpl : public torch::nn::Module {
public:
    ScalarCNNImpl();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;

    torch::nn::MaxPool2d pool1;
    torch::nn::MaxPool2d pool2;
    torch::nn::MaxPool2d pool3;

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;

    torch::nn::Tanh tanh;
};
TORCH_MODULE(ScalarCNN);

#endif
