#include "scalarCNN.hpp"

ScalarCNNImpl::ScalarCNNImpl() :
    conv1(torch::nn::Conv2dOptions(1, 16, 3).padding(1)),
    conv2(torch::nn::Conv2dOptions(16, 16, 3).padding(1)),
    conv3(torch::nn::Conv2dOptions(16, 16, 3).padding(1)),
    pool1(2),
    pool2(2),
    pool3(2),
    fc1(16*32*32, 128),
    fc2(128, 1),
    tanh()
{
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("pool1", pool1);
    register_module("pool2", pool2);
    register_module("pool3", pool3);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("tanh", tanh);
}

torch::Tensor ScalarCNNImpl::forward(torch::Tensor x)
{
    x = conv1(x);
    x = torch::nn::functional::relu(x);
    x = pool1(x);

    x = conv2(x);
    x = torch::nn::functional::relu(x);
    x = pool2(x);

    x = conv3(x);
    x = torch::nn::functional::relu(x);
    x = pool3(x);

    std::cout << "x before view: " << x.sizes() << std::endl;
    x = x.view({-1, 16*32*32});
    std::cout << "x after view: " << x.sizes() << std::endl;

    x = fc1(x);
    x = torch::nn::functional::relu(x);

    x = fc2(x);
    x = tanh(x);

    x = x.squeeze();

    return x;
}
