#ifndef RESNET_HPP
#define RESNET_HPP

#include <torch/torch.h>

class ResidualBlockImpl : public torch::nn::Module {
private:
    int m_out_channels;
    int m_stride;
    bool m_skip_connections;

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;

    torch::nn::BatchNorm2d bn1;
    torch::nn::BatchNorm2d bn2;

    torch::nn::AvgPool2d downsample;

    torch::Tensor adjustDims(torch::Tensor x);

public:
    ResidualBlockImpl(int out_channels, int stride=1, bool skip_connections=false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResidualBlock);

class ResNetImpl : public torch::nn::Module {
private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential stack1, stack2, stack3;
    torch::nn::AvgPool2d avg_pool;
    torch::nn::Linear fc;
    torch::nn::Tanh tanh;

public:
    ResNetImpl(int nblocks_per_stack, bool skip_connections=false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNet);

#endif