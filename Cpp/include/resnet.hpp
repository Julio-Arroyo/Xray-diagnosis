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

#endif