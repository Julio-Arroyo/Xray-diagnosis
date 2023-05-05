#ifndef RESNET_HPP
#define RESNET_HPP

#include <torch/torch.h>
#include <vector>

// #define N_STACKS 4;  // Resnet18 and 34 both have four stacks of residual

const int N_INPUT_CHANNELS = 1;  // CheXpert is grayscale

const int CONV1_FILTERS = 16;
const int CONV1_KERNEL = 7;
const int CONV1_STRIDE = 2;

const int MAX_POOL_KERNEL = 3;
const int MAX_POOL_STRIDE = 2;

// last stack's output is 7x7, so avg_pool turns it into 1x1
const int AVG_POOL_KERNEL = 7;

/*
    Implements a 2-layer residual block (Fig. 2)
*/
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
    ResidualBlockImpl(int out_channels, int stride=1, bool skip_connections=true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResidualBlock);

/*
    Implements Resnet18 and Resnet34.
*/
class ResNetImpl : public torch::nn::Module {
private:
    int num_stacks;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::MaxPool2d max_pool;

    std::vector<torch::nn::Sequential> stacks;
    // torch::nn::Sequential conv2_x, conv3_x, conv4_x, conv5_x;  // stacks of residual blocks

    torch::nn::AvgPool2d avg_pool;
    torch::nn::Linear fc;
    torch::nn::Tanh tanh;

public:
    ResNetImpl(const std::vector<int> &n_blocks,   // n_blocks[i] = # residual blocks in i-th stack
               const std::vector<int> &n_filters,  // n_filters[i] = # filters (out_channels) in i-th stack 
               bool skip_connections=true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNet);

#endif