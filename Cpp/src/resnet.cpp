#include <cassert>
#include <string>
#include "resnet.hpp"

/* RESIDUAL BLOCK */
ResidualBlockImpl::ResidualBlockImpl(int out_channels, int stride, bool skip_connections) :
    conv1(torch::nn::Conv2dOptions(out_channels/stride, out_channels, 3)
                            .padding(1)
                            .bias(false)
                            .stride(stride)),
    conv2(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                            .padding(1)
                            .bias(false)
                            .stride(1)),
    bn1(torch::nn::BatchNorm2dOptions(out_channels)),
    bn2(torch::nn::BatchNorm2dOptions(out_channels)),
    downsample(torch::nn::AvgPool2dOptions(1).stride(2))
{
    m_out_channels = out_channels;
    m_stride = stride;
    m_skip_connections = skip_connections;

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("downsample", downsample);
}

torch::Tensor ResidualBlockImpl::adjustDims(torch::Tensor x) {
    torch::Tensor x_halved_width_height = downsample(x);
    torch::Tensor extra_channels = torch::zeros_like(x_halved_width_height);
    return torch::cat({x_halved_width_height, extra_channels}, 1);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor out = torch::nn::functional::relu(bn1(conv1(x)));
    out = bn2(conv2(out));

    if (m_skip_connections) {
        torch::Tensor identity;
        if (out.sizes() == x.sizes()) {
            identity = x;
        } else {
            identity = adjustDims(x);
        }

        out = out + identity;
    }

    out = torch::nn::functional::relu(out);

    return out;
}

/* RESIDUAL NETWORK */
ResNetImpl::ResNetImpl(const std::vector<int> &n_blocks,
                       const std::vector<int> &n_filters,
                       bool skip_connections) :
    conv1(torch::nn::Conv2dOptions(N_INPUT_CHANNELS, CONV1_FILTERS, CONV1_KERNEL)
                        .stride(CONV1_STRIDE)
                        .padding(1)
                        .bias(false)),
    bn1(torch::nn::BatchNorm2dOptions(CONV1_FILTERS).track_running_stats(true)),
    max_pool(torch::nn::MaxPool2dOptions(MAX_POOL_KERNEL).stride(MAX_POOL_STRIDE)),

    avg_pool(torch::nn::AvgPool2dOptions(AVG_POOL_KERNEL).stride(1)),
    fc(128, 1),
    tanh()
{
    assert(n_blocks.size() == n_filters.size());
    num_stacks = n_blocks.size();

    for (int i = 0; i < num_stacks; i++) {
        torch::nn::Sequential curr_stack;

        // Add Residual Blocks
        for (int j = 0; j < n_blocks[i]; j++) {
            int stride;
            if (j == 0 && i > 0) {
                // first block of each stack (except first stack) has downsampling
                stride = 2;
            } else {
                stride = 1;
            }

            curr_stack->push_back(ResidualBlock(n_filters[i], stride, skip_connections));
        }

        stacks.push_back(curr_stack);

        // // DEBUG
        // int total_params = 0;
        // for (const auto &p : curr_stack.parameters()) {
        //     total_params += p.numel();
        // }
        // std::cout << "Stack #" << i << " params = " << total_params << std::endl;
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("max_pool", max_pool);

    for (int i = 0; i < num_stacks; i++) {
        std::string stack_name = "stack" + std::to_string(i+1);
        register_module(stack_name, stacks[i]);
    }

    register_module("avg_pool", avg_pool);
    register_module("fc", fc);
    register_module("tanh", tanh);
}

torch::Tensor ResNetImpl::forward(torch::Tensor x) {
    torch::Tensor out = max_pool(torch::nn::functional::relu(bn1(conv1(x))));

    for (int i = 0; i < num_stacks; i++) {
        out = stacks[i]->forward(out);
    }

    out = avg_pool(out);
    out = out.view({out.size(0), -1});

    out = tanh(fc(out));

    return out;
}
