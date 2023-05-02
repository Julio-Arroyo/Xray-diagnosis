#include <resnet.hpp>

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
ResNetImpl::ResNetImpl(int nblocks_per_stack, bool skip_connections) :
    conv1(torch::nn::Conv2dOptions(3, 16, 3)
                        .stride(1)
                        .padding(1)
                        .bias(false)),
    bn1(torch::nn::BatchNorm2dOptions(16).track_running_stats(true)),

    avg_pool(torch::nn::AvgPool2dOptions(8).stride(1)),
    fc(64, 1),
    tanh()
{
    // Build stacks by adding ResidualBlock's
    for (int i = 0; i < nblocks_per_stack; i++) {
        // first block of stacks 2 and 3 have downsampling
        int stride;
        if (i == 0) {
            stride = 2;
        } else {
            stride = 1;
        }

        
        stack1->push_back(ResidualBlock(16, stride, skip_connections));
        stack2->push_back(ResidualBlock(32, stride, skip_connections));
        stack3->push_back(ResidualBlock(64, stride, skip_connections));
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);

    register_module("stack1", stack1);
    register_module("stack2", stack2);
    register_module("stack3", stack3);

    register_module("avg_pool", avg_pool);
    register_module("fc", fc);
    register_module("tanh", tanh);
}

torch::Tensor ResNetImpl::forward(torch::Tensor x) {
    torch::Tensor out = torch::nn::functional::relu(bn1(conv1(x)));

    out = stack1->forward(out);
    out = stack2->forward(out);
    out = stack3->forward(out);

    out = avg_pool(out);
    std::cout << "DEBUG: in ResNet->forward, shape BEFORE view(): " << out.sizes() << std::endl;
    out = out.view({out.size(0), -1});
    std::cout << "DEBUG: in ResNet->forward, shape AFTER view(): " << out.sizes() << std::endl;

    out = tanh(fc(out));

    return out;
}
