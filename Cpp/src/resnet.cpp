#include <resnet.hpp>

ResidualBlockImpl::ResidualBlockImpl(int out_channels, int stride=1, bool skip_connections=false) :
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
    x_halved_width_height = downsample(x);
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