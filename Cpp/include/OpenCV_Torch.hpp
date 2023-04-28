#ifndef OPENCV_TORCH_HPP
#define OPENCV_TORCH_HPP

#include <torch/torch.h>
#include <opencv2/opencv.h>

auto ToTensor(cv::Mat &img);

#endif