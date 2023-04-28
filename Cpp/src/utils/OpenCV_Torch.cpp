#include <OpenCV_Torch.hpp>

auto ToTensor(cv::Mat &img) {
    torch::Tensor tensor_image = torch::from_blob(img.data,
                                                  {3, img.rows, img.cols},
                                                  torch::at::kByte);
}