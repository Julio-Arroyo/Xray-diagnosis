#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

// #include <torchvision/csrc/cpu/image/image.h>
// #include <opencv2/imgcodecs.hpp>

// #include "OpenCV_Torch.hpp"


int main() {
    // torch::Tensor t = vision::decodeJPEG("/groups/CS156b/2023/Xray-diagnosis/img/view1_frontal.jpg");
    // torch::Tensor x = torch::zeros(5);
    // std::cout << "Sizes: " << t.sizes() << ", " << x.sizes() << std::endl;
    // std::cout << "Options: " << t.options() << ", " << x.options() << std::endl;

    // cv::Mat mymat;
    // std::cout << "wtf: " << mymat.size() << std::endl;
    // cv::Mat m = cv::imread("/groups/CS156b/2023/Xray-diagnosis/img/view1_frontal.jpg",
    //                        cv::IMREAD_GRAYSCALE);
    // auto m_tensor = ToTensor(m);
    // std::cout << "bitches" << m.size() << std::endl;
    // std::cout << "SIUU" << m_tensor.sizes() << std::endl;

    torch::jit::script::Module dataset;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        dataset = torch::jit::load("/groups/CS156b/2023/Xray-diagnosis/Cpp/data/first60k.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }


    // torch::Tensor myX = dataset.named_parameters()["data"];
    // std::cout << "Options:" << myX.options() << std::endl;
    // std::cout << "Sizes: " << myX.sizes() << std::endl;

    for (const auto& buff : dataset.named_buffers()) {
        std::string name = buff.name;
        torch::Tensor t = buff.value;
        std::cout << name << "," << std::endl;
        std::cout << "Sizes: " << t.sizes() << std::endl;
        std::cout << "Options: " << t.options() << std::endl;
    }
}
