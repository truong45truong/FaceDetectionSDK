#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <iostream>

using namespace MNN;
using namespace MNN::Express;

int main() {
    const char* model_path = "model.mnn";

    auto net = std::shared_ptr<Interpreter>(Interpreter::createFromFile(model_path));
    if (!net) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;  // Có thể thay bằng MNN_FORWARD_OPENCL để dùng GPU
    config.numThread = 4;           // Số luồng CPU
    auto session = net->createSession(config);
    if (!session) {
        std::cerr << "Failed to create session!" << std::endl;
        return -1;
    }

    auto inputTensor = net->getSessionInput(session, nullptr);

    std::vector<float> inputData(224 * 224, 1.0f);  // Giá trị mẫu
    auto inputDims = inputTensor->shape();

    // Tạo tensor với shape từ inputDims, kiểu dữ liệu float
    MNN::Tensor* tensor = new MNN::Tensor();  // Khởi tạo tensor đúng cách

    // Sao chép dữ liệu vào Tensor
    tensor->copyFromHostTensor(Tensor::create(inputDims, halide_type_of<float>(), inputData.data()));  // Sử dụng halide_type_of<float>()

    // Gán dữ liệu vào input tensor
    inputTensor->copyFromHostTensor(tensor);

    // Chạy mô hình
    net->runSession(session);

    // Lấy kết quả đầu ra
    auto outputTensor = net->getSessionOutput(session, nullptr);
    auto outputData = outputTensor->host<float>();
    auto outputSize = outputTensor->elementSize();

    std::cout << "Model output:" << std::endl;
    for (int i = 0; i < outputSize; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    delete tensor;  // Giải phóng bộ nhớ

    return 0;
}
