cmake_minimum_required(VERSION 3.10)

project(MNNExample)

# Thêm đường dẫn tới thư viện MNN
set(MNN_LIB_PATH "/home/truong/workspace/FaceDetectionSDK/libs/libMNN.so")
include_directories("/home/truong/workspace/FaceDetectionSDK/FaceDetectionSDK/include")

add_executable(MNNExample /home/truong/workspace/FaceDetectionSDK/SampleApp/main.cpp)
target_link_libraries(MNNExample ${MNN_LIB_PATH})
