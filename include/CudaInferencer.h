#pragma once

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

class CudaInferencer
{
public:
    CudaInferencer(const std::string &model_path);
    ~CudaInferencer();

    std::pair<float *, float *> inference_on_gpu(float *d_input_image);

    std::vector<int64_t> get_model_output_shape() const { return output_shape_0_; }
    std::vector<int64_t> get_tap_output_shape() const { return output_shape_2_; }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_cuda_{nullptr};

    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_0_;
    std::vector<int64_t> output_shape_2_;

    float *d_output_0_ = nullptr;
    float *d_output_2_ = nullptr;
    size_t output_0_size_bytes_ = 0;
    size_t output_2_size_bytes_ = 0;
};