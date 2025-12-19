#include "../include/CudaInferencer.h"
#include <numeric>
#include <algorithm>
#include <iostream>
#include <onnxruntime_c_api.h>

// Modified from ONNX Runtime CUDA example
CudaInferencer::CudaInferencer(const std::string &model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "CudaInferencer"),
      memory_info_cuda_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    (void)OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_ = Ort::Session(env_, model_path.c_str(), session_options);
    memory_info_cuda_ = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    input_names_.push_back(strdup(input_name_ptr.get()));
    input_shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape_[0] == -1)
        input_shape_[0] = 1;

    // Model Outputs
    auto out0_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    output_names_.push_back(strdup(out0_name_ptr.get()));
    output_shape_0_ = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape_0_[0] == -1)
        output_shape_0_[0] = 1;

    // Tap Outputs
    auto out2_name_ptr = session_.GetOutputNameAllocated(2, allocator);
    output_names_.push_back(strdup(out2_name_ptr.get()));
    output_shape_2_ = session_.GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape_2_[0] == -1)
        output_shape_2_[0] = 1;

    // Keep a pointer to the output buffers on GPU for ART
    auto calc_size = [](std::vector<int64_t> &shape) -> int64_t
    {
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] < 0)
            {
                shape[i] = 300;
            }
        }
        return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    };

    int64_t out0_elements = calc_size(output_shape_0_);
    int64_t out2_elements = calc_size(output_shape_2_);

    output_0_size_bytes_ = out0_elements * sizeof(float);
    output_2_size_bytes_ = out2_elements * sizeof(float);

    cudaMalloc((void **)&d_output_0_, output_0_size_bytes_);
    cudaMalloc((void **)&d_output_2_, output_2_size_bytes_);
}

CudaInferencer::~CudaInferencer()
{
    if (d_output_0_)
        cudaFree(d_output_0_);
    if (d_output_2_)
        cudaFree(d_output_2_);
}

std::pair<float *, float *> CudaInferencer::inference_on_gpu(float *d_input_image)
{
    Ort::IoBinding binding(session_);

    int64_t input_count = std::accumulate(input_shape_.begin(), input_shape_.end(), 1LL, std::multiplies<int64_t>());

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda_,
        d_input_image,
        input_count,
        input_shape_.data(),
        input_shape_.size());

    binding.BindInput(input_names_[0], input_tensor);

    int64_t out0_count = output_0_size_bytes_ / sizeof(float);
    Ort::Value output_tensor_0 = Ort::Value::CreateTensor<float>(
        memory_info_cuda_,
        d_output_0_,
        out0_count,
        output_shape_0_.data(),
        output_shape_0_.size());
    binding.BindOutput(output_names_[0], output_tensor_0);

    int64_t out2_count = output_2_size_bytes_ / sizeof(float);
    Ort::Value output_tensor_2 = Ort::Value::CreateTensor<float>(
        memory_info_cuda_,
        d_output_2_,
        out2_count,
        output_shape_2_.data(),
        output_shape_2_.size());
    binding.BindOutput(output_names_[1], output_tensor_2);
    session_.Run(Ort::RunOptions{nullptr}, binding);

    return {d_output_0_, d_output_2_};
}