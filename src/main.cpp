#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime.h>
#include <string>

#include "../include/CudaInferencer.h"
#include "../include/FuzzyART.h"
#include "../include/CPUFuzzyART.h"

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cout << "Usage: ./onnx_gpu_app <path_to_model.onnx> [num_frames]" << std::endl;
        std::cout << "Example: ./onnx_gpu_app model.onnx 100000" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    int num_frames = (argc == 3) ? std::stoi(argv[2]) : 100;

    try
    {
        std::cout << "Loading Model..." << std::endl;
        CudaInferencer inferencer(modelPath);

        std::vector<int64_t> tapShape = inferencer.get_tap_output_shape();
        int channels = 0;
        int height = 1;
        int width = 1;

        if (tapShape.size() == 4)
        {
            channels = tapShape[1];
            height = tapShape[2];
            width = tapShape[3];
        }
        else
        {
            std::cerr << "Unexpected TAP shape. Assuming [1, C, H, W]" << std::endl;
            channels = tapShape[1];
        }

        std::cout << "Input Dim: " << channels << std::endl;

        int max_categories = std::max(1000, num_frames);
        std::cout << "Max Categories set to: " << max_categories << std::endl;

        FuzzyART gpuArt(channels, max_categories, 0.99f, 0.001f, 1.0f);
        CPUFuzzyART cpuArt(channels, max_categories, 0.99f, 0.001f, 1.0f);

        int inputH = 640;
        int inputW = 640;
        size_t inputByteSize = 1 * 3 * inputH * inputW * sizeof(float);

        float *d_input;
        cudaMalloc(&d_input, inputByteSize);

        float *d_art_input;
        cudaMalloc(&d_art_input, channels * sizeof(float));

        std::cout << "Benchmarking " << num_frames << " frames..." << std::endl;

        double total_onnx_time = 0.0;
        double total_gpu_art_time = 0.0;
        double total_cpu_art_time = 0.0;
        double total_transfer_time = 0.0;

        int log_interval = std::max(1, num_frames / 10);

        for (int i = 0; i < num_frames; ++i)
        {
            cv::Mat fakeImg(inputH, inputW, CV_8UC3);
            cv::randu(fakeImg, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            cv::Mat blob = cv::dnn::blobFromImage(fakeImg, 1.0 / 255.0, cv::Size(inputW, inputH), cv::Scalar(), true, false);
            cudaMemcpy(d_input, blob.ptr<float>(), inputByteSize, cudaMemcpyHostToDevice);

            auto onnx_inference_start_time = std::chrono::high_resolution_clock::now();

            std::pair<float *, float *> results = inferencer.inference_on_gpu(d_input);
            cudaDeviceSynchronize();

            auto onnx_inference_end_time = std::chrono::high_resolution_clock::now();
            double onnx_inference_duration_ms = std::chrono::duration<double, std::milli>(onnx_inference_end_time - onnx_inference_start_time).count();
            total_onnx_time += onnx_inference_duration_ms;

            float *d_tap_raw = results.second;

            cudaMemset(d_art_input, 0, channels * sizeof(float));
            launch_gap_kernel(d_tap_raw, d_art_input, channels, height, width);
            cudaDeviceSynchronize();

            // GPU Fuzzy ART
            auto gpu_start_time = std::chrono::high_resolution_clock::now();
            gpuArt.run(d_art_input);
            cudaDeviceSynchronize();
            auto gpu_end_time = std::chrono::high_resolution_clock::now();
            double gpu_art_ms = std::chrono::duration<double, std::milli>(gpu_end_time - gpu_start_time).count();
            total_gpu_art_time += gpu_art_ms;

            // ONNX -> CPU Transfer Time
            auto transfer_start_time = std::chrono::high_resolution_clock::now();
            std::vector<float> h_art_input(channels);
            cudaMemcpy(h_art_input.data(), d_art_input, channels * sizeof(float), cudaMemcpyDeviceToHost);
            auto transfer_end_time = std::chrono::high_resolution_clock::now();
            double transfer_ms = std::chrono::duration<double, std::milli>(transfer_end_time - transfer_start_time).count();
            total_transfer_time += transfer_ms;

            // CPU Fuzzy ART
            auto cpu_start_time = std::chrono::high_resolution_clock::now();
            cpuArt.run(h_art_input);
            auto cpu_end_time = std::chrono::high_resolution_clock::now();
            double cpu_art_ms = std::chrono::duration<double, std::milli>(cpu_end_time - cpu_start_time).count();
            total_cpu_art_time += cpu_art_ms;

            // Threaded Fuzzy ART
            // To Be Implemented

            if (i % log_interval == 0 || i == num_frames - 1)
            {
                std::cout << "Frame " << i << " | ONNX: " << onnx_inference_duration_ms << "ms" << " | GPU ART: " << gpu_art_ms << "ms" << " | CPU ART: " << cpu_art_ms << "ms" << " | Cats: " << gpuArt.get_num_categories() << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "Final Results for " << num_frames << " frames:" << std::endl;
        std::cout << "Total GPU Categories Created: " << gpuArt.get_num_categories() << std::endl;
        std::cout << "Total CPU Categories Created: " << cpuArt.get_num_categories() << std::endl;
        std::cout << std::endl;
        std::cout << "Average ONNX Inference Time: " << total_onnx_time / num_frames << " ms" << std::endl;
        std::cout << "Average GPU ART Time: " << total_gpu_art_time / num_frames << " ms" << std::endl;
        std::cout << "Average CPU ART Time: " << total_cpu_art_time / num_frames << " ms" << std::endl;
        std::cout << "Average Transfer Time: " << total_transfer_time / num_frames << " ms" << std::endl;
        std::cout << std::endl;

        cudaFree(d_input);
        cudaFree(d_art_input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Exception] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}