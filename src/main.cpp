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
#include "../include/ThreadedFuzzyArt.h"

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 7)
    {
        std::cout << "Usage: ./art_gpu <model.onnx> [frames] [categories] [cpu_threads] [gpu_threads] [gpu_blocks]" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    int num_frames = (argc >= 3) ? std::stoi(argv[2]) : 100;
    int num_categories = (argc >= 4) ? std::stoi(argv[3]) : 128;
    int cpu_threads = (argc >= 5) ? std::stoi(argv[4]) : 1;
    int gpu_threads = (argc >= 6) ? std::stoi(argv[5]) : 256;
    int gpu_blocks = (argc >= 7) ? std::stoi(argv[6]) : 128;

    // Loads the model and prepares for inference
    CudaInferencer inferencer(modelPath);

    std::vector<int64_t> tapShape = inferencer.get_tap_output_shape();
    int channels = tapShape[1];
    int height = tapShape[2];
    int width = tapShape[3];

    std::cout << "Input Dim: " << channels << std::endl;

    // For our testing we set the number of categories to a fixed value
    std::cout << "Categories: " << num_categories << std::endl;

    std::cout << "CPU Threads: " << cpu_threads << std::endl;
    std::cout << "GPU Threads Per Block: " << gpu_threads << " GPU Blocks: " << gpu_blocks << std::endl;

    FuzzyART gpuArt(channels, num_categories, 0.99f, 0.001f, 1.0f, num_categories, gpu_blocks, gpu_threads);
    CPUFuzzyART cpuArt(channels, num_categories, 0.99f, 0.001f, 1.0f, num_categories);
    ThreadedFuzzyART threadedArt(channels, num_categories, 0.99f, 0.001f, 1.0f, num_categories, cpu_threads);

    int inputH = 640;
    int inputW = 640;
    size_t inputByteSize = 1 * 3 * inputH * inputW * sizeof(float);

    float *d_input;
    cudaMalloc(&d_input, inputByteSize);

    float *d_art_input;
    cudaMalloc(&d_art_input, channels * sizeof(float));

    std::cout << "Benchmarking " << num_frames << " frames" << std::endl;

    double total_gpu_art_time = 0.0;
    double total_cpu_art_time = 0.0;
    double total_threaded_art_time = 0.0;

    int log_interval = std::max(1, num_frames / 10);

    for (int i = 0; i < num_frames; i++)
    {
        // We generate a random image for testing
        cv::Mat fakeImg(inputH, inputW, CV_8UC3);
        cv::randu(fakeImg, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::Mat blob = cv::dnn::blobFromImage(fakeImg, 1.0 / 255.0, cv::Size(inputW, inputH), cv::Scalar(), true, false);
        cudaMemcpy(d_input, blob.ptr<float>(), inputByteSize, cudaMemcpyHostToDevice);

        // ONNX Inference
        std::pair<float *, float *> results = inferencer.inference_on_gpu(d_input);
        cudaDeviceSynchronize();

        // Global Average Pooling on TAP output to get ART input
        // We mainly just do this on GPU since the data is already there
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

        // Data Transfer to Host for CPU and Threaded ART
        std::vector<float> h_art_input(channels);
        cudaMemcpy(h_art_input.data(), d_art_input, channels * sizeof(float), cudaMemcpyDeviceToHost);

        // CPU Fuzzy ART
        auto cpu_start_time = std::chrono::high_resolution_clock::now();
        cpuArt.run(h_art_input);
        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        double cpu_art_ms = std::chrono::duration<double, std::milli>(cpu_end_time - cpu_start_time).count();
        total_cpu_art_time += cpu_art_ms;

        // Threaded Fuzzy ART
        auto threaded_start_time = std::chrono::high_resolution_clock::now();
        threadedArt.run(h_art_input);
        auto threaded_end_time = std::chrono::high_resolution_clock::now();
        double threaded_art_ms = std::chrono::duration<double, std::milli>(threaded_end_time - threaded_start_time).count();
        total_threaded_art_time += threaded_art_ms;
    }

    std::cout << std::endl;
    std::cout << "Final Results for " << num_frames << " frames" << std::endl;
    std::cout << "Total GPU Categories: " << gpuArt.get_num_categories() << std::endl;
    std::cout << "Total CPU Categories: " << cpuArt.get_num_categories() << std::endl;
    std::cout << "Total Threaded Categories: " << threadedArt.get_num_categories() << std::endl;

    std::cout << std::endl;
    std::cout << "Average GPU ART Time: " << total_gpu_art_time / num_frames << " ms" << std::endl;
    std::cout << "Average CPU ART Time: " << total_cpu_art_time / num_frames << " ms" << std::endl;
    std::cout << "Average Threaded ART Time: " << total_threaded_art_time / num_frames << " ms" << std::endl;
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_art_input);

    return 0;
}