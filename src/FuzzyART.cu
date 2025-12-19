#include "../include/FuzzyART.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <random>

// These are just some preprocessing kernels for FuzzyART,
// The global average pooling kernel condenses the intermediate layers
// Complement coding kernel prepares the input for FuzzyART processing

// These are not very effecient implementations but we are not measuring performance here
__global__ void global_average_pooling_kernel(float *d_in, float *d_out, int Num_Channels, int H, int W)
{
    int channel = blockIdx.x;
    if (channel >= Num_Channels)
        return;

    int thread_id = threadIdx.x;
    int stride = blockDim.x;
    int num_pixels = H * W;

    float sum = 0.0f;
    for (int i = thread_id; i < num_pixels; i += stride)
    {
        sum += d_in[channel * num_pixels + i];
    }

    float val = sum / (float)num_pixels;
    atomicAdd(&d_out[channel], val);
}

void launch_gap_kernel(float *d_in, float *d_out, int Num_Channels, int H, int W)
{
    global_average_pooling_kernel<<<Num_Channels, 256>>>(d_in, d_out, Num_Channels, H, W);
}

__global__ void complement_coding_kernel(float *d_in, float *d_out, int original_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < original_dim; i += stride)
    {
        float val = d_in[i];
        if (val < 0.0f)
            val = 0.0f;
        if (val > 1.0f)
            val = 1.0f;

        d_out[i] = val;
        d_out[i + original_dim] = 1.0f - val;
    }
}

// Each thread computes the choice function for one category
__global__ void calculate_categorical_choice(float *d_input, float *d_weights, float *device_cat_activations, int dim, int num_cats, float alpha, float vigilance, float norm_input)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int current_cat = idx; current_cat < num_cats; current_cat += stride)
    {
        float norm_w = 0.0f;
        float norm_intersection = 0.0f;

        int offset = current_cat * dim;

        for (int i = 0; i < dim; i++)
        {
            float w = d_weights[offset + i];
            float in = d_input[i];

            norm_w += w;
            norm_intersection += fminf(in, w);
        }

        float match_score = norm_intersection / norm_input;

        if (match_score >= vigilance)
        {
            device_cat_activations[current_cat] = norm_intersection / (alpha + norm_w);
        }
        else
        {
            device_cat_activations[current_cat] = -1.0f;
        }
    }
}

// Update weights of the winning category
__global__ void update_weights_kernel(float *d_input, float *d_weights, int dim, int winner_idx, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < dim; i += stride)
    {
        int w_idx = winner_idx * dim + i;
        float w_old = d_weights[w_idx];
        float in = d_input[i];

        float intersection = fminf(in, w_old);
        d_weights[w_idx] = beta * intersection + (1.0f - beta) * w_old;
    }
}

// Initialize weights for a new category
__global__ void init_new_category_kernel(float *d_input, float *d_weights, int dim, int new_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < dim; i += stride)
    {
        d_weights[new_idx * dim + i] = d_input[i];
    }
}

FuzzyART::FuzzyART(int input_dim, int max_categories, float vigilance, float choice_alpha, float learning_rate, int init_categories, int max_blocks, int threads_per_block)
    : input_dim_(input_dim), max_categories_(max_categories),
      vigilance_(vigilance), choice_alpha_(choice_alpha), learning_rate_(learning_rate),
      max_blocks_(max_blocks), threads_per_block_(threads_per_block)
{
    art_dim_ = input_dim * 2;
    init_device_memory();

    if (init_categories > 0)
    {
        int limit = std::min(init_categories, max_categories);
        std::vector<float> host_weights(limit * art_dim_);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Initialize random weights
        for (int i = 0; i < limit; i++)
        {
            int offset = i * art_dim_;

            for (int k = 0; k < input_dim_; k++)
            {
                float val = dis(gen);

                host_weights[offset + k] = val;
                host_weights[offset + k + input_dim_] = 1.0f - val;
            }
        }

        size_t bytes_to_copy = host_weights.size() * sizeof(float);
        num_initialized_categories_ = limit;
    }
    else
    {
        num_initialized_categories_ = 0;
    }
}

FuzzyART::~FuzzyART()
{
    if (d_weights_)
        cudaFree(d_weights_);
    if (device_cat_activations)
        cudaFree(device_cat_activations);
    if (d_input_compl_)
        cudaFree(d_input_compl_);
}

void FuzzyART::init_device_memory()
{
    cudaMalloc(&d_weights_, max_categories_ * art_dim_ * sizeof(float));
    cudaMalloc(&device_cat_activations, max_categories_ * sizeof(float));
    cudaMalloc(&d_input_compl_, art_dim_ * sizeof(float));

    cudaMemset(d_weights_, 0, max_categories_ * art_dim_ * sizeof(float));
}

int FuzzyART::run(float *d_input)
{
    // Complement Coding
    int needed_blocks = (input_dim_ + threads_per_block_ - 1) / threads_per_block_;

    int grid_size = std::min(needed_blocks, max_blocks_);

    complement_coding_kernel<<<grid_size, threads_per_block_>>>(d_input, d_input_compl_, input_dim_);

    if (num_initialized_categories_ > 0)
    {
        // Choice Function
        int category_block_count = (num_initialized_categories_ + threads_per_block_ - 1) / threads_per_block_;
        int category_blocks = std::min(category_block_count, max_blocks_);

        calculate_categorical_choice<<<category_blocks, threads_per_block_>>>(
            d_input_compl_, d_weights_, device_cat_activations, art_dim_, num_initialized_categories_,
            choice_alpha_, vigilance_, (float)input_dim_);

        // Find Winner, we do this on CPU for simplicity
        std::vector<float> cat_activations(num_initialized_categories_);
        cudaMemcpy(cat_activations.data(), device_cat_activations, num_initialized_categories_ * sizeof(float), cudaMemcpyDeviceToHost);

        auto max_iter = std::max_element(cat_activations.begin(), cat_activations.end());
        float max_val = *max_iter;
        int winner_idx = std::distance(cat_activations.begin(), max_iter);

        // Update Weights
        if (max_val > -0.5f)
        {
            int weight_block_count = (art_dim_ + threads_per_block_ - 1) / threads_per_block_;
            int weight_blocks = std::min(weight_block_count, max_blocks_);

            update_weights_kernel<<<weight_blocks, threads_per_block_>>>(d_input_compl_, d_weights_, art_dim_, winner_idx, learning_rate_);
            return winner_idx;
        }
    }

    // Create New Category
    if (num_initialized_categories_ < max_categories_)
    {
        int new_idx = num_initialized_categories_;

        int weight_block_count = (art_dim_ + threads_per_block_ - 1) / threads_per_block_;
        int weight_blocks = std::min(weight_block_count, max_blocks_);
        init_new_category_kernel<<<weight_blocks, threads_per_block_>>>(d_input_compl_, d_weights_, art_dim_, new_idx);

        num_initialized_categories_++;
        return new_idx;
    }

    return -1;
}