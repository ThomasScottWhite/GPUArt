#pragma once

#include <vector>
#include <string>

void launch_gap_kernel(float *d_in, float *d_out, int C, int H, int W);

class FuzzyART
{
public:
    FuzzyART(int input_dim, int max_categories = 1000, float vigilance = 0.9f, float choice_alpha = 1e-3f, float learning_rate = 1.0f, int init_categories = 0, int max_blocks = 128, int threads_per_block = 256);
    ~FuzzyART();

    int run(float *d_input);

    int get_num_categories() const { return num_initialized_categories_; }

private:
    int input_dim_;
    int art_dim_;
    int max_categories_;

    int max_blocks_ = 128;
    int threads_per_block_ = 256;

    float vigilance_;
    float choice_alpha_;
    float learning_rate_;

    int num_initialized_categories_ = 0;

    float *d_weights_ = nullptr;
    float *device_cat_activations = nullptr;
    float *d_input_compl_ = nullptr;

    void init_device_memory();
};