#include "../include/ThreadedFuzzyArt.h"
#include <numeric>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
#include <omp.h>

ThreadedFuzzyART::ThreadedFuzzyART(int input_dim, int max_categories, float vigilance, float choice_alpha, float learning_rate, int init_categories)
    : input_dim_(input_dim), max_categories_(max_categories),
      vigilance_(vigilance), choice_alpha_(choice_alpha), learning_rate_(learning_rate)
{
    art_dim_ = input_dim * 2;
    weights_.resize(max_categories * art_dim_);

    if (init_categories > 0)
    {
        int limit = std::min(init_categories, max_categories);
        num_initialized_categories_ = limit;

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < limit; i++)
        {
            int offset = i * art_dim_;
            for (int k = 0; k < input_dim_; k++)
            {
                float val = dis(gen);
                weights_[offset + k] = val;
                weights_[offset + k + input_dim_] = 1.0f - val;
            }
        }
    }
    else
    {
        num_initialized_categories_ = 0;
    }
}

int ThreadedFuzzyART::run(const std::vector<float> &input)
{
    if ((int)input.size() != input_dim_)
    {
        std::cerr << "[Error] ThreadedFuzzyART input dimension mismatch." << std::endl;
        return -1;
    }

    std::vector<float> I(art_dim_);
    for (int i = 0; i < input_dim_; i++)
    {
        float val = std::clamp(input[i], 0.0f, 1.0f);
        I[i] = val;
        I[i + input_dim_] = 1.0f - val;
    }

    std::vector<std::pair<float, int>> candidates(num_initialized_categories_);

#pragma omp parallel for schedule(static)
    for (int j = 0; j < num_initialized_categories_; ++j)
    {
        float norm_intersection = 0.0f;
        float norm_w = 0.0f;

        int offset = j * art_dim_;

        for (int k = 0; k < art_dim_; k++)
        {
            float w = weights_[offset + k]; // Fast linear access
            norm_intersection += std::min(I[k], w);
            norm_w += w;
        }

        float T = norm_intersection / (choice_alpha_ + norm_w);
        candidates[j] = {T, j};
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    float norm_I = (float)input_dim_;

    for (const auto &cand : candidates)
    {
        int idx = cand.second;
        int offset = idx * art_dim_;

        float norm_intersection = 0.0f;
        for (int k = 0; k < art_dim_; k++)
        {
            norm_intersection += std::min(I[k], weights_[offset + k]);
        }

        float match_score = norm_intersection / norm_I;

        if (match_score >= vigilance_)
        {
            for (int k = 0; k < art_dim_; k++)
            {
                float w_old = weights_[offset + k];
                float intersection = std::min(I[k], w_old);
                weights_[offset + k] = learning_rate_ * intersection + (1.0f - learning_rate_) * w_old;
            }
            return idx;
        }
    }

    if (num_initialized_categories_ < max_categories_)
    {
        int new_idx = num_initialized_categories_;
        int offset = new_idx * art_dim_;

        for (int k = 0; k < art_dim_; k++)
        {
            weights_[offset + k] = I[k];
        }

        num_initialized_categories_++;
        return new_idx;
    }

    return -1;
}