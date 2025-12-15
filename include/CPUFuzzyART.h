#ifndef CPU_FUZZY_ART_H
#define CPU_FUZZY_ART_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>

class CPUFuzzyART
{
public:
    CPUFuzzyART(int input_dim, int max_categories = 1000, float vigilance = 0.9f, float choice_alpha = 1e-3f, float learning_rate = 1.0f, int init_categories = 0);
    ~CPUFuzzyART() = default;

    int run(const std::vector<float> &input);

    int get_num_categories() const { return num_initialized_categories_; }
    int get_art_dim() const { return art_dim_; }

private:
    int input_dim_;
    int art_dim_;
    int max_categories_;

    float vigilance_;
    float choice_alpha_;
    float learning_rate_;

    int num_initialized_categories_ = 0;

    std::vector<std::vector<float>> weights_;
};

#endif