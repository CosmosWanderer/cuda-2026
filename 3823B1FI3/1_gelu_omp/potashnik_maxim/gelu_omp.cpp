#include <vector>
#include <cmath>

#include "gelu_omp.h"


float GELU(const float& x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;
    float x3 = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = std::tanh(tanh_arg);
    return 0.5 * x * (1.0f + tanh_val);
} 

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int sz = static_cast<int>(input.size());
    std::vector<float> res(sz);
    
    for (int i = 0; i < sz; i++)
    {
        res[i] = GELU(input[i]);
    }
    return res;
}