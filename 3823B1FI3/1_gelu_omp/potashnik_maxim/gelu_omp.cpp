#include <vector>
#include <cmath>

#include "gelu_omp.h"

/* Optimizations

1. Optimized calculations
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = 1 - 2 / (e^(2*x) + 1) 
2. Inlined function into loop

*/

constexpr float sqrt_2_over_pi = 0.7978845608028654f;
constexpr float coeff = 0.044715f;


std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int sz = static_cast<int>(input.size());
    std::vector<float> res(sz);
    
    for (int i = 0; i < sz; i++)
    {
        float x = input[i];

        float x3 = x * x * x;
        float exp_arg = 2 * sqrt_2_over_pi * (x + coeff * x3);
        float exp_val = expf(exp_arg);
        float tanh_val = 1 - 2 / (exp_val + 1);

        res[i] = 0.5 * x * (1.0f + tanh_val);
    }
    return res;
}