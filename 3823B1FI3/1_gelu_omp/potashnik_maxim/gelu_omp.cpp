#ifdef __GNUC__
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#endif

#include <vector>
#include <cmath>
#include <omp.h>

#include "gelu_omp.h"

/* Optimizations

1. Optimized calculations (sigmoid form) 
2. Inlined function into loop
3. Thread parallelism + SIMD for each thread
4. Flags
5. loop-unrolling (should help?)
*/

constexpr float coeff_1 = 1.59576912f;
constexpr float coeff_2 = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int sz = static_cast<int>(input.size());
    std::vector<float> res(sz);
    
#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < sz; i++)
    {
        float x = input[i];

        float x3 = x * x * x;
        float exp_arg = coeff_1 * (x + coeff_2 * x3);
        float sigmoid = 1.0f / (1.0f + expf(-exp_arg));
        
        res[i] = x * sigmoid;
    }
    return res;
}