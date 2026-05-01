#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

//__restrict__ - указание nvcc о том, что данные доступны только через этот указатель в данной области видимости. Тем самым мы уменьшаем количество проверок с его стороны
__global__ void naiveGEMM_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, const int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < n && row < n)
    {
        float sum = 0.0F;
        for (int k = 0; k < n; k++)
            sum += a[row * n + k] * b[k * n + col];
        c[row*n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) 
{
    float *a_gpu = nullptr;
    float *b_gpu = nullptr;
    float *c_gpu = nullptr;

    const int bytes = n * n * sizeof(float);

    cudaMalloc(&a_gpu, bytes);
    cudaMalloc(&b_gpu, bytes);
    cudaMalloc(&c_gpu, bytes);

    cudaMemcpy(a_gpu, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16); 
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

    naiveGEMM_kernel <<< grid_size, block_size >>> (a_gpu, b_gpu, c_gpu, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), c_gpu, bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return c;
}