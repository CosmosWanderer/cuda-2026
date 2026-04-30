#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
static size_t allocated_bytes = 0;
static cublasHandle_t handle = nullptr;
static bool initialized = false;

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = n * n * sizeof(float);
    
    if (!initialized) {
        cublasCreate(&handle);
        initialized = true;
    }
    
    if (allocated_bytes < bytes) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        allocated_bytes = bytes;
    }
    
    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_C, n);
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    return c;
}