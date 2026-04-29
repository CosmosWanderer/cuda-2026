#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

static const float C1 = 1.59576912f; 
static const float C2 = 0.044715f;

__global__ void Kernel(float4* in, float4* out, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {

        float4 x4 = in[i];
        float4 res;

        float argX = C1 * (x4.x + C2 * x4.x * x4.x * x4.x);
        res.x = x4.x / (1.0f + expf(-argX));

        float argY = C1 * (x4.y + C2 * x4.y * x4.y * x4.y);
        
        res.y = x4.y / (1.0f + expf(-argY));
        
        float argZ = C1 * (x4.z + C2 * x4.z * x4.z * x4.z);
        res.z = x4.z / (1.0f + expf(-argZ));

        float argW = C1 * (x4.w + C2 * x4.w * x4.w * x4.w);
        res.w = x4.w / (1.0f + expf(-argW));

        out[i] = res; 
    }
}


std::vector<float> GeluCUDA(const std::vector<float>& input) { 
    int bytes = static_cast<int>(input.size()) * static_cast<int>(sizeof(float));

    static float* vec_in;
    static float* vec_res;
    static int allocated_size = 0;
    static cudaStream_t stream;

    if (allocated_size < bytes) {
        if (d_in)
            cudaFree(d_in);
        if (d_out)
            cudaFree(d_out);

        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        if (!stream)
            cudaStreamCreate(&stream);

        allocated_size = bytes;
    }

    cudaMemcpyAsync(vec_in, input, bytes, cudaMemcpyHostToDevice, stream);

    int n = static_cast<int>(input.size() / 4);
    int tpb = 512;
    int grid = (n + tpb - 1) / tpb;

    Kernel<<<grid, tpb, 0, stream>>>((float4*)vec_in, (float4*)vec_res, n);


    std::vector<float> output(n); 
    cudaMemcpyAsync(output.data(), vec_res, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return output;
}
