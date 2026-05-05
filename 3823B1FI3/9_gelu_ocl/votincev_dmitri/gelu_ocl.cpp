#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <string>

// OpenCL Kernel
// используем native_exp для максимальной скорости на GPU
// формула tanh(x) = 1 - 2 / (exp(2*x) + 1)
const char* kernelSource = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float x3 = x * x * x;
        float v = 0.79788456f * (x + 0.044715f * x3); 
        
        // tanh через native_exp
        float e = native_exp(2.0f * v);
        float t = (e - 1.0f) / (e + 1.0f);
        
        output[i] = 0.5f * x * (1.0f + t);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIdx) {
    size_t n = input.size();
    std::vector<float> output(n);

    // статические переменные для однократной инициализации (Performance Hint 1)
    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_program program = nullptr;
    static cl_kernel kernel = nullptr;
    static bool initialized = false;

    if (!initialized) {
        cl_uint numPlatforms;
        clGetPlatformIDs(0, nullptr, &numPlatforms);
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

        cl_platform_id platform = platforms[platformIdx];

        cl_device_id device;
        // CL_DEVICE_TYPE_GPU и 0-е устройство
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        
        // очередь (для асинхронности)
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_kernel", nullptr);

        initialized = true;
    }

    // создание буферов
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                   sizeof(float) * n, (void*)input.data(), nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                    sizeof(float) * n, nullptr, nullptr);

    // установка аргументов
    int count = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &count);

    // выполнение
    size_t globalSize = n;
    // округление globalSize до кратного 64 или 256 (улучшение)
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // чтение результата (блокирующее чтение для возврата вектора)
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * n, output.data(), 0, nullptr, nullptr);

    // очистка локальных ресурсов (буферы)
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);

    return output;
}