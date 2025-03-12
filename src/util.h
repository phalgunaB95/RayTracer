#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda_runtime.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " (code "
                  << static_cast<unsigned int>(result) << ") at "
                  << file << ":" << line << " in " << func << "\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#endif// CUDA_UTIL_H
