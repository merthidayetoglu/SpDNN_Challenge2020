#pragma once

#include <cstdio>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t result, const char *file, const int line, bool fatal=false) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n",  file, line, int(result),
            cudaGetErrorString(result));\
    if (fatal) {
        exit(EXIT_FAILURE);
    }
  }
}

#define OR_PRINT(stmt) checkCuda(stmt, __FILE__, __LINE__);
#define OR_FATAL(stmt) checkCuda(stmt, __FILE__, __LINE__, true);