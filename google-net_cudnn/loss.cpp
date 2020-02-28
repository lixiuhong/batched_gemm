#include "cudnn.h"
#include "util.h"
#include <cmath>

void loss(cublasHandle_t cublas_handle, int N, int C, int K,
        float *input, float *filter, float *output) {
    float alpha = 1.f, beta = 0.f;

    ErrChk(cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, C,
                (void*) &alpha, (void*) input, CUDA_R_32F, C,
                (void*) filter, CUDA_R_32F, C,
                (void*) &beta, (void*) output, CUDA_R_32F, N, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT));
}
