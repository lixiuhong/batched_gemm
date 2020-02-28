#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "cudnn.h"


static inline const char* cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}


#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
static inline void Assert(cudaError_t  code, const char *file, int line){
	if(code!=cudaSuccess) {
		printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line, cudaGetErrorString(code));
		exit(EXIT_FAILURE);
	}
}
static inline void Assert(cudnnStatus_t code, const char *file, int line){
    if (code!=CUDNN_STATUS_SUCCESS){
		printf("cuDNN API Error: %s:%d:'%s'\n", file, line, cudnnGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}
static inline void Assert(cublasStatus_t code, const char *file, int line){
    if (code!=CUBLAS_STATUS_SUCCESS){
		printf("cuBLAS API Error: %s:%d:'%s'\n", file, line, cublasGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}


#define KernelErrChk(){\
		cudaError_t errSync  = cudaGetLastError();\
		cudaError_t errAsync = cudaDeviceSynchronize();\
		if (errSync != cudaSuccess) {\
			  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));\
			  exit(EXIT_FAILURE);\
		}\
		if (errAsync != cudaSuccess){\
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));\
			exit(EXIT_FAILURE);\
		}\
}
#endif
