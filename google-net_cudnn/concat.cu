#include "util.h"

__global__ void cudaConcatKernel(size_t numIns, size_t innerStride,
        size_t outerStride, size_t* concatDims, const float **ins, float *out) {
    size_t batchSize = 0;
    for (size_t i = 0; i < numIns; ++i) {
        batchSize += concatDims[i]*innerStride;
    }

    size_t iOuter = blockIdx.x;
    float* outPtr = out + iOuter*batchSize;
    for (size_t j = 0; j < numIns; ++j) {
        for (size_t k = 0; k < concatDims[j]; ++k) {
            for (size_t l = 0; l < (innerStride - 1)/blockDim.x + 1; ++l) {
                size_t x = l*blockDim.x + threadIdx.x;
                if (x < innerStride) {
                    outPtr[k*innerStride + x] = *(ins[j] +
                        iOuter*concatDims[j]*innerStride + k*innerStride + x);
                }
            }
        }
        outPtr += concatDims[j]*innerStride;
    }
}

void launchCudaConcatKernel(size_t numIns,
        size_t innerStride, size_t outerStride, size_t* concatDims,
        const float **ins, float *out) {
    size_t gridsize = outerStride;
    size_t blocksize = 256;
    switch ((innerStride + 63)/64) {
        case 1: blocksize = 64; break;
        case 2: blocksize = 128; break;
        case 3: blocksize = 192; break;
        default: blocksize = 256; break;
    }
    cudaConcatKernel<<<gridsize, blocksize, 0>>>(numIns,
        innerStride, outerStride, concatDims, ins, out);
    KernelErrChk();
}

size_t* concatDims = new size_t[4];
float** ins = new float*[4];
void concat(int N, int H, int W, int C1, int C2, int C3, int C4,
        float *input1, float *input2, float *input3, float *input4,
        float *buf, float *output) {
    concatDims[0] = static_cast<size_t>(C1);
    concatDims[1] = static_cast<size_t>(C2);
    concatDims[2] = static_cast<size_t>(C3);
    concatDims[3] = static_cast<size_t>(C4);
    ins[0] = input1;
    ins[1] = input2;
    ins[2] = input3;
    ins[3] = input4;
    size_t *devConcatDims = (size_t*)buf;
    const float **devIns = (const float **)(buf + 128);//bigger step
	ErrChk(cudaMemcpy(devIns, ins, 4*sizeof(float*), cudaMemcpyHostToDevice));
	ErrChk(cudaMemcpy(devConcatDims, concatDims, 4*sizeof(size_t), cudaMemcpyHostToDevice));
    
    launchCudaConcatKernel((size_t)4, size_t(H * W), size_t(N), devConcatDims, (const float **)devIns, output);
}
