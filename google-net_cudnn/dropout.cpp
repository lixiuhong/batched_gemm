#include "cudnn.h"
#include "util.h"
#include <cmath>

void dropout(cudnnHandle_t handle, float dropout, int N, int C, int H, int W,
        float *input, float *buf, float *output) {
	cudnnDropoutDescriptor_t dropoutDesc;
    ErrChk(cudnnCreateDropoutDescriptor(&dropoutDesc));
    size_t stateSize;
	ErrChk(cudnnDropoutGetStatesSize(handle, &stateSize));
    ErrChk(cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, buf, stateSize, 462565));

	cudnnTensorDescriptor_t xDesc;
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	cudnnTensorDescriptor_t yDesc;
	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    size_t reserveSize;
	ErrChk(cudnnDropoutGetReserveSpaceSize(xDesc, &reserveSize));

	ErrChk(cudnnDropoutForward(handle, dropoutDesc, xDesc, input, yDesc, output, buf + stateSize, reserveSize));

	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));
    ErrChk(cudnnDestroyDropoutDescriptor(dropoutDesc));
}
