#include <cmath>
#include "cudnn.h"
#include "util.h"

void pooling(cudnnHandle_t handle, int N, int C, int H, int W, int R, int S, int U, int V, int pad_h, int pad_w, int P, int Q, float *input, float *output, cudaStream_t s){

	ErrChk(cudnnSetStream(handle, s));

	cudnnPoolingDescriptor_t poolingDesc;
	ErrChk(cudnnCreatePoolingDescriptor(&poolingDesc));
	ErrChk(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, R, S, pad_h, pad_w, U, V));
	
	cudnnTensorDescriptor_t xDesc;
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	cudnnTensorDescriptor_t yDesc;
	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, P, Q));
	
	float one = 1.0, zero = 0.0;
	ErrChk(cudnnPoolingForward(handle, poolingDesc, &one, xDesc, input, &zero, yDesc, output));	
	
	ErrChk(cudnnDestroyPoolingDescriptor(poolingDesc));
	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));
}
