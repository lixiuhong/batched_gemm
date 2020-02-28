#include "cudnn.h"
#include "util.h"
#include <cmath>

void softmax(cudnnHandle_t handle, int N, int C, float *input, float *output){

	float one = 1.0, zero = 0.0;
	size_t size;

	cudnnTensorDescriptor_t xDesc;
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));

	cudnnTensorDescriptor_t yDesc;
	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));

	cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
	cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

	ErrChk(cudnnSoftmaxForward(handle, algo, mode, &one, xDesc, input, &zero, yDesc, output));

	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));
}
