#include "cudnn.h"
#include "util.h"
#include <cmath>

void activation(cudnnHandle_t handle, int N, int C, int H, int W, float *input, float *output, cudaStream_t s){

	float one = 1.0, zero = 0.0;
	
	ErrChk(cudnnSetStream(handle, s));

	cudnnActivationDescriptor_t activationDesc;
	ErrChk(cudnnCreateActivationDescriptor(&activationDesc));
	ErrChk(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.f));

	cudnnTensorDescriptor_t xDesc;
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	cudnnTensorDescriptor_t yDesc;
	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
	
	ErrChk(cudnnActivationForward(handle, activationDesc, &one, xDesc, input, &zero, yDesc, output));
	
	ErrChk(cudnnDestroyActivationDescriptor(activationDesc));
	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));

}
