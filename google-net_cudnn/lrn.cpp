#include <cmath>
#include "cudnn.h"
#include "util.h"

void lrn(cudnnHandle_t handle, int N, int C, int H, int W, int R, int S, float lrnAlpha, float lrnBeta, float lrnK, float *input, float *output){

	cudnnTensorDescriptor_t xDesc;
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	cudnnTensorDescriptor_t yDesc;
	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	cudnnLRNDescriptor_t normDesc;
	ErrChk(cudnnCreateLRNDescriptor(&normDesc));
	ErrChk(cudnnSetLRNDescriptor(normDesc, R, lrnAlpha, lrnBeta, lrnK));


	float one = 1.f, zero = 0.f;
	ErrChk(cudnnLRNCrossChannelForward(handle, normDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, xDesc, input, &zero, yDesc, output));

	ErrChk(cudnnDestroyLRNDescriptor(normDesc));
	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));
}
