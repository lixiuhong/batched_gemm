#include "cudnn.h"
#include "util.h"
#include <cmath>

void conv(cudnnHandle_t handle, int N, int C, int H, int W, int K, int R, int S,
        int U, int V, int pad_h, int pad_w, int P, int Q,
        float *input, float *filter,
        float *buf, float *output,
        int algo,
        cudaStream_t s){
	
	float one = 1.0, zero = 0.0;
	size_t size;

	ErrChk(cudnnSetStream(handle, s));

	cudnnTensorDescriptor_t xDesc;
	ErrChk(cudnnCreateTensorDescriptor(&xDesc));
	ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

	cudnnTensorDescriptor_t yDesc;
	ErrChk(cudnnCreateTensorDescriptor(&yDesc));
	ErrChk(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));

	cudnnFilterDescriptor_t filterDesc;
	ErrChk(cudnnCreateFilterDescriptor(&filterDesc));
	ErrChk(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

	cudnnConvolutionDescriptor_t convDesc;
	ErrChk(cudnnCreateConvolutionDescriptor(&convDesc));
	ErrChk(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	ErrChk(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, filterDesc, convDesc, yDesc, (cudnnConvolutionFwdAlgo_t)algo, (size_t *)&(size)));

	ErrChk(cudnnConvolutionForward(handle, &one, xDesc, input, filterDesc, filter, convDesc, (cudnnConvolutionFwdAlgo_t)algo, buf, size, &zero, yDesc, output));

	ErrChk(cudnnDestroyTensorDescriptor(xDesc));
	ErrChk(cudnnDestroyTensorDescriptor(yDesc));
	ErrChk(cudnnDestroyFilterDescriptor(filterDesc));
	ErrChk(cudnnDestroyConvolutionDescriptor(convDesc));
}
