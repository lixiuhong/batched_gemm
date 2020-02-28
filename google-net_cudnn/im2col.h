/*
 * im2col.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef IM2COL_H_
#define IM2COL_H_

template<int BLOCK_SIZE>
__global__ void im2col_1101(float *out, float *in, int N, int C, int H, int W){
	//C*N blocks, and each block is responsible for a H*W data block of transformed matrix
	int n = blockIdx.x/C;
	int c = blockIdx.x%C;

	float *in_start = in + n*C*H*W + c*H*W;
	float *out_start = out + c*N*H*W + n*H*W;

	for (int i=0; i<H*W/BLOCK_SIZE; ++i){
		float *src = in_start + i*BLOCK_SIZE + threadIdx.x;
		float *des = out_start + i*BLOCK_SIZE + threadIdx.x;
		*des = *src;
	}
	if ((H*W%BLOCK_SIZE)!=0 && threadIdx.x<(H*W%BLOCK_SIZE)){
		float *src = in_start + (H*W/BLOCK_SIZE)*BLOCK_SIZE + threadIdx.x;
		float *des = out_start + (H*W/BLOCK_SIZE)*BLOCK_SIZE + threadIdx.x;
		*des = *src;
	}
}

template<int BLOCK_SIZE>
__global__ void im2col_3311_version1(float *out, float *in, int N, int C, int H, int W, int R, int S, int P, int Q){
	//C*N*(Q+S-1) blocks, and each block is assigned for a series of P*R data blocks along the diagonal
	int c = blockIdx.z;
	int n = blockIdx.y;

	int q = (blockIdx.x>=Q)? (Q-1):blockIdx.x;
	int s = (blockIdx.x>=Q)? (blockIdx.x-Q+1):0;

	int task = (q>1 && s==0)? 3:2;

	extern __shared__ float line_buffer[];

	float *result = out + c*N*Q*S*P*R + n*P*Q + s*(N*P*Q*R) + q*P;
	if ( ((q==0) && (s==0)) || ( (q==(Q-1)) && (s==(S-1)) ) ) {

		for(int j=0; j<(P*R)/BLOCK_SIZE; ++j){

			int y = (j*BLOCK_SIZE+threadIdx.x)/P;
			int x = (j*BLOCK_SIZE+threadIdx.x)%P;

			int ind = y*P*Q*N + x;

			result[ind] = 0.f;
		}

		if (((P*R)%BLOCK_SIZE)!=0 && threadIdx.x<((P*R)%BLOCK_SIZE)){
			int y = (((P*R)/BLOCK_SIZE)*BLOCK_SIZE + threadIdx.x)/P;
			int x = (((P*R)/BLOCK_SIZE)*BLOCK_SIZE + threadIdx.x)%P;

			int ind = y*P*Q*N + x;

			result[ind] = 0.f;
		}
	}
	else {
		float *data = in + n*C*H*W + c*H*W + (q+s-1)*W;
		line_buffer[0] = 0.f;

		for(int j=0; j<W/BLOCK_SIZE; ++j)
			line_buffer[1+threadIdx.x+j*BLOCK_SIZE] = data[threadIdx.x+j*BLOCK_SIZE];

		if ((W%BLOCK_SIZE)!=0 && threadIdx.x<(W%BLOCK_SIZE))
			line_buffer[1+threadIdx.x+(W/BLOCK_SIZE)*BLOCK_SIZE] = data[threadIdx.x+(W/BLOCK_SIZE)*BLOCK_SIZE];

		line_buffer[W+1] = 0.f;
		__syncthreads();

		for (int i=0; i<task; ++i){
			for(int j=0; j<(P*R)/BLOCK_SIZE; ++j){

				int y = (j*BLOCK_SIZE+threadIdx.x)/P;
				int x = (j*BLOCK_SIZE+threadIdx.x)%P;

				int ind = y*P*Q*N + x;

				result[ind] = line_buffer[y+x];
			}

			if (((P*R)%BLOCK_SIZE)!=0 && threadIdx.x<((P*R)%BLOCK_SIZE)){
				int y = (((P*R)/BLOCK_SIZE)*BLOCK_SIZE + threadIdx.x)/P;
				int x = (((P*R)/BLOCK_SIZE)*BLOCK_SIZE + threadIdx.x)%P;

				int ind = y*P*Q*N + x;

				result[ind] = line_buffer[y+x];
			}
			result += (N*P*Q*R - Q);
		}
	}
}



static void im2col(cudnnHandle_t handle, int N, int C, int H, int W, int K, int R, int S, int U, int V, int pad_h, int pad_w, float *input, float *output){

	int P = H;
	int Q = W;

	if (N==1 && R==1 && U==1 && pad_h==0){
		//There is no need to conduct im2col
	}
	else if (R==1 && U==1 && pad_h==0){
		im2col_1101<128><<<N*C, 128>>>(output, input, N, C, H, W);
		KernelErrChk();
	}
	else if (R==3 && U==1 && pad_h==1){
		dim3 grid;
		grid.x = Q+S-1;
		grid.y = N;
		grid.z = C;
		im2col_3311_version1<32><<<grid, 64, (W+2)*sizeof(float)>>>(output, input, N, C, H, W, R, S, P, Q);
		KernelErrChk();
	}
	else{
		cudnnTensorDescriptor_t xDesc;
		ErrChk(cudnnCreateTensorDescriptor(&xDesc));
		ErrChk(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

		cudnnFilterDescriptor_t filterDesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
		ErrChk(cudnnCreateFilterDescriptor(&filterDesc));
		ErrChk(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

		cudnnConvolutionDescriptor_t convDesc;
		ErrChk(cudnnCreateConvolutionDescriptor(&convDesc));
		ErrChk(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
		ErrChk(cudnnIm2Col(handle, xDesc, input, filterDesc, convDesc, output));

		ErrChk(cudnnDestroyTensorDescriptor(xDesc));
		ErrChk(cudnnDestroyFilterDescriptor(filterDesc));
		ErrChk(cudnnDestroyConvolutionDescriptor(convDesc));
	}
}


#endif /* IM2COL_H_ */
