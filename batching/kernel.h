#include "kernel_128.h"
#include "kernel_256.h"

template<int kThreads>
__global__ void gemm(int M[], int N[], int K[], float *A[], float *B[], float *C[], int T_strategy[], int B_strategy[]){}


/*
template<>
__global__ void gemm<128>(int M[], int N[], int K[], float *A[], float *B[], float *C[], int T_strategy[]){
	
	extern __shared__ float sh[];

	int begin = Tile[blockIdx.x];
	int end = Tile[blockIdx.x+1];
	int t = T_strategy[blockIdx.z];

	//main loop for all tiles assigned to this block
#pragma unroll
	for (int b=begin; b<end; ++b){
		
		int ind = GEMM[b];
		int m = M[ind];
		int n = N[ind];
		int k = K[ind];
		
		float *a = A[ind];
		float *b = B[ind];
		float *c = C[ind];
	
		int by = Y_Coord[ind];
		int bx = X_Coord[ind];	

		switch(t){
			case 0:
				if (blockIdx.x * 16 < M[i] && blockIdx.y * 16 < N[i])	
					gemm_128_16x16(M[i], N[i], K[i], A[i], B[i], C[i], sh);
				break;
			case 1:
				if (blockIdx.x * 32 < M[i] && blockIdx.y * 32 < N[i])	
					gemm_128_32x32(M[i], N[i], K[i], A[i], B[i], C[i], sh);
				break;
			case 2:
				if (blockIdx.x * 64 < M[i] && blockIdx.y * 64 < N[i])	
					gemm_128_64x64(M[i], N[i], K[i], A[i], B[i], C[i], sh);
				break;
			case 3:
				if (blockIdx.x * 128 < M[i] && blockIdx.y * 64 < N[i])	
					gemm_128_128x64(M[i], N[i], K[i], A[i], B[i], C[i], sh);
				break;
			case 4:
				if (blockIdx.x * 64 < M[i] && blockIdx.y * 128 < N[i])	
					gemm_128_64x128(M[i], N[i], K[i], A[i], B[i], C[i], sh);
				break;
			case 5:
	//			if (blockIdx.x * 128 < M[i] && blockIdx.y * 128 < N[i])	
	//				gemm_128_128x128(M[i], N[i], K[i], A[i], B[i], C[i], sh);
				break;
		}
	}

	return;
}
*/

//template<>
__global__ void gemm_256(int M[], int N[], int K[], float *A[], float *B[], float *C[], int T_strategy[], int B_strategy[]){
	
	extern __shared__ float sh[];

	int i = blockIdx.z;
	int t = T_strategy[i];
	int b = B_strategy[i];
	int by;
	int bx;
	//main loop for all tiles assigned to this block

	for (int j=0; j<b; ++j){
		
		switch(t){
			case 0:
				by = blockIdx.x * 16 * b + j*16;		
				bx = blockIdx.y * 16;		
				if (blockIdx.x *b* 16 < M[i] && blockIdx.y * 16 < N[i])	
					gemm_256_16x16(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, sh);
				break;
			case 1:
				by = blockIdx.x *b* 32 * b + j*32;		
				bx = blockIdx.y * 32;		
				if (blockIdx.x * 32 < M[i] && blockIdx.y * 32 < N[i])	
					gemm_256_32x32(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, sh);
				break;
			case 2:
				by = blockIdx.x * 64 * b + j*64;		
				bx = blockIdx.y * 64;		
				if (blockIdx.x *b* 64 < M[i] && blockIdx.y * 64 < N[i])	
					gemm_256_64x64(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, sh);
				break;
			case 3:
				by = blockIdx.x * 128 * b + j*128;		
				bx = blockIdx.y * 64;		
				if (blockIdx.x *b* 128 < M[i] && blockIdx.y * 64 < N[i])	
					gemm_256_128x64(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, sh);
				break;
			case 4:
				by = blockIdx.x * 64 * b + j*64;		
				bx = blockIdx.y * 128;		
				if (blockIdx.x *b* 64 < M[i] && blockIdx.y * 128 < N[i])	
					gemm_256_64x128(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, sh);
				break;
			case 5:
				by = blockIdx.x * 128 * b + j*128;		
				bx = blockIdx.y * 128;		
				if (blockIdx.x *b* 128 < M[i] && blockIdx.y * 128 < N[i])	
					gemm_256_128x128(M[i], N[i], K[i], A[i], B[i], C[i], by, bx, sh);
				break;
		}
	}

	return;
}
