#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cublas_v2.h>
#include "../include/util.h"

#define N_RUNS 10

int  main (int argc, char** argv) {

	ErrChk(cudaSetDevice(0));

	if(argc<2){
		printf("Usage: input the batch size\n");
		exit(EXIT_FAILURE);
	}

	int BATCH = atoi(argv[1]);
	
	int *M;
	int *N;
	int *K;

	M = (int*) malloc(BATCH * sizeof(int));
	N = (int*) malloc(BATCH * sizeof(int));
	K = (int*) malloc(BATCH * sizeof(int));

	std::fstream fs;
	fs.open("../data/input");
	if (!fs.is_open()){
		printf("Error opening input\n");
		exit(EXIT_FAILURE);
	}
	
	//read matrix config	
	for (int i=0; i<BATCH; ++i){
		fs>>M[i]>>N[i]>>K[i];
	}

    float **A;
	float **B;
	float **C;
	float alpha = 1.f;
	float beta = 0.f;

	A = (float**) malloc(BATCH * sizeof(float*));
	B = (float**) malloc(BATCH * sizeof(float*));
	C = (float**) malloc(BATCH * sizeof(float*));

	for (int i=0; i<BATCH; ++i){
		ErrChk(cudaMalloc((void**)&A[i], M[i]*K[i]*sizeof(float)));
		ErrChk(cudaMalloc((void**)&B[i], K[i]*N[i]*sizeof(float)));
		ErrChk(cudaMalloc((void**)&C[i], M[i]*N[i]*sizeof(float)));
	}

	float elapsedTime = 0.f;
    double time=0.f;
	float gflops_per_sec = 0.f;
	double gflops = 0.f;
	for (int i=0; i<BATCH; ++i)
		gflops += ((2 * int64_t(M[i]) * int64_t(N[i]) * int64_t(K[i])) + (2 * int64_t(M[i]) * int64_t(N[i])) ) / 1.0e9;

	cudaEvent_t start, stop;
	ErrChk(cudaEventCreate(&start));
	ErrChk(cudaEventRecord(start,0));

    cublasHandle_t handle;
    ErrChk(cublasCreate(&handle));

	//warm-up
	for (int i=0; i<BATCH; ++i){
		ErrChk(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M[i], N[i], K[i], (const void*) &alpha, (void*) A[i], CUDA_R_32F, M[i], (void*) B[i], CUDA_R_32F, K[i], (const void*) &beta, (void*) C[i], CUDA_R_32F, M[i], CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
	}
	ErrChk(cudaDeviceSynchronize());


	ErrChk(cudaEventCreate(&start));
	ErrChk(cudaEventRecord(start,0));

	for (int run=0; run<N_RUNS; ++run){
		for (int i=0; i<BATCH; ++i){
			ErrChk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M[i], N[i], K[i], &alpha, A[i], M[i], B[i], K[i], &beta, C[i], M[i]));
		}
	}
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);

	time = elapsedTime/N_RUNS;
	time /= 1.0e3; //convert time unit from millisecond to second
	gflops_per_sec   = gflops / time;
	printf("%f\n", gflops_per_sec);

	for (int i=0; i<BATCH; ++i){
		ErrChk(cudaFree(A[i]));		
		ErrChk(cudaFree(B[i]));		
		ErrChk(cudaFree(C[i]));		
	}

	free(M);
	free(N);
	free(K);
	free(A);
	free(B);
	free(C);

	return 0;
}
