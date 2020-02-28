__device__ void gemm_256_128x128(int M, int N, int K, float *A, float *B, float *C, float *sh){

	if (threadIdx.x>=256)
		return;
	
    float *sh_A = sh;
	float *sh_B = sh + 2*128*8;

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];
	
	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
	int block_base_y = blockIdx.x*128;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	C_start += (16*M - 16);
	reg_C[8] = *(C_start);
	reg_C[9] = *(C_start + M/4);
	reg_C[10] = *(C_start + M/2);
	reg_C[11] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[12] = *(C_start);
	reg_C[13] = *(C_start + M/4);
	reg_C[14] = *(C_start + M/2);
	reg_C[15] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%32)*4 + (threadIdx.x/32)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/128)*4 + (threadIdx.x%128)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
		
	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%16)*4;
		int B_offset = double_buffer + ((threadIdx.x/16)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+64];
			reg_A[5] = sh_A[A_offset+65];
			reg_A[6] = sh_A[A_offset+66];
			reg_A[7] = sh_A[A_offset+67];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+256];
			reg_B[5] = sh_B[B_offset+260];
			reg_B[6] = sh_B[B_offset+264];
			reg_B[7] = sh_B[B_offset+268];

			reg_C[0].x = fma(reg_A[0], reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0], reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0], reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0], reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0], reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0], reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0], reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0], reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[4], reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[4], reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[4], reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[4], reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[4], reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[4], reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[4], reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[4], reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[1], reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1], reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[1], reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1], reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[1], reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[1], reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[1], reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[1], reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[5], reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[5], reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[5], reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[5], reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[5], reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[5], reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[5], reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[5], reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[2], reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[2], reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[2], reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[2], reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[2], reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[2], reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[2], reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[2], reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[6], reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[6], reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[6], reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[6], reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[6], reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[6], reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[6], reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[6], reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[3], reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[3], reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[3], reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[3], reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[3], reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[3], reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[3], reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[3], reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[7], reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[7], reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[7], reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[7], reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[7], reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[7], reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[7], reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[7], reg_B[7], reg_C[15].w);

			A_offset += 128;
			if (i==3) B_offset += 508;
			B_offset += 1;
		}

		double_buffer ^= 1024;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
		}
				
	}
	C_start -= (16*M + 16);
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start += (16*M - 16);
	*(C_start) = reg_C[8];
	*(C_start + M/4) = reg_C[9];
	*(C_start + M/2) = reg_C[10];
	*(C_start + 3*M/4) = reg_C[11];

	C_start += 16;
	*(C_start) = reg_C[12];
	*(C_start + M/4) = reg_C[13];
	*(C_start + M/2) = reg_C[14];
	*(C_start + 3*M/4) = reg_C[15];
}

__device__ void gemm_128_64x128(int M, int N, int K, float *A, float *B, float *C, float *sh){

	if (threadIdx.x>=128)
		return;

	float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[16];
	float4 reg_A[2];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*128;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*4*M);

	reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 8;
	reg_C[4] = *C_start;
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	C_start += (16*M - 8);
	reg_C[8] = *C_start;
	reg_C[9] = *(C_start + M/4);
	reg_C[10] = *(C_start + M/2);
	reg_C[11] = *(C_start + 3*M/4);

	C_start += 8;
	reg_C[12] = *C_start;
	reg_C[13] = *(C_start + M/4);
	reg_C[14] = *(C_start + M/2);
	reg_C[15] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load B from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + threadIdx.x*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
	*((float4*) (sh_B + 512 + 4*threadIdx.x)) = *(B_start + 1);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer_A + (threadIdx.x%8)*4;
		int B_offset = double_buffer_B + ((threadIdx.x/8)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = *((float4*)(sh_A+A_offset));
			reg_A[1] = *((float4*)(sh_A+A_offset+32));

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+256];
			reg_B[5] = sh_B[B_offset+260];
			reg_B[6] = sh_B[B_offset+264];
			reg_B[7] = sh_B[B_offset+268];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0].x, reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0].x, reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0].x, reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0].x, reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0].x, reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0].x, reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0].x, reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[1].x, reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[1].x, reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[1].x, reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[1].x, reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[1].x, reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[1].x, reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[1].x, reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[1].x, reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[0].y, reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[0].y, reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[0].y, reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[0].y, reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[0].y, reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[0].y, reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[0].y, reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[1].y, reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[1].y, reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[1].y, reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[1].y, reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[1].y, reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[1].y, reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[1].y, reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[1].y, reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[0].z, reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[0].z, reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[0].z, reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[0].z, reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[0].z, reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[0].z, reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[0].z, reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[1].z, reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[1].z, reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[1].z, reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[1].z, reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[1].z, reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[1].z, reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[1].z, reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[1].z, reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[0].w, reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[0].w, reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[0].w, reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[0].w, reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[0].w, reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[0].w, reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[0].w, reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[1].w, reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[1].w, reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[1].w, reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[1].w, reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[1].w, reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[1].w, reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[1].w, reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[1].w, reg_B[7], reg_C[15].w);

			A_offset += 64;
			if (i==3) B_offset += 508;
			B_offset += 1;
		}

		double_buffer_A ^= 512;
		double_buffer_B ^= 1024;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer_B + 4*threadIdx.x)) = *(B_start);
			*((float4*) (sh_B + double_buffer_B + 512 + 4*threadIdx.x)) = *(B_start + 1);
		}
				
	}
	C_start -= (16*M + 8);
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

	C_start += 8;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start += (16*M - 8);
	*(C_start) = reg_C[8];
	*(C_start + M/4) = reg_C[9];
	*(C_start + M/2) = reg_C[10];
	*(C_start + 3*M/4) = reg_C[11];

	C_start += 8;
	*(C_start) = reg_C[12];
	*(C_start + M/4) = reg_C[13];
	*(C_start + M/2) = reg_C[14];
	*(C_start + 3*M/4) = reg_C[15];
}

__device__ void gemm_128_128x64(int M, int N, int K, float *A, float *B, float *C, float *sh){

	if (threadIdx.x>=128)
		return;

    float *sh_A = sh;
	float *sh_B = sh + 2*128*8;

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*128;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	C_start += (8*M - 16);
	reg_C[8] = *(C_start);
	reg_C[9] = *(C_start + M/4);
	reg_C[10] = *(C_start + M/2);
	reg_C[11] = *(C_start + 3*M/4);

	C_start += 16;
	reg_C[12] = *(C_start);
	reg_C[13] = *(C_start + M/4);
	reg_C[14] = *(C_start + M/2);
	reg_C[15] = *(C_start + 3*M/4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%32)*4 + (threadIdx.x/32)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);
	*((float4*) (sh_A + 512 + 4*threadIdx.x)) = *(A_start + M);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x/64)*4 + (threadIdx.x%64)*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
		
	int double_buffer_A = 0;
	int double_buffer_B = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){

		__syncthreads();
		int A_offset = double_buffer_A + (threadIdx.x%16)*4;
		int B_offset = double_buffer_B + ((threadIdx.x/16)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+64];
			reg_A[5] = sh_A[A_offset+65];
			reg_A[6] = sh_A[A_offset+66];
			reg_A[7] = sh_A[A_offset+67];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+128];
			reg_B[5] = sh_B[B_offset+132];
			reg_B[6] = sh_B[B_offset+136];
			reg_B[7] = sh_B[B_offset+140];

			reg_C[0].x = fma(reg_A[0], reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0], reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0], reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0], reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0], reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0], reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0], reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0], reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[4], reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[4], reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[4], reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[4], reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[4], reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[4], reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[4], reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[4], reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[1], reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1], reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[1], reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1], reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[1], reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[1], reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[1], reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[1], reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[5], reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[5], reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[5], reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[5], reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[5], reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[5], reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[5], reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[5], reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[2], reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[2], reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[2], reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[2], reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[2], reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[2], reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[2], reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[2], reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[6], reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[6], reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[6], reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[6], reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[6], reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[6], reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[6], reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[6], reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[3], reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[3], reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[3], reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[3], reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[3], reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[3], reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[3], reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[3], reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[7], reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[7], reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[7], reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[7], reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[7], reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[7], reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[7], reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[7], reg_B[7], reg_C[15].w);

			A_offset += 128;
			if (i==3) B_offset += 252;
			B_offset += 1;
		}

		double_buffer_A ^= 1024;
		double_buffer_B ^= 512;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer_A + 4*threadIdx.x)) = *(A_start);
			*((float4*) (sh_A + double_buffer_A + 512 + 4*threadIdx.x)) = *(A_start + M);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer_B + 4*threadIdx.x)) = *(B_start);
		}
				
	}
	C_start -= (8*M + 16);
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

	C_start += 16;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start += (8*M - 16);
	*(C_start) = reg_C[8];
	*(C_start + M/4) = reg_C[9];
	*(C_start + M/2) = reg_C[10];
	*(C_start + 3*M/4) = reg_C[11];

	C_start += 16;
	*(C_start) = reg_C[12];
	*(C_start + M/4) = reg_C[13];
	*(C_start + M/2) = reg_C[14];
	*(C_start + 3*M/4) = reg_C[15];
}


__device__ void gemm_64_64x64(int M, int N, int K, float *A, float *B, float *C, float *sh){ 

	if (threadIdx.x>=64)
		return;
	
    float *sh_A = sh;
	float *sh_B = sh + 2*64*8;

	float4 reg_C[16];
	float reg_A[8];
	float reg_B[8];
	
	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*64;
	int block_base_y = blockIdx.x*64;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*4*M);

    reg_C[0] = *C_start;
	reg_C[1] = *(C_start + M/4);
	reg_C[2] = *(C_start + M/2);
	reg_C[3] = *(C_start + 3*M/4);

	C_start += 8;
	reg_C[4] = *(C_start);
	reg_C[5] = *(C_start + M/4);
	reg_C[6] = *(C_start + M/2);
	reg_C[7] = *(C_start + 3*M/4);

	C_start += (8*M - 8);
	reg_C[8] = *(C_start);
	reg_C[9] = *(C_start + M/4);
	reg_C[10] = *(C_start + M/2);
	reg_C[11] = *(C_start + 3*M/4);

	C_start += 8;
	reg_C[12] = *(C_start);
	reg_C[13] = *(C_start + M/4);
	reg_C[14] = *(C_start + M/2);
	reg_C[15] = *(C_start + 3*M/4);


	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%16)*4 + (threadIdx.x/16)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);
	*((float4*) (sh_A + 4*threadIdx.x + 256)) = *(A_start + M);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + threadIdx.x*K); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);
	*((float4*) (sh_B + 4*threadIdx.x + 256)) = *(B_start + 1);


	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%8)*4;
		int B_offset = double_buffer + ((threadIdx.x/8)*16);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+32];
			reg_A[5] = sh_A[A_offset+33];
			reg_A[6] = sh_A[A_offset+34];
			reg_A[7] = sh_A[A_offset+35];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+4];
			reg_B[2] = sh_B[B_offset+8];
			reg_B[3] = sh_B[B_offset+12];
			reg_B[4] = sh_B[B_offset+128];
			reg_B[5] = sh_B[B_offset+132];
			reg_B[6] = sh_B[B_offset+136];
			reg_B[7] = sh_B[B_offset+140];

			reg_C[0].x = fma(reg_A[0], reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[0], reg_B[1], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0], reg_B[2], reg_C[2].x);
			reg_C[3].x = fma(reg_A[0], reg_B[3], reg_C[3].x);
			reg_C[8].x = fma(reg_A[0], reg_B[4], reg_C[8].x);
			reg_C[9].x = fma(reg_A[0], reg_B[5], reg_C[9].x);
			reg_C[10].x = fma(reg_A[0], reg_B[6], reg_C[10].x);
			reg_C[11].x = fma(reg_A[0], reg_B[7], reg_C[11].x);
			reg_C[4].x = fma(reg_A[4], reg_B[0], reg_C[4].x);
			reg_C[5].x = fma(reg_A[4], reg_B[1], reg_C[5].x);
			reg_C[6].x = fma(reg_A[4], reg_B[2], reg_C[6].x);
			reg_C[7].x = fma(reg_A[4], reg_B[3], reg_C[7].x);
			reg_C[12].x = fma(reg_A[4], reg_B[4], reg_C[12].x);
			reg_C[13].x = fma(reg_A[4], reg_B[5], reg_C[13].x);
			reg_C[14].x = fma(reg_A[4], reg_B[6], reg_C[14].x);
			reg_C[15].x = fma(reg_A[4], reg_B[7], reg_C[15].x);

			reg_C[0].y = fma(reg_A[1], reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1], reg_B[1], reg_C[1].y);
			reg_C[2].y = fma(reg_A[1], reg_B[2], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1], reg_B[3], reg_C[3].y);
			reg_C[8].y = fma(reg_A[1], reg_B[4], reg_C[8].y);
			reg_C[9].y = fma(reg_A[1], reg_B[5], reg_C[9].y);
			reg_C[10].y = fma(reg_A[1], reg_B[6], reg_C[10].y);
			reg_C[11].y = fma(reg_A[1], reg_B[7], reg_C[11].y);
			reg_C[4].y = fma(reg_A[5], reg_B[0], reg_C[4].y);
			reg_C[5].y = fma(reg_A[5], reg_B[1], reg_C[5].y);
			reg_C[6].y = fma(reg_A[5], reg_B[2], reg_C[6].y);
			reg_C[7].y = fma(reg_A[5], reg_B[3], reg_C[7].y);
			reg_C[12].y = fma(reg_A[5], reg_B[4], reg_C[12].y);
			reg_C[13].y = fma(reg_A[5], reg_B[5], reg_C[13].y);
			reg_C[14].y = fma(reg_A[5], reg_B[6], reg_C[14].y);
			reg_C[15].y = fma(reg_A[5], reg_B[7], reg_C[15].y);

			reg_C[0].z = fma(reg_A[2], reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[2], reg_B[1], reg_C[1].z);
			reg_C[2].z = fma(reg_A[2], reg_B[2], reg_C[2].z);
			reg_C[3].z = fma(reg_A[2], reg_B[3], reg_C[3].z);
			reg_C[8].z = fma(reg_A[2], reg_B[4], reg_C[8].z);
			reg_C[9].z = fma(reg_A[2], reg_B[5], reg_C[9].z);
			reg_C[10].z = fma(reg_A[2], reg_B[6], reg_C[10].z);
			reg_C[11].z = fma(reg_A[2], reg_B[7], reg_C[11].z);
			reg_C[4].z = fma(reg_A[6], reg_B[0], reg_C[4].z);
			reg_C[5].z = fma(reg_A[6], reg_B[1], reg_C[5].z);
			reg_C[6].z = fma(reg_A[6], reg_B[2], reg_C[6].z);
			reg_C[7].z = fma(reg_A[6], reg_B[3], reg_C[7].z);
			reg_C[12].z = fma(reg_A[6], reg_B[4], reg_C[12].z);
			reg_C[13].z = fma(reg_A[6], reg_B[5], reg_C[13].z);
			reg_C[14].z = fma(reg_A[6], reg_B[6], reg_C[14].z);
			reg_C[15].z = fma(reg_A[6], reg_B[7], reg_C[15].z);

			reg_C[0].w = fma(reg_A[3], reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[3], reg_B[1], reg_C[1].w);
			reg_C[2].w = fma(reg_A[3], reg_B[2], reg_C[2].w);
			reg_C[3].w = fma(reg_A[3], reg_B[3], reg_C[3].w);
			reg_C[8].w = fma(reg_A[3], reg_B[4], reg_C[8].w);
			reg_C[9].w = fma(reg_A[3], reg_B[5], reg_C[9].w);
			reg_C[10].w = fma(reg_A[3], reg_B[6], reg_C[10].w);
			reg_C[11].w = fma(reg_A[3], reg_B[7], reg_C[11].w);
			reg_C[4].w = fma(reg_A[7], reg_B[0], reg_C[4].w);
			reg_C[5].w = fma(reg_A[7], reg_B[1], reg_C[5].w);
			reg_C[6].w = fma(reg_A[7], reg_B[2], reg_C[6].w);
			reg_C[7].w = fma(reg_A[7], reg_B[3], reg_C[7].w);
			reg_C[12].w = fma(reg_A[7], reg_B[4], reg_C[12].w);
			reg_C[13].w = fma(reg_A[7], reg_B[5], reg_C[13].w);
			reg_C[14].w = fma(reg_A[7], reg_B[6], reg_C[14].w);
			reg_C[15].w = fma(reg_A[7], reg_B[7], reg_C[15].w);

			A_offset += 64;
			if (i==3) B_offset += 252;
			B_offset += 1;
		}

		double_buffer ^= 512;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x + 256)) = *(A_start + M);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x + 256)) = *(B_start + 1);
		}
				
	}
	
	C_start -= (8*M + 8);
    *C_start = reg_C[0];
	*(C_start + M/4) = reg_C[1];
	*(C_start + M/2) = reg_C[2];
	*(C_start + 3*M/4) = reg_C[3];

	C_start += 8;
	*(C_start) = reg_C[4];
	*(C_start + M/4) = reg_C[5];
	*(C_start + M/2) = reg_C[6];
	*(C_start + 3*M/4) = reg_C[7];

	C_start += (8*M - 8);
	*(C_start) = reg_C[8];
	*(C_start + M/4) = reg_C[9];
	*(C_start + M/2) = reg_C[10];
	*(C_start + 3*M/4) = reg_C[11];

	C_start += 8;
	*(C_start) = reg_C[12];
	*(C_start + M/4) = reg_C[13];
	*(C_start + M/2) = reg_C[14];
	*(C_start + 3*M/4) = reg_C[15];

}
__device__ void gemm_64_16x16(int M, int N, int K, float *A, float *B, float *C, float *sh){

	if (threadIdx.x>=64)
		return;

	float *sh_A = sh;
	float *sh_B = sh + 2*16*8;
	
	float4 reg_C;

	float reg_A[8]={0.f};
	float reg_B[2]={0.f};

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*16;
	int block_base_y = blockIdx.x*16;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%4)*4 + (threadIdx.x/4)*M);

	reg_C = *C_start;

	//load A from global memory to shared memory
	float2 *A_start = (float2*) (A + block_base_y + (threadIdx.x%8)*2 + (threadIdx.x/8)*M); 
	*((float2*) (sh_A + 2*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float2 *B_start = (float2*) (B + K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K);
	*((float2*) (sh_B + 2*threadIdx.x)) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%4)*4;
		int B_offset = double_buffer + ((threadIdx.x/4)*2);
			
#pragma unroll
		for (int i=0; i<8; i+=2)	{
			
			reg_A[0] = sh_A[A_offset];
			reg_A[1] = sh_A[A_offset+1];
			reg_A[2] = sh_A[A_offset+2];
			reg_A[3] = sh_A[A_offset+3];
			reg_A[4] = sh_A[A_offset+16];
			reg_A[5] = sh_A[A_offset+17];
			reg_A[6] = sh_A[A_offset+18];
			reg_A[7] = sh_A[A_offset+19];

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+1];

			reg_C.x = fma(reg_A[0], reg_B[0], reg_C.x);
			reg_C.y = fma(reg_A[1], reg_B[0], reg_C.y);
			reg_C.z = fma(reg_A[2], reg_B[0], reg_C.z);
			reg_C.w = fma(reg_A[3], reg_B[0], reg_C.w);
			reg_C.x = fma(reg_A[4], reg_B[1], reg_C.x);
			reg_C.y = fma(reg_A[5], reg_B[1], reg_C.y);
			reg_C.z = fma(reg_A[6], reg_B[1], reg_C.z);
			reg_C.w = fma(reg_A[7], reg_B[1], reg_C.w);

			A_offset += 32;
			B_offset += 32;
		}

		double_buffer ^= 128;

		if (k+8 < K){
			A_start += 4*M; 
			*((float2*) (sh_A + double_buffer + 2*threadIdx.x)) = *(A_start);
			B_start += 4; 
			*((float2*) (sh_B + double_buffer + 2*threadIdx.x)) = *(B_start);
		}
	}

	
    *C_start = reg_C;
}


__device__ void gemm_64_32x32(int M, int N, int K, float *A, float *B, float *C, float *sh){

	if (threadIdx.x>=64)
		return;

	float *sh_A = sh;
	float *sh_B = sh + 2*32*8;

	float4 reg_C[4];
	float4 reg_A[2];
	float reg_B[2];

	// Compute block's starting coordinate
	int block_base_x = blockIdx.y*32;
	int block_base_y = blockIdx.x*32;

	//Load C from global memory to register file
	float4 *C_start = (float4*) (C + block_base_x*M + block_base_y + (threadIdx.x%4)*4 + (threadIdx.x/4)*M);

	reg_C[0] = *C_start;
	reg_C[1] = *(C_start + 4);
	reg_C[2] = *(C_start + 4*M);
	reg_C[3] = *(C_start + 4*M + 4);

	//load A from global memory to shared memory
	float4 *A_start = (float4*) (A + block_base_y + (threadIdx.x%8)*4 + (threadIdx.x/8)*M); 
	*((float4*) (sh_A + 4*threadIdx.x)) = *(A_start);

	//load A from global memory to shared memory
	float4 *B_start = (float4*) (B + K*block_base_x + (threadIdx.x%32)*K + (threadIdx.x/32)*4); 
	*((float4*) (sh_B + 4*threadIdx.x)) = *(B_start);

	int double_buffer = 0;
#pragma unroll
	for(int k=0; k<K; k+=8){
		__syncthreads();
		int A_offset = double_buffer + (threadIdx.x%4)*4;
		int B_offset = double_buffer + ((threadIdx.x/4)*4);
			
#pragma unroll
		for (int i=0; i<8; ++i)	{
			
			reg_A[0] = *((float4 *) (sh_A + A_offset));
			reg_A[1] = *((float4 *) (sh_A + A_offset + 16));

			reg_B[0] = sh_B[B_offset];
			reg_B[1] = sh_B[B_offset+64];

			reg_C[0].x = fma(reg_A[0].x, reg_B[0], reg_C[0].x);
			reg_C[1].x = fma(reg_A[1].x, reg_B[0], reg_C[1].x);
			reg_C[2].x = fma(reg_A[0].x, reg_B[1], reg_C[2].x);
			reg_C[3].x = fma(reg_A[1].x, reg_B[1], reg_C[3].x);

			reg_C[0].y = fma(reg_A[0].y, reg_B[0], reg_C[0].y);
			reg_C[1].y = fma(reg_A[1].y, reg_B[0], reg_C[1].y);
			reg_C[2].y = fma(reg_A[0].y, reg_B[1], reg_C[2].y);
			reg_C[3].y = fma(reg_A[1].y, reg_B[1], reg_C[3].y);

			reg_C[0].z = fma(reg_A[0].z, reg_B[0], reg_C[0].z);
			reg_C[1].z = fma(reg_A[1].z, reg_B[0], reg_C[1].z);
			reg_C[2].z = fma(reg_A[0].z, reg_B[1], reg_C[2].z);
			reg_C[3].z = fma(reg_A[1].z, reg_B[1], reg_C[3].z);

			reg_C[0].w = fma(reg_A[0].w, reg_B[0], reg_C[0].w);
			reg_C[1].w = fma(reg_A[1].w, reg_B[0], reg_C[1].w);
			reg_C[2].w = fma(reg_A[0].w, reg_B[1], reg_C[2].w);
			reg_C[3].w = fma(reg_A[1].w, reg_B[1], reg_C[3].w);

			A_offset += 32;
			if (i==3) B_offset += 124;
			B_offset += 1;
		}

		double_buffer ^= 256;

		if (k+8 < K){
			A_start += 2*M; 
			*((float4*) (sh_A + double_buffer + 4*threadIdx.x)) = *(A_start);

			B_start += 2; 
			*((float4*) (sh_B + double_buffer + 4*threadIdx.x)) = *(B_start);
		}
				
	}
	
    *C_start = reg_C[0];
	*(C_start + 4) = reg_C[1];
	*(C_start + 4*M) = reg_C[2];
	*(C_start + 4*M + 4) = reg_C[3];
}

template<int kThreads, int tile_Y, int tile_X>
__global__ void gemm(int M[], int N[], int K[], float *A[], float *B[], float *C[]);


template<>
__global__ void gemm<64, 16, 16>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];

	if (blockIdx.x * 16 < M[i] && blockIdx.y * 16 < N[i])	
		gemm_64_16x16(M[i], N[i], K[i], A[i], B[i], C[i], sh);

	return;
}

template<>
__global__ void gemm<64, 32, 32>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];

	if (blockIdx.x * 32 < M[i] && blockIdx.y * 32 < N[i])	
		gemm_64_32x32(M[i], N[i], K[i], A[i], B[i], C[i], sh);

	return;
}

template<>
__global__ void gemm<64, 64, 64>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];

	if (blockIdx.x * 64 < M[i] && blockIdx.y * 64 < N[i])	
		gemm_64_64x64(M[i], N[i], K[i], A[i], B[i], C[i], sh);

	return;
}

template<>
__global__ void gemm<128, 128, 64>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];

	if (blockIdx.x * 128 < M[i] && blockIdx.y * 64 < N[i])	
		gemm_128_128x64(M[i], N[i], K[i], A[i], B[i], C[i], sh);

	return;
}

template<>
__global__ void gemm<128, 64, 128>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];

	if (blockIdx.x * 64 < M[i] && blockIdx.y * 128 < N[i])	
		gemm_128_64x128(M[i], N[i], K[i], A[i], B[i], C[i], sh);

	return;
}

template<>
__global__ void gemm<256, 128, 128>(int M[], int N[], int K[], float *A[], float *B[], float *C[]){
	
	int i = blockIdx.z;
	extern __shared__ float sh[];

	if (blockIdx.x * 128 < M[i] && blockIdx.y * 128 < N[i])	
		gemm_256_128x128(M[i], N[i], K[i], A[i], B[i], C[i], sh);

	return;
}
