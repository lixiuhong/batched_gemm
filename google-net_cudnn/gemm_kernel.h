/*
 * gemm_kernel.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef GEMM_KERNEL_H_
#define GEMM_KERNEL_H_


//(N*P*Q)%16==0 && (P*Q)%4==0
__device__ void gemm_64_16x16_1(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

	float* sh_A = sh;
    float* sh_B = sh + 2*16*8;

    float4 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;
	reg_C.z =0.f;
	reg_C.w =0.f;

    float reg_A[8];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*16;
    int block_base_y = blockIdx.x*16;

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
        for (int i=0; i<8; i+=2)    {

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

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.x;
    C[C_offset+1] = reg_C.y;
    C[C_offset+2] = reg_C.z;
    C[C_offset+3] = reg_C.w;
}

//(N*P*Q)%16==0 && (P*Q)%4!=0
__device__ void gemm_64_16x16_2(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

	float* sh_A = sh;
    float* sh_B = sh + 2*16*8;

    float4 reg_C;
	reg_C.x =0.f;
	reg_C.y =0.f;
	reg_C.z =0.f;
	reg_C.w =0.f;

    float reg_A[8];
    float reg_B[2];

    // Compute block's starting coordinate
    int block_base_x = blockIdx.y*16;
    int block_base_y = blockIdx.x*16;

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
        for (int i=0; i<8; i+=2)    {

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

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.x;
    C_offset = (ind+1)/(P*Q)*(P*Q*N) + (ind+1)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.y;
    C_offset = (ind+2)/(P*Q)*(P*Q*N) + (ind+2)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.z;
    C_offset = (ind+3)/(P*Q)*(P*Q*N) + (ind+3)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
    C[C_offset] = reg_C.w;
}

//(N*P*Q%16!=0)
__device__ void gemm_64_16x16_3(int M, int N, int K, int P, int Q, float *A, float *B, float *C, float *sh){

   float* sh_A = sh;
   float* sh_B = sh + 2*16*8;

   float reg_C[4];
   reg_C[0] = 0.f;
   reg_C[1] = 0.f;
   reg_C[2] = 0.f;
   reg_C[3] = 0.f;

   float reg_A[8]={0.f};
   float reg_B[2]={0.f};

   // Compute block's starting coordinate
   int block_base_x = blockIdx.y*16;
   int block_base_y = blockIdx.x*16;


   //load A from global memory to shared memory
   int A_offset = block_base_y + (threadIdx.x%16) + (threadIdx.x/16)*M;
   sh_A[threadIdx.x] = A[A_offset%(M*K)];
   sh_A[threadIdx.x+64] = A[(A_offset+4*M)%(M*K)];

   //load A from global memory to shared memory
   int B_offset =  K*block_base_x + (threadIdx.x/16)*2 + (threadIdx.x%16)*K;
   sh_B[threadIdx.x*2] = B[B_offset%(K*N)];
   sh_B[threadIdx.x*2+1] = B[(B_offset+1)%(K*N)];

   int double_buffer = 0;
#pragma unroll
   for(int k=0; k<K; k+=8){
       __syncthreads();
       int shA_offset = double_buffer + (threadIdx.x%4)*4;
       int shB_offset = double_buffer + ((threadIdx.x/4)*2);
#pragma unroll
       for (int i=0; i<8; i+=2)    {

           reg_A[0] = sh_A[shA_offset];
           reg_A[1] = sh_A[shA_offset+1];
           reg_A[2] = sh_A[shA_offset+2];
           reg_A[3] = sh_A[shA_offset+3];
           reg_A[4] = sh_A[shA_offset+16];
           reg_A[5] = sh_A[shA_offset+17];
           reg_A[6] = sh_A[shA_offset+18];
           reg_A[7] = sh_A[shA_offset+19];

           reg_B[0] = sh_B[shB_offset];
           reg_B[1] = sh_B[shB_offset+1];

           reg_C[0] = fma(reg_A[0], reg_B[0], reg_C[0]);
           reg_C[1] = fma(reg_A[1], reg_B[0], reg_C[1]);
           reg_C[2] = fma(reg_A[2], reg_B[0], reg_C[2]);
           reg_C[3] = fma(reg_A[3], reg_B[0], reg_C[3]);
           reg_C[0] = fma(reg_A[4], reg_B[1], reg_C[0]);
           reg_C[1] = fma(reg_A[5], reg_B[1], reg_C[1]);
           reg_C[2] = fma(reg_A[6], reg_B[1], reg_C[2]);
           reg_C[3] = fma(reg_A[7], reg_B[1], reg_C[3]);

           shA_offset += 32;
           shB_offset += 32;
       }

       double_buffer ^= 128;
       double_buffer ^= 128;

       if (k+8 < K){
           A_offset += 8*M;
           sh_A[double_buffer+threadIdx.x] = A[A_offset%(M*K)];
           sh_A[double_buffer+threadIdx.x+64] = A[(A_offset+4*M)%(M*K)];
           B_offset += 8;
           sh_B[double_buffer+threadIdx.x*2] = B[B_offset%(K*N)];
           sh_B[double_buffer+threadIdx.x*2+1] = B[(B_offset+1)%(K*N)];
       }
   }

	int ind = blockIdx.x*16 + (threadIdx.x%4)*4;
    int C_offset = ind/(P*Q)*(P*Q*N) + ind%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);

   if (blockIdx.x<M/16){
       C[C_offset] = reg_C[0];
     	C_offset = (ind+1)/(P*Q)*(P*Q*N) + (ind+1)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
       C[C_offset] = reg_C[1];
     	C_offset = (ind+2)/(P*Q)*(P*Q*N) + (ind+2)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
       C[C_offset] = reg_C[2];
     	C_offset = (ind+3)/(P*Q)*(P*Q*N) + (ind+3)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
       C[C_offset] = reg_C[3];
   }
   else{
       int ruler = (threadIdx.x%4)*4;
       int rag = M%16;
       if ((ruler)<rag){
           C[C_offset] = reg_C[0];
		}
       if ((ruler+1)<rag){
     		C_offset = (ind+1)/(P*Q)*(P*Q*N) + (ind+1)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
           C[C_offset] = reg_C[1];
		}
       if ((ruler+2)<rag){
     	C_offset = (ind+2)/(P*Q)*(P*Q*N) + (ind+2)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
           C[C_offset] = reg_C[2];
		}
       if ((ruler+3)<rag){
     		C_offset = (ind+3)/(P*Q)*(P*Q*N) + (ind+3)%(P*Q) + (threadIdx.x/4)*(P*Q) + blockIdx.y*16*(P*Q);
           C[C_offset] = reg_C[3];
		}
   }
}



__global__ void gemm_2(int M1, int M2, int N1, int N2, int K1, int K2, int P, int Q, float *A1, float *A2, float *B1, float *B2, float *C1, float *C2){

	int id = blockIdx.z;

    extern __shared__ float sh[];

    int M = (id==0)?(M1):(M2);
    int N = (id==0)?(N1):(N2);
    int K = (id==0)?(K1):(K2);
    float *A = (id==0)?(A1):(A2);
    float *B = (id==0)?(B1):(B2);
    float *C = (id==0)?(C1):(C2);

    if (blockIdx.x*16 < (M + (M%16!=0)*16) && blockIdx.y*16 < (N + (N%16!=0)*16)){
   		if (M%16==0 && P%2==0){
   			//(N*P*Q)%16==0 && (P*Q)%4==0
   			gemm_64_16x16_1(M, N, K, P, Q, A, B, C, sh);
   		}
   		else if (M%16==0){
    		//(N*P*Q)%16==0 && (P*Q)%4!=0
   			gemm_64_16x16_2(M, N, K, P, Q, A, B, C, sh);
   		}
   		else{
   			//(N*P*Q%16!=0)
   			gemm_64_16x16_3(M, N, K, P, Q, A, B, C, sh);
    	}
    }
}



__global__ void gemm_4(int M, int N1, int N2, int N3, int N4, int K, int P, int Q, float *A1, float *A2, float *B1, float *B2, float *B3, float *B4, float *C1, float *C2, float *C3, float *C4){

	int id = blockIdx.z;
    extern __shared__ float sh[];

    int N;
    float *A, *B, *C;

    switch(id){
    case 0:
    	N = N1;
    	A = A1;
    	B = B1;
    	C = C1;
    	break;
    case 1:
    	N = N2;
    	A = A1;
    	B = B2;
    	C = C2;
    	break;
    case 2:
    	N = N3;
    	A = A1;
    	B = B3;
    	C = C3;
    	break;
    case 3:
    	N = N4;
    	A = A2;
    	B = B4;
    	C = C4;
    	break;
    }

    if (blockIdx.x*16 < (M + (M%16!=0)*16) && blockIdx.y*16 < (N + (N%16!=0)*16)){
   		if (M%16==0 && P%2==0){
   			//(N*P*Q)%16==0 && (P*Q)%4==0
   			gemm_64_16x16_1(M, N, K, P, Q, A, B, C, sh);
   		}
   		else if (M%16==0){
    		//(N*P*Q)%16==0 && (P*Q)%4!=0
   			gemm_64_16x16_2(M, N, K, P, Q, A, B, C, sh);
   		}
   		else{
   			//(N*P*Q%16!=0)
   			gemm_64_16x16_3(M, N, K, P, Q, A, B, C, sh);
    	}
    }
}



#endif /* GEMM_KERNEL_H_ */
