#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
void activation(cudnnHandle_t handle, int N, int C, int H, int W, float *input, float *output, cudaStream_t s=0);
#endif
