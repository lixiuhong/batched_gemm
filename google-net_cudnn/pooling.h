#ifndef __POOLING_H__
#define __POOLING_H__
void pooling(cudnnHandle_t handle, int N, int C, int H, int W, int R, int S, int U, int V, int pad_h, int pad_w, int P, int Q, float *input, float *output, cudaStream_t s=0);
#endif
