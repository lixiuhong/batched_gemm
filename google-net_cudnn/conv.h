#ifndef __CONV_H__
#define __CONV_H__
void conv(cudnnHandle_t handle, int N, int C, int H, int W, int K, int R, int S,
        int U, int V, int pad_h, int pad_w, int P, int Q,
        float *input, float *filter,
        float *buf, float *output,
        int algo,
        cudaStream_t s=0);
#endif
