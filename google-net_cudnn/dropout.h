#ifndef __DROPOUT_H__
#define __DROPOUT_H__
void dropout(cudnnHandle_t handle, float dropout, int N, int C, int H, int W,
        float *input, float *buf, float *output);
#endif
