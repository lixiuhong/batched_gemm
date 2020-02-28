#ifndef __LRN_H__
#define __LRN_H__
void lrn(cudnnHandle_t handle, int N, int C, int H, int W, int R, int S, float lrnAlpha, float lrnBeta, float lrnK, float *input, float *output);
#endif
