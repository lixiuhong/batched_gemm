/*
 * inception.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef INCEPTION_H_
#define INCEPTION_H_

/*
 * Do Inception
 *
 * This func will consume 6 filters and 8 features.
 * Use feature[featureIndex] as input, which should be set before this func.
 * Use feature[featureIndex + 8] as output.
 *
 */
void cudnnGoogleNetInception(cudnnHandle_t handle, const int N, const int C,
        const int H, const int W, const int xIdx, const int filterIdx,
        const int K1, const int K2, const int K3, const int K4, const int K5,
        const int K6, int *reC, float **x, float** filter, float* buf,
        cudaStream_t *s, const int *algo_best);

#endif /* INCEPTION_H_ */
