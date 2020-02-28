/*
 * batch-inception.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef BATCH_INCEPTION_H_
#define BATCH_INCEPTION_H_

/*
 * do Inception
 *
 * This func will consume 6 filters and 4 features.
 * Use x[xIdx] as x[xIdx], which should be set before this func.
 * Use x[xIdx + 4] as output.
 *
 */
void batchGoogleNetInception(cudnnHandle_t handle, const int N, const int C,
        const int H, const int W, const int xIdx, const int filterIdx,
        const int K1, const int K2, const int K3, const int K4, const int K5,
        const int K6, int *reC, float **x, float** filter, float* buf,
        const int *algo_best);

#endif /* BATCH_INCEPTION_H_ */
