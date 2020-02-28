/*
 * loss.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef LOSS_H_
#define LOSS_H_


void loss(cublasHandle_t cublas_handle, int N, int C, int K, float *input,
        float *filter, float *output);


#endif /* LOSS_H_ */
