/*
 * softmax.h
 *
 *  Created on: Nov 5, 2018
 *      Author: cambricon
 */

#ifndef SOFTMAX_H_
#define SOFTMAX_H_


void softmax(cudnnHandle_t handle, int N, int C, float *input, float *output);


#endif /* SOFTMAX_H_ */
