#include <algorithm>

#include "cudnn.h"
#include "util.h"
#include <cmath>
#include "conv.h"
#include "pooling.h"
#include "activation.h"
#include "dropout.h"
#include "lrn.h"
#include "concat.h"
#include "im2col.h"
#include "gemm_kernel.h"


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
        const int *algo_best) {
    /*
     * Use x[xIdx + 8] as output.
     * We can concat the result directly when N == 1.
     */
    float *output = x[xIdx + 4];
    float *output1 = output;
    float *output2 = output1 + K1 * H * W;
    float *output3 = output2 + K3 * H * W;
    float *output4 = output3 + K5 * H * W;

    //pool
    pooling(handle, N, C, H, W, 3, 3, 1, 1, 1, 1, H, W,
            x[xIdx], x[xIdx + 3]);

    // the first four-batch conv
    int M_MAX = N * H * W;
    int N_MAX = std::max(K1, std::max(K2, std::max(K3, K4)));
    dim3 grid_size((M_MAX - 1) / 16 + 1, (N_MAX - 1) / 16 + 1, 4);
   	dim3 block_size(64, 1, 1);
    gemm_4<<<grid_size, block_size, (1U << 9) * sizeof(float)>>>(
            N * H * W, K1, K2, K4, K6, C, H, W, x[xIdx], x[xIdx + 3],
            filter[filterIdx], filter[filterIdx + 1], filter[filterIdx + 3],
            filter[filterIdx + 5], output1, x[xIdx + 1], x[xIdx + 2], output4);
   	KernelErrChk();

    //relu 1*1
    activation(handle, N, K1, H, W, output1, output1);

    //relu 3*3 reduce
    activation(handle, N, K2, H, W, x[xIdx + 1],
            x[xIdx + 1]);

    //3*3
    int algo = algo_best[(filterIdx+2)*7];
    conv(handle, N, C, H, W, K3, 3, 3, 1, 1, 1, 1, H, W,
            x[xIdx + 1], filter[filterIdx+2], buf,
            output2, algo);

    //relu 3*3
    activation(handle, N, K3, H, W, output2, output2);

    //relu 5*5 reduce
    activation(handle, N, K4, H, W, x[xIdx + 2],
            x[xIdx + 2]);

    //5*5
    algo = algo_best[(filterIdx+4)*7];
    conv(handle, N, C, H, W, K5, 5, 5, 1, 1, 2, 2, H, W,
            x[xIdx + 2], filter[filterIdx+4], buf,
            output3, algo);

    //relu 5*5
    activation(handle, N, K5, H, W, output3, output3);

    //relu pool proj
    activation(handle, N, K6, H, W, output4, output4);

    //compute return shape
    *reC = K1 + K3 + K5 + K6;
}
