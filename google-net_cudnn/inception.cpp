#include "cudnn.h"
#include "util.h"
#include <cmath>
#include "conv.h"
#include "pooling.h"
#include "activation.h"
#include "dropout.h"
#include "lrn.h"
#include "concat.h"

/*
 * Do Inception
 *
 * This func will consume 6 filters and 4 feature(x).
 * Use x[xIdx] as input, which should be set before this func.
 * Use x[xIdx + 4] as output.
 *
 */
void cudnnGoogleNetInception(cudnnHandle_t handle, const int N, const int C,
        const int H, const int W, const int xIdx, const int filterIdx,
        const int K1, const int K2, const int K3, const int K4, const int K5,
        const int K6, int *reC, float **x, float** filter, float* buf,
        cudaStream_t *s, const int *algo_best) {
    /*
     * Use x[xIdx + 4] as output.
     * We can concat the result directly when N == 1.
     */
    float *output = x[xIdx + 4];
    float *output1 = output;
    float *output2 = output1 + K1 * H * W;
    float *output3 = output2 + K3 * H * W;
    float *output4 = output3 + K5 * H * W;

    //1*1 conv
    int algo = algo_best[filterIdx*7];
    conv(handle, N, C, H, W, K1, 1, 1, 1, 1, 0, 0, H, W,
            x[xIdx], filter[filterIdx], buf,
            output1, algo, s[0]);

    //relu 1*1
    activation(handle, N, K1, H, W, output1, output1, s[0]);

    //3*3 reduce
    algo = algo_best[(filterIdx+1)*7];
    conv(handle, N, C, H, W, K2, 1, 1, 1, 1, 0, 0, H, W,
            x[xIdx], filter[filterIdx+1], buf,
            x[xIdx + 1], algo, s[1]);

    //relu 3*3 reduce
    activation(handle, N, K2, H, W, x[xIdx + 1],
            x[xIdx + 1], s[1]);

    //3*3
    algo = algo_best[(filterIdx+2)*7];
    conv(handle, N, C, H, W, K3, 3, 3, 1, 1, 1, 1, H, W,
            x[xIdx + 1], filter[filterIdx+2], buf,
            output2, algo, s[1]);

    //relu 3*3
    activation(handle, N, K3, H, W, output2, output2, s[1]);

    //5*5 reduce
    algo = algo_best[(filterIdx+3)*7];
    conv(handle, N, C, H, W, K4, 1, 1, 1, 1, 0, 0, H, W,
            x[xIdx], filter[filterIdx+2], buf,
            x[xIdx + 2], algo, s[2]);

    //relu 5*5 reduce
    activation(handle, N, K4, H, W, x[xIdx + 2],
            x[xIdx + 2], s[2]);

    //5*5
    algo = algo_best[(filterIdx+4)*7];
    conv(handle, N, C, H, W, K5, 5, 5, 1, 1, 2, 2, H, W,
            x[xIdx + 2], filter[filterIdx+4], buf,
            output3, algo, s[2]);

    //relu 5*5
    activation(handle, N, K5, H, W, output3, output3, s[2]);

    //pool
    pooling(handle, N, C, H, W, 3, 3, 1, 1, 1, 1, H, W,
            x[xIdx], x[xIdx + 3], s[3]);

    //pool proj
    algo = algo_best[(filterIdx+5)*7];
    conv(handle, N, C, H, W, K6, 1, 1, 1, 1, 0, 0, H, W,
            x[xIdx + 3], filter[filterIdx+5], buf, output4, algo, s[3]);

    //relu pool proj
    activation(handle, N, K6, H, W, output4, output4, s[3]);


    ErrChk(cudaDeviceSynchronize());

    *reC = K1 + K3 + K5 + K6;
}
