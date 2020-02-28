#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "cudnn.h"
#include "util.h"
#include "conv.h"
#include "activation.h"
#include "pooling.h"
#include "concat.h"
#include "dropout.h"
#include "lrn.h"
#include "loss.h"
#include "softmax.h"
#include "inception.h"
#include "batch-inception.h"


void batchGoogleNetForward(cudnnHandle_t handle, cublasHandle_t cublas_handle,
        int N, float **x, float **filter, float* buf, const int *algo_best) {
    int C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q;

    // conv1/7x7_s2
    C = 3;
    H = W = 227;
    K = 64;
    R = S = 7;
    U = V = 2;
    pad_h = pad_w = 3;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);

    int algo = algo_best[0];
    conv(handle, N, C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q,
            x[0], filter[0], buf, x[1], algo);

    // conv1/relu_7x7
    C = 64;
    H = W = 114;
    activation(handle, N, C, H, W, x[1], x[1]);

    // pool1/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[1], x[2]);

    H = P;
    W = Q;

    // pool1/norm1
    R = 5;
    S = 5;
    float lrnAlpha = 0.0001f;
    float lrnBeta = 0.75f;
    float lrnK = 2.f;

    lrn(handle, N, C, H, W, R, S, lrnAlpha, lrnBeta, lrnK, x[2], x[3]);

    // conv2/3x3_reduce
    K = 64;
    R = 1;
    S = 1;
    U = 1;
    V = 1;
    pad_h = 0;
    pad_w = 0;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);

    algo = algo_best[7];
    conv(handle, N, C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q,
            x[3], filter[1], buf, x[4], algo);
    C = K;
    H = P;
    W = Q;

    // conv2/relu_3x3_reduce
    activation(handle, N, C, H, W, x[4], x[4]);

    // conv2/3x3
    K = 192;
    R = 3;
    S = 3;
    U = 1;
    V = 1;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);

    algo = algo_best[14];
    conv(handle, N, C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q,
            x[4], filter[2], buf, x[5], algo);
    C = K;
    H = P;
    W = Q;


    // conv2/relu_3x3
    activation(handle, N, C, H, W, x[5], x[5]);

    // conv2/norm2
    R = 5;
    S = 5;
    lrnAlpha = 0.0001f;
    lrnBeta = 0.75f;
    lrnK = 2.f;

    lrn(handle, N, C, H, W, R, S, lrnAlpha, lrnBeta, lrnK, x[5], x[6]);

    // pool2/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 0;
    pad_w = 0;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[6], x[7]);

    // inception3a
    H = P;
    W = Q;
    batchGoogleNetInception(handle, N, C, H, W, 7, 3,
            64, 96, 128, 16, 32, 32, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // inception3b
    batchGoogleNetInception(handle, N, C, H, W, 11, 9,
            128, 128, 192, 32, 96, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // pool3/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[15], x[16]);

    // inception4a
    H = P;
    W = Q;
    batchGoogleNetInception(handle, N, C, H, W, 16, 15,
            192, 96, 208, 16, 48, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // inception4b
    batchGoogleNetInception(handle, N, C, H, W, 20, 21,
            160, 112, 224, 24, 64, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // inception4c
    batchGoogleNetInception(handle, N, C, H, W, 24, 27,
            128, 128, 256, 24, 64, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // inception4d
    batchGoogleNetInception(handle, N, C, H, W, 28, 33,
            112, 144, 288, 32, 64, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // inception4e
    batchGoogleNetInception(handle, N, C, H, W, 32, 39,
            256, 160, 320, 32, 128, 128, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // pool4/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[36], x[37]);

    // inception5a
    H = P;
    W = Q;
    batchGoogleNetInception(handle, N, C, H, W, 37, 45,
            256, 160, 320, 32, 128, 128, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // inception5b
    batchGoogleNetInception(handle, N, C, H, W, 41, 51,
            384, 192, 384, 48, 128, 128, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, algo_best);

    // pool5/3x3_s2
    R = 7;
    S = 7;
    U = 1;
    V = 1;
    pad_h = 0;
    pad_w = 0;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[45], x[46]);

    // loss3
    K = 1000;
    loss(cublas_handle, N, C, K, x[46], filter[57], x[47]);

    // softmax
    softmax(handle, N, C, x[47], x[48]);
}


void cudnnGoogleNetForward(cudnnHandle_t handle, cublasHandle_t cublas_handle,
        int N, float **x, float** filter, float* buf, const int *algo_best) {
    int C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q;

    // conv1/7x7_s2
    C = 3;
    H = W = 227;
    K = 64;
    R = S = 7;
    U = V = 2;
    pad_h = pad_w = 3;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);

    int algo = algo_best[0];
    conv(handle, N, C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q,
            x[0], filter[0], buf, x[1], algo);

    // conv1/relu_7x7
    C = 64;
    H = W = 114;
    activation(handle, N, C, H, W, x[1], x[1]);

    // pool1/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[1], x[2]);

    H = P;
    W = Q;

    // pool1/norm1
    R = 5;
    S = 5;
    float lrnAlpha = 0.0001f;
    float lrnBeta = 0.75f;
    float lrnK = 2.f;

    lrn(handle, N, C, H, W, R, S, lrnAlpha, lrnBeta, lrnK, x[2], x[3]);

    // conv2/3x3_reduce
    K = 64;
    R = 1;
    S = 1;
    U = 1;
    V = 1;
    pad_h = 0;
    pad_w = 0;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);

    algo = algo_best[7];
    conv(handle, N, C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q,
            x[3], filter[1], buf, x[4], algo);
    C = K;
    H = P;
    W = Q;

    // conv2/relu_3x3_reduce
    activation(handle, N, C, H, W, x[4], x[4]);

    // conv2/3x3
    K = 192;
    R = 3;
    S = 3;
    U = 1;
    V = 1;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);

    algo = algo_best[14];
    conv(handle, N, C, H, W, K, R, S, U, V, pad_h, pad_w, P, Q,
            x[4], filter[2], buf, x[5], algo);
    C = K;
    H = P;
    W = Q;


    // conv2/relu_3x3
    activation(handle, N, C, H, W, x[5], x[5]);

    // conv2/norm2
    R = 5;
    S = 5;
    lrnAlpha = 0.0001f;
    lrnBeta = 0.75f;
    lrnK = 2.f;

    lrn(handle, N, C, H, W, R, S, lrnAlpha, lrnBeta, lrnK, x[5], x[6]);

    // pool2/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 0;
    pad_w = 0;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[6], x[7]);

#ifdef USE_MULTI_STREAM
    cudaStream_t s[4];
    ErrChk(cudaStreamCreate(&s[0]));
    ErrChk(cudaStreamCreate(&s[1]));
    ErrChk(cudaStreamCreate(&s[2]));
    ErrChk(cudaStreamCreate(&s[3]));
#else
    cudaStream_t s[4] = {0, 0, 0, 0};
#endif

    // inception3a
    H = P;
    W = Q;
    cudnnGoogleNetInception(handle, N, C, H, W, 7, 3,
            64, 96, 128, 16, 32, 32, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // inception3b
    cudnnGoogleNetInception(handle, N, C, H, W, 11, 9,
            128, 128, 192, 32, 96, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // pool3/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[15], x[16]);

    // inception4a
    H = P;
    W = Q;
    cudnnGoogleNetInception(handle, N, C, H, W, 16, 15,
            192, 96, 208, 16, 48, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // inception4b
    cudnnGoogleNetInception(handle, N, C, H, W, 20, 21,
            160, 112, 224, 24, 64, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // inception4c
    cudnnGoogleNetInception(handle, N, C, H, W, 24, 27,
            128, 128, 256, 24, 64, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // inception4d
    cudnnGoogleNetInception(handle, N, C, H, W, 28, 33,
            112, 144, 288, 32, 64, 64, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // inception4e
    cudnnGoogleNetInception(handle, N, C, H, W, 32, 39,
            256, 160, 320, 32, 128, 128, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // pool4/3x3_s2
    R = 3;
    S = 3;
    U = 2;
    V = 2;
    pad_h = 1;
    pad_w = 1;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[36], x[37]);

    // inception5a
    H = P;
    W = Q;
    cudnnGoogleNetInception(handle, N, C, H, W, 37, 45,
            256, 160, 320, 32, 128, 128, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // inception5b
    cudnnGoogleNetInception(handle, N, C, H, W, 41, 51,
            384, 192, 384, 48, 128, 128, // K1, K2, K3, K4, K5, K6
            &C, x, filter, buf, s, algo_best);

    // pool5/3x3_s2
    R = 7;
    S = 7;
    U = 1;
    V = 1;
    pad_h = 0;
    pad_w = 0;
    P = ceil((float)(H - R + 1 + 2 * pad_h)/(float)U);
    Q = ceil((float)(W - S + 1 + 2 * pad_w)/(float)V);
    pooling(handle, N, C, H, W, R, S, U, V, pad_h, pad_w, P, Q, x[45], x[46]);

    // loss3
    K = 1000;
    loss(cublas_handle, N, C, K, x[46], filter[57], x[47]);

    // softmax
    softmax(handle, N, C, x[47], x[48]);

#ifdef USE_MULTI_STREAM
    ErrChk(cudaStreamDestroy(s[0]));
    ErrChk(cudaStreamDestroy(s[1]));
    ErrChk(cudaStreamDestroy(s[2]));
    ErrChk(cudaStreamDestroy(s[3]));
#endif
}

const int algo_best[7*57] = {
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        6, 6, 6, 6, 7, 7, 5,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 1, 0, 0, 1, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 5, 5,
        0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 5, 5, 5, 5,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 0, 0, 0, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 7, 7, 7, 7,
        1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1,
        0, 1, 1, 1, 0, 0, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 0, 0,
        0, 7, 0, 7, 7, 7, 7,
        1, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 7, 7, 7, 7,
        1, 0, 1, 0, 0, 0, 1,
        0, 1, 1, 1, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 0, 0,
        7, 7, 7, 7, 7, 7, 7,
        1, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 1, 1, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 0, 0,
        7, 7, 7, 7, 7, 7, 7,
        0, 0, 1, 1, 0, 0, 0,
        1, 1, 0, 1, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 1,
        6, 6, 6, 6, 7, 7, 7,
        0, 0, 0, 0, 0, 1, 0,
        7, 7, 7, 7, 7, 7, 4,
        0, 1, 0, 0, 1, 1, 0,
        1, 1, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 1, 1, 0,
        6, 6, 6, 7, 7, 7, 7,
        1, 1, 1, 1, 1, 0, 1,
        7, 7, 7, 7, 7, 7, 4,
        0, 1, 0, 0, 1, 1, 0
};

int main() {
    const int warmupIters = 2;
    const int TestIters = 10;

    int N = 1; // batch size
    const int filterNum = 58;
    const int xNum = 50;

    float **filter = new float*[filterNum]; // filter
    float **x = new float*[xNum]; // result

    const int MAX_TENSOR_SIZE=N * 200704 * 9;
    ErrChk(cudaMalloc(&x[0], (xNum + 10) *MAX_TENSOR_SIZE * sizeof(float)));
    for (int i = 1; i < xNum; ++i) {
        x[i] = x[i - 1] + MAX_TENSOR_SIZE;
    }
    float *buf = x[xNum - 1] + MAX_TENSOR_SIZE;

    const int MAX_FILTER_SIZE = 8000000;
    ErrChk(cudaMalloc(&filter[0], filterNum * MAX_FILTER_SIZE * sizeof(float)));
    for (int i = 1; i < filterNum; ++i) {
        filter[i] = filter[i - 1] + MAX_FILTER_SIZE;
    }

    const int RESULT_SIZE=1000;
    float *h_cudnn_result = (float*)malloc(2 * RESULT_SIZE * sizeof(float));
    float *h_our_result = h_cudnn_result + RESULT_SIZE;

    // prepare data
    float *h_input = (float*) malloc(MAX_TENSOR_SIZE * sizeof(float));
    for (int j = 0; j < MAX_TENSOR_SIZE; ++j)
        h_input[j] = j%10;
    float *h_filter = (float*) malloc(
            filterNum * MAX_FILTER_SIZE * sizeof(float));
    for (int j = 0; j < filterNum*MAX_FILTER_SIZE; ++j)
        h_filter[j] = j%5;
    ErrChk(cudaMemcpy(x[0], h_input, MAX_TENSOR_SIZE * sizeof(float),
                cudaMemcpyHostToDevice));
    ErrChk(cudaMemcpy(filter[0], h_filter,
                filterNum * MAX_FILTER_SIZE * sizeof(float),
                cudaMemcpyHostToDevice));

    cudnnHandle_t handle;
    cublasHandle_t cublas_handle;
    ErrChk(cudnnCreate(&handle));
    ErrChk(cublasCreate(&cublas_handle));

    // warm up
    for (int i = 0; i < warmupIters; ++i) {
        cudnnGoogleNetForward(handle, cublas_handle, N, x, filter, buf,
                algo_best);
    }

    cudaEvent_t start, stop;
    float elapsedTime = 0;
    ErrChk(cudaEventCreate(&start));
    ErrChk(cudaEventCreate(&stop));
    ErrChk(cudaEventRecord(start,0));

    for (int i = 0; i < TestIters; ++i) {
        cudnnGoogleNetForward(handle, cublas_handle, N, x, filter, buf,
                algo_best);
    }

    ErrChk(cudaEventRecord(stop, 0));
    ErrChk(cudaEventSynchronize(stop));
    ErrChk(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("Time for cuDNN implementation is %0.6f\n", elapsedTime / TestIters);
    ErrChk(cudaMemcpy(h_cudnn_result, x[48], RESULT_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost));

    // warm up
    for (int i = 0; i < warmupIters; ++i) {
        batchGoogleNetForward(handle, cublas_handle, N, x, filter, buf,
                algo_best);
    }

    ErrChk(cudaEventRecord(start,0));
    for (int i = 0; i < TestIters; ++i) {
        batchGoogleNetForward(handle, cublas_handle, N, x, filter, buf,
                algo_best);
    }

    ErrChk(cudaEventRecord(stop, 0));
    ErrChk(cudaEventSynchronize(stop));
    ErrChk(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time for batched-Conv implementation is %0.6f\n",
            elapsedTime / TestIters);
    ErrChk(cudaMemcpy(h_our_result, x[48], RESULT_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost));

    // compare the result
    double ep = 0.0001;
    for (int i = 0; i < RESULT_SIZE; ++i) {
       if (std::abs(h_our_result[i] - (double)h_cudnn_result[i]) > ep) {
           printf("result error at %d: %f, %f\n", i, h_our_result[i],
                                                  h_cudnn_result[i]);
           return -1;
       }
    }
    printf("result is correctly!\n");

    ErrChk(cublasDestroy(cublas_handle));
    ErrChk(cudnnDestroy(handle));

    return 0;
}
