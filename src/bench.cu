#include "../include/bench.cuh"

#include "../include/spmv_gpu.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif

#include "../include/spmv_cpu.h"

#ifdef __cplusplus
}
#endif

void print_results(const struct results res, const char *fmt)
{
    puts("--");
    printf("format   : %s\n", fmt);
    printf("time     : %f\n", res.exec_time);
    //printf("bandwidth: %f\n", res.mem_bandwidth);
    printf("gflops   : %f\n", res.gflops);
    printf("correct  : %s\n", res.is_correct ? "yes" : "no");
    puts("--");
}


struct results spmv_gpu_coo_prof(const int * I,
                                 const int * J,
                                 const float * val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const float * X)
{
    struct results res = { -1, -1, 1 };

    float *Y, *Ybis;

    Y = (float *)calloc(M, sizeof(float));
    Ybis = (float *)calloc(M, sizeof(float));

    int *dI, *dJ;
    float *dval, *dX, *dY;

    cudaMalloc(&dI, nz * sizeof(int));
    cudaMalloc(&dJ, nz * sizeof(int));
    cudaMalloc(&dval, nz * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dI, I, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, val, nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int threadsPerBlock = min(1024, maxThreadsPerBlock);
    const int numBlocks = (nz + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum = 0.f;
    for (int k = 0; k < WARMUP; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        spmv_gpu_coo<<<numBlocks, threadsPerBlock>>>(dI, dJ, dval, nz, dX, dY);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    cudaDeviceSynchronize();

    for (int k = 0; k < ITERATION; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        cudaEventRecord(start);
        spmv_gpu_coo<<<numBlocks, threadsPerBlock>>>(dI, dJ, dval, nz, dX, dY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sum += ms;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    res.exec_time = sum / ITERATION;

    res.gflops = (2.0f * nz) / (res.exec_time * 1e6f);
    //res.mem_bandwidth = (24.0f * nz) / (res.exec_time * 1e6f);

    cudaMemcpy(Ybis, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dI);
    cudaFree(dJ);
    cudaFree(dval);
    cudaFree(dX);
    cudaFree(dY);

    spmv_cpu_coo(I, J, val, nz, X, Y);

    float max_rel = 0.0f;
    float max_abs = 0.0f;
    float tol_abs = 5e-3f;
    float tol_rel = 1e-3f;
    for (int i = 0; i < M; i++)
    {
        float abs_err = fabsf(Ybis[i] - Y[i]);
        float rel_err = fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f);

        float tol = 1e-5f * sqrtf((float)row_nz[i]);

        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > max_abs) max_abs = abs_err;

        if (fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f) > tol_rel && fabsf(Ybis[i] - Y[i]) > tol_abs && fabsf(Ybis[i] - Y[i]) > tol)
        {
            fprintf(stderr, "FAIL row %d : cpu=%.8f gpu=%.8f rel_err=%.2e\n",
               i, Y[i], Ybis[i], rel_err);
            res.is_correct = 0;
        }
    }

    fprintf(stderr, "max relative error = %.2e\n", max_rel);
    fprintf(stderr, "max absolute error = %.2e\n", max_abs);

    free(Y);
    free(Ybis);

    return res;
}

struct results spmv_gpu_csr_prof(const int * O,
                                 const int * J,
                                 const float * val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const float * X)
{
    struct results res = { -1, -1, 1 };

    float *Y, *Ybis;

    Y = (float *)calloc(M, sizeof(float));
    Ybis = (float *)calloc(M, sizeof(float));

    int *dO, *dJ;
    float *dval, *dX, *dY;

    cudaMalloc(&dO, (M + 1) * sizeof(int));
    cudaMalloc(&dJ, nz * sizeof(int));
    cudaMalloc(&dval, nz * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dO, O, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, val, nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int threadsPerBlock = min(1024, maxThreadsPerBlock);
    const int numBlocks = (M + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum = 0.f;
    for (int k = 0; k < WARMUP; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        spmv_gpu_csr<<<numBlocks, threadsPerBlock>>>(dO, dJ, dval, M, dX, dY);
    }
    cudaDeviceSynchronize();

    for (int k = 0; k < ITERATION; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        cudaEventRecord(start);
        spmv_gpu_csr<<<numBlocks, threadsPerBlock>>>(dO, dJ, dval, M, dX, dY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sum += ms;
    }

    res.exec_time = sum / ITERATION;

    res.gflops = (2.0f * nz - M) / (res.exec_time * 1e6f);
    //res.mem_bandwidth = (12.0f * nz + 12.0f * M) / (res.exec_time * 1e6f);

    cudaMemcpy(Ybis, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dO);
    cudaFree(dJ);
    cudaFree(dval);
    cudaFree(dX);
    cudaFree(dY);

    spmv_cpu_csr(O, J, val, M, X, Y);

    float max_rel = 0.0f;
    float max_abs = 0.0f;
    float tol_abs = 5e-3f;
    float tol_rel = 1e-3f;
    for (int i = 0; i < M; i++)
    {
        float abs_err = fabsf(Ybis[i] - Y[i]);
        float rel_err = fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f);

        float tol = 1e-5f * sqrtf((float)row_nz[i]);

        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > max_abs) max_abs = abs_err;

        if (fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f) > tol_rel && fabsf(Ybis[i] - Y[i]) > tol_abs && fabsf(Ybis[i] - Y[i]) > tol)
        {
            fprintf(stderr, "FAIL row %d : cpu=%.8f gpu=%.8f rel_err=%.2e\n",
               i, Y[i], Ybis[i], rel_err);
            res.is_correct = 0;
        }
    }

    fprintf(stderr, "max relative error = %.2e\n", max_rel);
    fprintf(stderr, "max absolute error = %.2e\n", max_abs);

    free(Y);
    free(Ybis);

    return res;
}

struct results spmv_gpu_csr_opt_prof(const int * O,
                                     const int * J,
                                     const float * val,
                                     const int M,
                                     const int N,
                                     const int nz,
                                     const int *row_nz,
                                     const float * X)
{
    struct results res = { -1, -1, 1 };

    float *Y, *Ybis;

    Y = (float *)calloc(M, sizeof(float));
    Ybis = (float *)calloc(M, sizeof(float));

    int *dO, *dJ;
    float *dval, *dX, *dY;

    cudaMalloc(&dO, (M + 1) * sizeof(int));
    cudaMalloc(&dJ, nz * sizeof(int));
    cudaMalloc(&dval, nz * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dO, O, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, val, nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int threadsPerBlock = min(1024, maxThreadsPerBlock);
    const int numBlocks = (M * 32 + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMem = threadsPerBlock * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum = 0.f;
    for (int k = 0; k < WARMUP; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        spmv_gpu_csr_opt<<<numBlocks, threadsPerBlock, sharedMem>>>(dO, dJ, dval, M, dX, dY);
    }
    cudaDeviceSynchronize();

    for (int k = 0; k < ITERATION; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        cudaEventRecord(start);
        spmv_gpu_csr_opt<<<numBlocks, threadsPerBlock, sharedMem>>>(dO, dJ, dval, M, dX, dY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sum += ms;
    }

    res.exec_time = sum / ITERATION;

    res.gflops = (2.0f * nz - M) / (res.exec_time * 1e6f);
    //res.mem_bandwidth = (12.0f * nz + 12.0f * M) / (res.exec_time * 1e6f);

    cudaMemcpy(Ybis, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dO);
    cudaFree(dJ);
    cudaFree(dval);
    cudaFree(dX);
    cudaFree(dY);

    spmv_cpu_csr(O, J, val, M, X, Y);

    float max_rel = 0.0f;
    float max_abs = 0.0f;
    float tol_abs = 5e-3f;
    float tol_rel = 1e-3f;
    for (int i = 0; i < M; i++)
    {
        float abs_err = fabsf(Ybis[i] - Y[i]);
        float rel_err = fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f);

        float tol = 1e-5f * sqrtf((float)row_nz[i]);

        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > max_abs) max_abs = abs_err;

        if (fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f) > tol_rel && fabsf(Ybis[i] - Y[i]) > tol_abs && fabsf(Ybis[i] - Y[i]) > tol)
        {
            fprintf(stderr, "FAIL row %d : cpu=%.8f gpu=%.8f rel_err=%.2e\n",
               i, Y[i], Ybis[i], rel_err);
            res.is_correct = 0;
        }
    }

    fprintf(stderr, "max relative error = %.2e\n", max_rel);
    fprintf(stderr, "max absolute error = %.2e\n", max_abs);

    free(Y);
    free(Ybis);

    return res;
}

struct results spmv_gpu_ell_prof(const int * ell_J,
                                 const float * ell_val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const int max_nz,
                                 const float * X)
{
    struct results res = { -1, -1, 1 };

    float *Y, *Ybis;

    Y = (float *)calloc(M, sizeof(float));
    Ybis = (float *)calloc(M, sizeof(float));

    int *dJ;
    float *dval, *dX, *dY;
    const int len = max_nz * M;

    cudaMalloc(&dJ, len * sizeof(int));
    cudaMalloc(&dval, len * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dJ, ell_J, len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, ell_val, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int threadsPerBlock = min(1024, maxThreadsPerBlock);
    const int numBlocks = (nz + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum = 0.f;
    for (int k = 0; k < WARMUP; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        spmv_gpu_ell<<<numBlocks, threadsPerBlock>>>(dJ, dval, M, max_nz, dX, dY);
    }
    cudaDeviceSynchronize();

    for (int k = 0; k < ITERATION; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        cudaEventRecord(start);
        spmv_gpu_ell<<<numBlocks, threadsPerBlock>>>(dJ, dval, M, max_nz, dX, dY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sum += ms;
    }

    res.exec_time = sum / ITERATION;

    res.gflops = (2.0f * nz) / (res.exec_time * 1e6f);
    //res.mem_bandwidth = (24.0f * nz) / (res.exec_time * 1e6f);

    cudaMemcpy(Ybis, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dJ);
    cudaFree(dval);
    cudaFree(dX);
    cudaFree(dY);

    spmv_cpu_ell(ell_J, ell_val, M, max_nz, X, Y);

    float max_rel = 0.0f;
    float max_abs = 0.0f;
    float tol_abs = 5e-3f;
    float tol_rel = 1e-3f;
    for (int i = 0; i < M; i++)
    {
        float abs_err = fabsf(Ybis[i] - Y[i]);
        float rel_err = fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f);

        float tol = 1e-5f * sqrtf((float)row_nz[i]);

        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > max_abs) max_abs = abs_err;

        if (fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f) > tol_rel && fabsf(Ybis[i] - Y[i]) > tol_abs && fabsf(Ybis[i] - Y[i]) > tol)
        {
            fprintf(stderr, "FAIL row %d : cpu=%.8f gpu=%.8f rel_err=%.2e\n",
               i, Y[i], Ybis[i], rel_err);
            res.is_correct = 0;
        }
    }

    fprintf(stderr, "max relative error = %.2e\n", max_rel);
    fprintf(stderr, "max absolute error = %.2e\n", max_abs);

    free(Y);
    free(Ybis);

    return res;
}

struct results spmv_gpu_sell_c_prof(const int * sell_O,
                                 const int * sell_J,
                                 const float * sell_val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const int C,
                                 const int num_slices,
                                 const float * X)
{
    struct results res = { -1, -1, 1 };

    float *Y, *Ybis;

    Y = (float *)calloc(M, sizeof(float));
    Ybis = (float *)calloc(M, sizeof(float));

    int *dO, *dJ;
    float *dval, *dX, *dY;

    const int len = sell_O[num_slices];
    cudaMalloc(&dO, (num_slices + 1) * sizeof(int));
    cudaMalloc(&dJ, len * sizeof(int));
    cudaMalloc(&dval, len * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dO, sell_O, (num_slices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, sell_J, len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, sell_val, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, M * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int threadsPerBlock = min(1024, maxThreadsPerBlock);
    const int numBlocks = (len + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum = 0.f;
    for (int k = 0; k < WARMUP; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        spmv_gpu_sell_c<<<numBlocks, threadsPerBlock>>>(dO, dJ, dval, M, C, num_slices, dX, dY);
    }
    cudaDeviceSynchronize();

    for (int k = 0; k < ITERATION; k++)
    {
        cudaMemset(dY, 0, M * sizeof(float));

        cudaEventRecord(start);
        spmv_gpu_sell_c<<<numBlocks, threadsPerBlock>>>(dO, dJ, dval, M, C, num_slices, dX, dY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        sum += ms;
    }

    res.exec_time = sum / ITERATION;

    res.gflops = (2.0f * nz) / (res.exec_time * 1e6f);
    //res.mem_bandwidth = (24.0f * nz) / (res.exec_time * 1e6f);

    cudaMemcpy(Ybis, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dO);
    cudaFree(dJ);
    cudaFree(dval);
    cudaFree(dX);
    cudaFree(dY);

    spmv_cpu_sell_c(sell_O, sell_J, sell_val, M, C, num_slices, X, Y);

    float max_rel = 0.0f;
    float max_abs = 0.0f;
    float tol_abs = 5e-3f;
    float tol_rel = 1e-3f;
    for (int i = 0; i < M; i++)
    {
        float abs_err = fabsf(Ybis[i] - Y[i]);
        float rel_err = fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f);

        float tol = 1e-5f * sqrtf((float)row_nz[i]);

        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > max_abs) max_abs = abs_err;

        if (fabsf(Ybis[i] - Y[i]) / (fabsf(Y[i]) + 1e-10f) > tol_rel && fabsf(Ybis[i] - Y[i]) > tol_abs && fabsf(Ybis[i] - Y[i]) > tol)
        {
            fprintf(stderr, "FAIL row %d : cpu=%.8f gpu=%.8f rel_err=%.2e\n",
               i, Y[i], Ybis[i], rel_err);
            res.is_correct = 0;
        }
    }

    fprintf(stderr, "max relative error = %.2e\n", max_rel);
    fprintf(stderr, "max absolute error = %.2e\n", max_abs);

    free(Y);
    free(Ybis);

    return res;
}
