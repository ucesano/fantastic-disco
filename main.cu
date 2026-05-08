#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "include/matrix.h"
#include "include/spmv.h"

#ifdef __cplusplus
}
#endif

#define ITERATIONS 25

void print_device_properties(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("  Memory Clock rate:           %.0f Mhz\n", devProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:            %d bit\n",devProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:       %7.3f GB/s\n",2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);

    printf("  Multiprocessors:             %3d\n",devProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor:  %d\n",devProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",devProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           devProp.maxThreadsDim[0], devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           devProp.maxGridSize[0], devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("  Total amount of shared memory per block:       %zu bytes\n", devProp.sharedMemPerBlock);
}

void display_card_informations(void)
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        print_device_properties(devProp);
    }
}

__global__ void kernel_spmv_coo(int nnz,
                                const int * __restrict__ I,
                                const int * __restrict__ J,
                                const float * __restrict__ A,
                                const float * __restrict__ X,
                                float * __restrict__ Y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int s = blockDim.x * gridDim.x;

    /* We use striding to cover the entier matrix even
     * if there are more nnz than threads. */

    for (; i < nnz; i += s) atomicAdd(&Y[I[i]], A[i] * X[J[i]]);
}

/*
__inline__ __device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kernel_spmv_csr_warp(int M,
                                     const int * __restrict__ O,
                                     const int * __restrict__ J,
                                     const float * __restrict__ A,
                                     const float * __restrict__ X,
                                     float * __restrict__ Y)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = global_thread >> 5;

    int row = warp_id;
    if (row >= M) return;

    int p = O[row];
    int q = O[row + 1];

    float sum = 0.0f;
    for (int i = p + lane; i < q; i += 32)
        sum += A[i] * X[J[i]];

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        Y[row] = sum;
}
*/

__global__ void kernel_spmv_csr_warp(int M,
                                 const int * __restrict__ O,
                                 const int * __restrict__ J,
                                 const float * __restrict__ A,
                                 const float * __restrict__ X,
                                 float * __restrict__ Y)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;

    int warps_per_block = blockDim.x >> 5;

    // Global warp idx in grid
    int warp_global = blockIdx.x * warps_per_block + warp_in_block;

    int row = warp_global;
    if (row >= M) return;

    int p = O[row];
    int q = O[row + 1];

    // Per thread partial sum computation on row.
    float sum = 0.0f;
    for (int i = p + lane; i < q; i += 32)
        sum += A[i] * X[J[i]];

    // Write it in shared mem.
    sdata[tid] = sum;
    __syncthreads();

    // Intra-warp reduction.
    int base = warp_in_block * 32;

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (lane < offset)
        {
            sdata[base + lane] += sdata[base + lane + offset];
        }
        __syncthreads();
    }

    // Only lane 0 of warp write down the result.
    if (lane == 0)
    {
        Y[row] = sdata[base];
    }
}

__global__ void kernel_spmv_ell(int M, int K, const int *J, const float *A, const float *X, float *Y)
{
    __shared__ float s_X[32];

    int row = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int tx = threadIdx.x;

    if (tx < 32) s_X[tx] = X[tx];

    __syncthreads();

    if (row < M)
    {
        float sum = 0.f;

        for (int i = row * K + lane; i < (row + 1) * K; i += 32)
        {
            int col = J[i];
            if (col >= 0 && col < 32) sum += A[i] * s_X[col];
            else if (col >= 0) sum += A[i] * X[col];
        }

        sum = warp_reduce(sum);

        if (lane == 0) Y[row] = sum;
    }
}

/*
__global__ void kernel_spmv_ell(int M, int K, const int *J, const float *A, const float *X, float *Y)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M)
    {
        float sum = 0.0f;
        for (int i = row; i < K * M; i += M)
        {
            int col = J[i];
            if (col >= 0) sum += A[i] * X[col];
        }
        Y[row] = sum;
    }
}
*/

__global__ void kernel_spmv_sell(int M, // rows
                                 int *SO, // slice_offsets
                                 int *J, // column_indices
                                 float *A, // values
                                 float *X, // ones
                                 float *Y, // res
                                 int nslices,
                                 int sz_slice)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 0;
    int j = 0;

    int caca = nslices * sz_slice;

    if (tid <= caca)
    {
        int s = tid / sz_slice;
        int r = tid % sz_slice;

        int max_nnz = (SO[s + 1] - SO[s]) / sz_slice;

        float sum = 0.f;

        for (int k = 0; k < max_nnz; ++k)
        {
            i = SO[s] + k * sz_slice + r;

            j = J[i];
            if (j != -1) sum += A[i] * X[j];
        }

        Y[tid] = sum;
    }
}

void gpu_coo(char **argv)
{
    int M, N, nnz;
    int *hI, *hJ, *dI, *dJ;
    float *hA, *hX, *hY, *dA, *dX, *dY;

    matrix_load_coo(argv[1], &M, &N, &nnz, &hI, &hJ, &hA);

    hX = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hX[i] = 1.;
    hY = (float *)malloc(M * sizeof(float));

    cudaMalloc(&dI, nnz * sizeof(int));
    cudaMalloc(&dJ, nnz * sizeof(int));
    cudaMalloc(&dA, nnz * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dI, hI, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, hJ, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(dY, 0, M * sizeof(float));

    float time = 0.f;
    cudaEvent_t start_sell, stop_sell;

    cudaEventCreate(&start_sell);
    cudaEventCreate(&stop_sell);

    kernel_spmv_coo<<<(nnz + 255) / 256, 256>>>(nnz, dI, dJ, dA, dX, dY);
    cudaDeviceSynchronize();

    int num_iterations = ITERATIONS;
    float total_time = 0.f;

    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start_sell);
        kernel_spmv_coo<<<(nnz + 255) / 256, 256>>>(nnz, dI, dJ, dA, dX, dY);
        cudaEventRecord(stop_sell);
        cudaEventSynchronize(stop_sell);

        float iteration_time = 0.f;
        cudaEventElapsedTime(&iteration_time, start_sell, stop_sell);
        total_time += iteration_time;
    }

    time = total_time / num_iterations;

    cudaEventDestroy(start_sell);
    cudaEventDestroy(stop_sell);

    fprintf(stdout, "COO: %f ms\n", time);

    cudaMemcpy(hY, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dI);
    cudaFree(dJ);
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    free(hI);
    free(hJ);
    free(hA);
    free(hX);
    free(hY);
}

void gpu_csr(char **argv)
{
    int M, N, nnz;
    int *hO, *hJ, *dO, *dJ;
    float *hA, *hX, *hY, *dA, *dX, *dY;

    matrix_load_csr(argv[1], &M, &N, &nnz, &hO, &hJ, &hA);

    hX = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hX[i] = 1.;
    hY = (float *)malloc(M * sizeof(float));

    cudaMalloc(&dO, (M + 1) * sizeof(int));
    cudaMalloc(&dJ, nnz * sizeof(int));
    cudaMalloc(&dA, nnz * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dO, hO, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, hJ, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(dY, 0, M * sizeof(float));

    int block_size = 128;
    int warps_per_block = block_size / 32;
    int grid_size = (M + warps_per_block - 1) / warps_per_block;
    size_t shared_mem_size = block_size * sizeof(float);


    float time = 0.f;
    cudaEvent_t start_sell, stop_sell;

    cudaEventCreate(&start_sell);
    cudaEventCreate(&stop_sell);
    kernel_spmv_csr_warp<<<grid_size, block_size, shared_mem_size>>>(M, dO, dJ, dA, dX, dY);

    cudaDeviceSynchronize();

    int num_iterations = ITERATIONS;
    float total_time = 0.f;

    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start_sell);
        kernel_spmv_csr_warp<<<grid_size, block_size, shared_mem_size>>>(M, dO, dJ, dA, dX, dY);
        cudaEventRecord(stop_sell);
        cudaEventSynchronize(stop_sell);

        float iteration_time = 0.f;
        cudaEventElapsedTime(&iteration_time, start_sell, stop_sell);
        total_time += iteration_time;
    }

    time = total_time / num_iterations;

    cudaEventDestroy(start_sell);
    cudaEventDestroy(stop_sell);

    fprintf(stdout, "CSR: %f ms\n", time);

    cudaMemcpy(hY, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dO);
    cudaFree(dJ);
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    free(hO);
    free(hJ);
    free(hA);
    free(hX);
    free(hY);
}

void gpu_ell(char **argv)
{
    int M, N, nnz;
    int K, *hJ, *hK, *dJ;
    float *hA, *hX, *hY, *dA, *dX, *dY;

    matrix_load_ell(argv[1], &M, &N, &nnz, &K, &hK, &hJ, &hA);

    hX = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hX[i] = 1.;
    hY = (float *)malloc(M * sizeof(float));

    cudaMalloc(&dJ, M * K * sizeof(int));
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dJ, hJ, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice);

    float time = 0.;
    cudaEvent_t start_sell, stop_sell;

    cudaEventCreate(&start_sell);
    cudaEventCreate(&stop_sell);
    kernel_spmv_ell<<<(nnz + 255) / 256, 256>>>(M,
                                                    K,
                                                    dJ,
                                                    dA,
                                                    dX,
                                                    dY);
    cudaDeviceSynchronize();

    int num_iterations = ITERATIONS;
    float total_time = 0.f;

    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start_sell);
        cudaMemset(dY, 0, M * sizeof(float));

        kernel_spmv_ell<<<(nnz + 255) / 256, 256>>>(M,
                                                    K,
                                                    dJ,
                                                    dA,
                                                    dX,
                                                    dY);
        cudaEventRecord(stop_sell);
        cudaEventSynchronize(stop_sell);

        float iteration_time = 0.f;
        cudaEventElapsedTime(&iteration_time, start_sell, stop_sell);
        total_time += iteration_time;
    }

    time = total_time / num_iterations;

    cudaEventDestroy(start_sell);
    cudaEventDestroy(stop_sell);

    fprintf(stdout, "ELLPACK: %f ms\n", time);

    cudaMemcpy(hY, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dJ);
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    free(hJ);
    free(hA);
    free(hX);
    free(hY);
}

void gpu_sell(char **argv)
{
    const int sz_slice = 25;
    int M, N, nnz, sz_values, nslices;
    int *hK, *hSO, *hJ, *dSO, *dJ;
    float *hA, *hX, *hY, *dA, *dX, *dY;

    matrix_load_sell(argv[1], &M, &N, &nslices, &nnz, &sz_values, &hK, &hSO, &hJ, &hA, sz_slice);

    hX = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hX[i] = 1.;
    hY = (float *)malloc(M * sizeof(float));

    cudaMalloc(&dSO, (nslices + 1) * sizeof(int));
    cudaMalloc(&dJ, sz_values * sizeof(int));
    cudaMalloc(&dA, sz_values * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));

    cudaMemcpy(dSO, hSO, (nslices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, hJ, sz_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sz_values * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(dY, 0, M * sizeof(float));

    for (int i = 0; i < 4; i++) fprintf(stderr, "%d ", hSO[i]);
    fputc(10, stderr);
    for (int i = 0; i < 14; i++) fprintf(stderr, "%d ", hJ[i]);
    fputc(10, stderr);
    for (int i = 0; i < 14; i++) fprintf(stderr, "%.1f ", hA[i]);
    fputc(10, stderr);

    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;



    cudaMemcpy(hY, dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < M; i++) fprintf(stdout, "%f ", hY[i]);
    //fputc(10, stdout);

    float time = 0.;
    cudaEvent_t start_sell, stop_sell;

    cudaEventCreate(&start_sell);
    cudaEventCreate(&stop_sell);

    kernel_spmv_sell<<<grid_size, block_size>>>(M, dSO, dJ, dA, dX, dY, nslices, sz_slice);
    cudaDeviceSynchronize();

    // Mesure sur plusieurs itérations
    int num_iterations = ITERATIONS;
    float total_time = 0.f;

    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start_sell);
        kernel_spmv_sell<<<grid_size, block_size>>>(M, dSO, dJ, dA, dX, dY, nslices, sz_slice);
        cudaEventRecord(stop_sell);
        cudaEventSynchronize(stop_sell);

        float iteration_time = 0.f;
        cudaEventElapsedTime(&iteration_time, start_sell, stop_sell);
        total_time += iteration_time;
    }

    time = total_time / num_iterations;

    cudaEventDestroy(start_sell);
    cudaEventDestroy(stop_sell);

    fprintf(stdout, "SLICED ELLPACK (%d): %f ms\n", sz_slice, time);

    cudaFree(dSO);
    cudaFree(dJ);
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    free(hSO);
    free(hJ);
    free(hA);
    free(hX);
    free(hY);
}


int main(int argc, char **argv)
{
    int ret = 0;

    if (argc != 2)
    {
        fputs("Usage:\n  spmv [MatrixMarket file]\n", stderr);
        exit(1);
    }

    display_card_informations();

    gpu_coo(argv);

    cudaDeviceSynchronize();

    gpu_csr(argv);

    cudaDeviceSynchronize();

    gpu_ell(argv);

    cudaDeviceSynchronize();

    gpu_sell(argv);

    cudaDeviceSynchronize();

    return ret;
}
