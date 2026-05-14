#include "../include/spmv_gpu.cuh"

#include <stdlib.h>

__global__
void spmv_gpu_coo(const int *__restrict__ I,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int nz,
                  const float *__restrict__ X,
                  float *__restrict__ Y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < nz; i += stride)
        atomicAdd(&Y[I[i]], val[i] * X[J[i]]);
}

__global__
void spmv_gpu_csr(const int *__restrict__ O,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int M,
                  const float *__restrict__ X,
                  float *__restrict__ Y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.f;
    int start = O[row];
    int end   = O[row + 1];

    for (int j = start; j < end; ++j)
        sum += val[j] * X[J[j]];

    Y[row] = sum;
}

__global__
void spmv_gpu_csr_opt(const int   *__restrict__ O,
                         const int   *__restrict__ J,
                         const float *__restrict__ val,
                         const int    M,
                         const float *__restrict__ X,
                         float       *__restrict__ Y)
{
    extern __shared__ float vals[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = tid / 32;
    int lane = tid & (32 - 1);

    int row = wid;
    if (row >= M) return;

    int row_start = O[row];
    int row_end   = O[row + 1];

    vals[threadIdx.x] = 0.f;
    for (int j = row_start + lane; j < row_end; j += 32)
        vals[threadIdx.x] += val[j] * X[J[j]];

    __syncwarp();

    // Reduction in shared memory
    if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16]; __syncwarp();
    if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8]; __syncwarp();
    if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4]; __syncwarp();
    if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2]; __syncwarp();
    if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1];

    if (lane == 0)
        Y[row] = vals[threadIdx.x];
}


__global__
void spmv_gpu_ell(const int    *__restrict__ ell_J,
                  const float *__restrict__ ell_val,
                  const int    M,
                  const int    max_nz,
                  const float *__restrict__ X,
                  float       *__restrict__ Y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.f;

    for (int k = 0; k < max_nz; ++k)
    {
        size_t idx = (size_t)k * M + row;
        int col = ell_J[idx];

        if (col != -1)
            sum += ell_val[idx] * X[col];
    }

    Y[row] = sum;
}

__inline__ __device__
float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__
void spmv_gpu_sell_c(const int    *__restrict__ sell_ptr,
                     const int    *__restrict__ sell_J,
                     const float *__restrict__ sell_val,
                     const int    M,
                     const int    C,
                     const int    num_slices,
                     const float *__restrict__ X,
                     float       *__restrict__ Y)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int lane     = threadIdx.x & 31;
    int warp_id  = tid >> 5;
    int warp_cnt = (gridDim.x * blockDim.x) >> 5;

    for (int s = warp_id; s < num_slices; s += warp_cnt)
    {
        int row_start = s * C;
        int row_end   = row_start + C < M ? row_start + C : M;
        int slice_base = sell_ptr[s];
        int max_nz_s   = (sell_ptr[s + 1] - slice_base) / C;

        for (int row = row_start + lane; row < row_end; row += 32)
        {
            int local_row = row - row_start;
            float sum = 0.0;

            for (int k = 0; k < max_nz_s; ++k)
            {
                int idx = slice_base + k * C + local_row;
                int col = sell_J[idx];
                if (col != -1)
                    sum += sell_val[idx] * X[col];
            }

            Y[row] = sum;
        }
    }
}
