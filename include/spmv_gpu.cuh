#ifndef SPMV_GPU_CUH
#define SPMV_GPU_CUH

__global__
void spmv_gpu_coo(const int *__restrict__ I,
                  const int *__restrict__ J,
                  const float *__restrict__ val,
                  const int nz,
                  const float *__restrict__ X,
                  float *__restrict__ Y);

__global__ void spmv_gpu_csr(const int *__restrict__ O,
                             const int *__restrict__ J,
                             const float *__restrict__ val,
                             const int M,
                             const float *__restrict__ X,
                             float *__restrict__ Y);

__global__
void spmv_gpu_csr_opt(const int *__restrict__ O,
                         const int *__restrict__ J,
                         const float *__restrict__ val,
                         const int M,
                         const float *__restrict__ X,
                         float *__restrict__ Y);

__global__
void spmv_gpu_ell(const int    *__restrict__ ell_J,
                  const float *__restrict__ ell_val,
                  const int    M,
                  const int    max_nz,
                  const float *__restrict__ X,
                  float       *__restrict__ Y);

__global__
void spmv_gpu_sell_c(const int    *__restrict__ sell_ptr,
                     const int    *__restrict__ sell_J,
                     const float *__restrict__ sell_val,
                     const int    M,
                     const int    C,
                     const int    num_slices,
                     const float *__restrict__ X,
                     float       *__restrict__ Y);
#endif