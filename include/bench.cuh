#ifndef BENCH_H
#define BENCH_H

#define WARMUP 10
#define ITERATION 30

struct results
{
    float exec_time;
    //float mem_bandwidth;
    float gflops;
    char   is_correct;
};

void print_results(const struct results res, const char *fmt);

struct results spmv_gpu_coo_prof(const int * I,
                                 const int * J,
                                 const float * val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const float * X);

struct results spmv_gpu_csr_prof(const int * O,
                                 const int * J,
                                 const float * val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const float * X);

struct results spmv_gpu_csr_opt_prof(const int * O,
                                     const int * J,
                                     const float * val,
                                     const int M,
                                     const int N,
                                     const int nz,
                                     const int *row_nz,
                                     const float * X);

struct results spmv_gpu_ell_prof(const int * ell_J,
                                 const float * ell_val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const float * X);

struct results spmv_gpu_ell_prof(const int * ell_J,
                                 const float * ell_val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const int max_nz,
                                 const float * X);

struct results spmv_gpu_sell_c_prof(const int * sell_O,
                                 const int * sell_J,
                                 const float * sell_val,
                                 const int M,
                                 const int N,
                                 const int nz,
                                 const int *row_nz,
                                 const int C,
                                 const int num_slices,
                                 const float * X);
#endif