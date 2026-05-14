#ifndef SPMV_CPU_H
#define SPMV_CPU_H

void spmv_cpu_coo(const int * I,
                  const int * J,
                  const float * val,
                  const int nz,
                  const float * X,
                  float * Y);

void spmv_cpu_csr(const int * O,
                  const int * J,
                  const float * val,
                  const int M,
                  const float * X,
                  float * Y);

void spmv_cpu_ell(const int * ell_J,
                  const float * ell_val,
                  const int M,
                  const int max_nz,
                  const float * X,
                  float * Y);

void spmv_cpu_sell_c(const int * sell_O,
                     const int * sell_J,
                     const float * sell_val,
                     const int M,
                     const int C,
                     const int num_slices,
                     const float * X,
                     float * Y);

#endif