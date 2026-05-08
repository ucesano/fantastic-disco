#ifndef SPMV_H
#define SPMV_H

float * spmv_cpu_coo(int M,
                     int nnz,
                     int *I,
                     int *J,
                     float *values,
                     float *X);

float * spmv_cpu_csr(int M,
                   int *O,
                   int *J,
                   float *values,
                   float *X);

float * spmv_cpu_ell(int M,
                     int K, // max de nnz par ligne
                     int *J,
                     float *values,
                     float *X);

float * spmv_cpu_sell(int M,
                     int *SO,
                     int *J,
                     float *values,
                     float *X,
                     int nslices,
                     int sz_slice);

#endif
