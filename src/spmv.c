#include "../include/spmv.h"

#include <stdlib.h>

float * spmv_cpu_coo(int M,
                     int nnz,
                     int *I,
                     int *J,
                     float *values,
                     float *X)
{
    float * res = (float *)calloc(M, sizeof(float));

    for (int i = 0; i < nnz; i++) res[I[i]] += values[i] * X[J[i]];

    return res;
}

float * spmv_cpu_csr(int M,
                     int *O,
                     int *J,
                     float *values,
                     float *X)
{
    float * res = (float *)calloc(M, sizeof(float));

    for (int i = 0; i < M; i++)
    {
        for (int j = O[i]; j < O[i + 1]; j++) res[i] += values[j] * X[J[j]];
    }

    return res;
}


float * spmv_cpu_ell(int M,
                     int K,
                     int *J,
                     float *values,
                     float *X)
{
    float *res = (float *)calloc(M, sizeof(float));

    for (int i = 0; i < M; i++)
    {
        for (int k = 0; k < K; k++)
        {
            int j = J[i + M * k];
            if (j != -1)
                res[i] += values[i + M * k] * X[j];
        }
    }

    return res;
}

/*
void matrix_load_sell(char *fname,
                      int *M, // nombre de lignes
                      int *N, // nombre de colonne
                      int *nslices, // nombre de slices
                      int *nnz, // nombre de non-zéros
                      int *sz_values, // nombre de nnz + padding
                      int **SO, // slice offset
                      int **J, // Indices de colonnes en zero based index avec -1 comme padding
                      float **values, // valeurs avec 0.0 comme padding
                      const int sz_slice) // taille d'un slice

*/

float * spmv_cpu_sell(int M,
                     int *SO,
                     int *J,
                     float *values,
                     float *X,
                     int nslices,
                     int sz_slice)
{
    float *res = (float *)calloc(M, sizeof(float));

    for (int s = 0; s < nslices; s++)
    {
        int rb = s * sz_slice;
        int re = rb + sz_slice;
        if (re > M) re = M;

        int sb = SO[s];
        int se = SO[s + 1];
        int nnz_max = (se - sb) / sz_slice;

        for (int k = 0; k < nnz_max; k++)
        {
            for (int r = rb; r < re; r++)
            {
                int idx = sb + k * sz_slice + (r - rb);
                int j = J[idx];
                if (j != -1) res[r] += values[idx] * X[j];
            }
        }
    }

    return res;
}
