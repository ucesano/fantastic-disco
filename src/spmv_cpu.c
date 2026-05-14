#include "../include/spmv_cpu.h"

#include <stdlib.h>

void spmv_cpu_coo(const int * I,
                  const int * J,
                  const float * val,
                  const int nz,
                  const float * X,
                  float * Y)
{
    int i;

    for (i = 0; i < nz; ++i) Y[I[i]] += val[i] * X[J[i]];
}


void spmv_cpu_csr(const int * O,
                  const int * J,
                  const float * val,
                  const int M,
                  const float * X,
                  float * Y)
{
    int i, j;

    for (i = 0; i < M; ++i)
    {
        int start = O[i];
        int end = O[i + 1];

        for (j = start; j < end; ++j) Y[i] += val[j] * X[J[j]];
    }
}

void spmv_cpu_ell(const int * ell_J,
                  const float * ell_val,
                  const int M,
                  const int max_nz,
                  const float * X,
                  float * Y)
{
    int i, k;

    for (k = 0; k < max_nz; ++k)
    {
        for (i = 0; i < M; ++i)
        {
            int col = ell_J[(size_t)k * M + i];

            if (col != -1) Y[i] += ell_val[(size_t)k * M + i] * X[col];
        }
    }
}

void spmv_cpu_sell_c(const int * sell_O,
                     const int * sell_J,
                     const float * sell_val,
                     const int M,
                     const int C,
                     const int num_slices,
                     const float * X,
                     float * Y)
{
    int s, i, k;

    for (s = 0; s < num_slices; ++s)
    {
        int slice_base = sell_O[s];
        int max_nz_s   = (sell_O[s + 1] - slice_base) / C;
        int row_start  = s * C;
        int row_end    = row_start + C < M ? row_start + C : M;

        for (k = 0; k < max_nz_s; ++k)
        {
            for (i = row_start; i < row_end; ++i)
            {
                int local_row = i - row_start;
                int col = sell_J[slice_base + k * C + local_row];

                if (col != -1) Y[i] += sell_val[slice_base + k * C + local_row] * X[col];
            }
        }
    }
}
