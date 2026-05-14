#include "../include/mmfmt.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct
{
    int row;
    int col;
    float val;
}
coo_entry_t;

static int coo_cmp(const void *a, const void *b)
{
    const coo_entry_t *ea = (const coo_entry_t *)a;
    const coo_entry_t *eb = (const coo_entry_t *)b;

    return (ea->row != eb->row) ? ea->row - eb->row : ea->col - eb->col;
}

void mm_sort_coo(int *I, int *J, float *val, const int nz)
{
    int i;

    /* Construire un tableau de triplets */
    coo_entry_t *entries = (coo_entry_t *) malloc(nz * sizeof(coo_entry_t));
    if (!entries)
    {
        fprintf(stderr, "sort_coo: malloc failed\n");
        exit(1);
    }

    for (i = 0; i < nz; ++i)
    {
        entries[i].row = I[i];
        entries[i].col = J[i];
        entries[i].val = val[i];
    }

    qsort(entries, nz, sizeof(coo_entry_t), coo_cmp);

    /* Réécrire les tableaux d'origine */
    for (i = 0; i < nz; ++i)
    {
        I[i]   = entries[i].row;
        J[i]   = entries[i].col;
        val[i] = entries[i].val;
    }

    free(entries);
}

void mm_coo_to_csr_row_ptr(const int *I, const int nz, const int M, int *O)
{
    int i;

    for (i = 0; i <= M; ++i) O[i] = 0;

    for (i = 0; i < nz; ++i) O[I[i] + 1]++;

    for (i = 1; i <= M; ++i) O[i] += O[i - 1];
}

void mm_csr_to_ellpack(const int *O, const int *J, const float *val, const int M, int *max_nz, int **ell_J, float **ell_val)
{
    size_t i, j, k;

    *max_nz = 0;
    for (i = 0; i < (size_t)M; i++)
    {
        int row_len = (size_t)O[i + 1] - (size_t)O[i];

        if (row_len > *max_nz) *max_nz = row_len;
    }

    size_t len = (size_t)M * (size_t)(*max_nz);
    *ell_J = (int *) malloc(len * sizeof(int));
    for (i = 0; i < len; i++) (*ell_J)[i] = -1;
    *ell_val = (float *) calloc(len, sizeof(float));

    for (i = 0; i < (size_t)M; i++)
    {
        k = 0;

        for (j = (size_t)O[i]; j < (size_t)O[i + 1]; ++j, ++k)
        {
            (*ell_J)  [k * M + i] = J[j];
            (*ell_val)[k * M + i] = val[j];
        }
    }
}

void mm_csr_to_sell_c(const int *O, const int *J, const float *val, const int M, const int C, int *num_slices, int **sell_O, int **sell_J, float **sell_val)
{
    int s, i, j, k;

    *num_slices = (M + C - 1) / C;

    *sell_O = (int *) malloc((*num_slices + 1) * sizeof(int));
    (*sell_O)[0] = 0;

    for (s = 0; s < *num_slices; s++)
    {
        int row_start = s * C;
        int row_end   = row_start + C < M ? row_start + C : M; /* min(s*C+C, M) */

        int max_nz_s = 0;
        for (i = row_start; i < row_end; i++)
        {
            int row_len = O[i + 1] - O[i];
            if (row_len > max_nz_s) max_nz_s = row_len;
        }

        (*sell_O)[s + 1] = (*sell_O)[s] + C * max_nz_s;
    }

    int total = (*sell_O)[*num_slices];

    *sell_J = (int *) malloc(total * sizeof(int));
    *sell_val = (float *) calloc(total, sizeof(float));

    for (i = 0; i < total; i++) (*sell_J)[i] = -1;

    for (s = 0; s < *num_slices; s++)
    {
        int row_start  = s * C;
        int row_end    = row_start + C < M ? row_start + C : M;
        int slice_base = (*sell_O)[s];
        //int max_nz_s   = ((*sell_O)[s + 1] - slice_base) / C;

        for (i = row_start; i < row_end; i++)
        {
            k = 0;
            for (j = O[i]; j < O[i + 1]; ++j, ++k)
            {
                int local_row = i - row_start;

                (*sell_J)  [slice_base + k * C + local_row] = J[j];
                (*sell_val)[slice_base + k * C + local_row] = val[j];
            }
        }
    }
}
