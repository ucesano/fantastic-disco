#include "../include/matrix.h"

#include "../include/mmio.h"
#include <string.h>
#include <math.h>
#include <stdarg.h>

/* Matrix entry with:
 * - i: line
 * - j: column
 * v: value */
struct entry
{
    int i;
    int j;
    float v;
};

static int compare_rowwise(const void *a, const void *b)
{
    /* For convenience. */
    struct entry *ae = (struct entry *)a;
    struct entry *be = (struct entry *)b;

    if (ae->i < be->i) return -1;
    if (ae->i > be->i) return 1;

    if (ae->j < be->j) return -1;
    if (ae->j > be->j) return 1;

    return 0;
}

static int compare_columnwise(const void *a, const void *b)
{
    /* For convenience. */
    struct entry *ae = (struct entry *)a;
    struct entry *be = (struct entry *)b;

    if (ae->j < be->j) return -1;
    if (ae->j > be->j) return 1;

    if (ae->i < be->i) return -1;
    if (ae->i > be->i) return 1;

    return 0;
}

/* Sorting COO sparse matrix representation line-wise with:
 * - nnz: number of elements
 * - I: row indices
 * - J: column indices
 * - values: values */
static void sort_rowwise(const int nnz,
                 int *I,
                 int *J,
                 float *values)
{
    struct entry * temp = (struct entry *)malloc(nnz * sizeof(struct entry));
    int i;

    for (i = 0; i < nnz; i++)
    {
        temp[i].i = I[i];
        temp[i].j = J[i];
        temp[i].v = values[i];
    }

    qsort(temp, nnz, sizeof(temp[0]), compare_rowwise);

    for (i = 0; i < nnz; i++)
    {
        I[i] = temp[i].i;
        J[i] = temp[i].j;
        values[i] = temp[i].v;
    }

    free(temp);
}

/* Sorting COO sparse matrix representation column-wise with:
 * - nnz: number of elements
 * - I: row indices
 * - J: column indices
 * - values: non-zero values */
static void sort_columnwise(const int nnz,
                    int *I,
                    int *J,
                    float *values)
{
    struct entry * temp = (struct entry *)malloc(nnz * sizeof(struct entry));
    int i;

    for (i = 0; i < nnz; i++)
    {
        temp[i].i = I[i];
        temp[i].j = J[i];
        temp[i].v = values[i];
    }

    qsort(temp, nnz, sizeof(temp[0]), compare_columnwise);

    for (i = 0; i < nnz; i++)
    {
        I[i] = temp[i].i;
        J[i] = temp[i].j;
        values[i] = temp[i].v;
    }

    free(temp);
}

/* Loading sparse matrix from MatrixMarket file in COO format with:
 * - fname: MM file UNIX path
 * - M: number of rows
 * - N: number of columns
 * - nnz: number of non-zeros
 * - I: row indices
 * - J: column indices
 * - values: non-zero values*/
void matrix_load_coo(char *fname,
                      int *M,
                      int *N,
                      int *nnz,
                      int **I,
                      int **J,
                      float **values)
{
    MM_typecode matcode;

    if (mm_read_mtx_crd(fname, M, N, nnz, I, J, values, &matcode)) return;

    /* Going from 1-base index to 0-based index. */
    for (int i = 0; i < *nnz; i++)
    {
        (*I)[i]--;
        (*J)[i]--;
    }

    /* Sorting row-wise by default. */
    sort_rowwise(*nnz, *I, *J, *values);
}

/* Loading sparse matrix from MatrixMarket file in CSR format with:
 * - fname: MM file UNIX path
 * - M: number of rows
 * - N: number of columns
 * - nnz: number of non-zeros
 * - O: row offsets
 * - J: column indices
 * - values: non-zero values*/
void matrix_load_csr(char *fname,
                      int *M,
                      int *N,
                      int *nnz,
                      int **O,
                      int **J,
                      float **values)
{
    int i;
    int *I;

    matrix_load_coo(fname, M, N, nnz, &I, J, values);

    *O = (int*)calloc((*M + 1), sizeof(i));

    /* Computing row offsets. */
    for (i = 0; i < *nnz; i++) (*O)[I[i] + 1]++;
    for (i = 0; i < *M; i++) (*O)[i + 1] += (*O)[i];

    free(I);
}

/* Loading sparse matrix from MatrixMarket file in Ellpack format with:
 * - fname: MM file UNIX path
 * - M: number of rows
 * - N: number of columns
 * - nnz: number of non-zeros
 * - max_nnz_per_row: max number of non-zeros per row
 * - nnz_per_row: number of non-zeros per row
 * - J: column indices
 * - values: non-zero values*/
void matrix_load_ell(char *fname,
                      int *M,
                      int *N,
                      int *nnz,
                      int *max_nnz_per_row,
                      int **nnz_per_row,
                      int **J,
                      float **values)
{
    int i, j;
    int *I;
    int *coo_J;
    float *coo_val;
    int *O;

    matrix_load_coo(fname, M, N, nnz, &I, &coo_J, &coo_val);

    *nnz_per_row = (int*)calloc((*M + 1), sizeof(int));
    O = (int*)calloc((*M + 1), sizeof(i));

    for (i = 0; i < *nnz; i++) (*nnz_per_row)[I[i]]++;
    memcpy(O + 1, *nnz_per_row, (*M) * sizeof(int));
    for (i = 0; i < *M; i++) O[i + 1] += O[i];

    *max_nnz_per_row = (*nnz_per_row)[0];
    for (i = 1; i < *M; i++)
    {
        if ((*nnz_per_row)[i] > *max_nnz_per_row)
            *max_nnz_per_row = (*nnz_per_row)[i];
    }

    *J = (int *)malloc(*M * *max_nnz_per_row * sizeof(i));
    for (i = 0; i < *M * *max_nnz_per_row; i++) (*J)[i] = -1;
    *values = (float *)calloc(*M * *max_nnz_per_row, sizeof(float));



    for (i = 0; i < *M; i++)
    {
        for (j = O[i]; j < O[i + 1]; j++)
        {
            (*J)[*M * (j - O[i]) + i] = coo_J[j];
            (*values)[*M * (j - O[i]) + i] = coo_val[j];
        }
    }

    free(I);
    free(coo_J);
    free(coo_val);
    free(O);
}

/* Loading sparse matrix from MatrixMarket file in CSR format with:
 * - fname: MM file UNIX path
 * - M: number of rows
 * - N: number of columns
 * - nslice: number of slices
 * - nnz: number of non-zeros
 * - sz_values: size of values array
 * - SO: slice offsets
 * - J: column indices
 * - values: non-zero values
 * - sz_slice: slice size */
void matrix_load_sell(char *fname,
                      int *M,
                      int *N,
                      int *nslices,
                      int *nnz,
                      int *sz_values,
                      int **nnz_per_row,
                      int **SO,
                      int **J,
                      float **values,
                      const int sz_slice)
{
    int i;
    int j;

    int *I;
    int *coo_J;
    int *O;

    float *coo_val;

    matrix_load_coo(fname, M, N, nnz, &I, &coo_J, &coo_val);

    *nslices = (*M + sz_slice - 1) / sz_slice;
    *sz_values = 0;

    *nnz_per_row = (int *)calloc(*M, sizeof(int));
    O = (int *)calloc(*M + 1, sizeof(int));
    *SO = (int *)calloc(*nslices + 1, sizeof(int));

    for (i = 0; i < *nnz; i++) (*nnz_per_row)[I[i]]++;

    for (i = 0; i < *M; i++) O[i + 1] = O[i] + (*nnz_per_row)[i];

    for (int s = 0; s < *M; s += sz_slice)
    {
        int slice_id = s / sz_slice;
        int slice_end = s + sz_slice;

        if (slice_end > *M) slice_end = *M;

        int max_nnz_per_row = 0;
        for (i = s; i < slice_end; i++)
        {
            if ((*nnz_per_row)[i] > max_nnz_per_row)
                max_nnz_per_row = (*nnz_per_row)[i];
        }

        (*SO)[slice_id + 1] = (*SO)[slice_id] + sz_slice * max_nnz_per_row;
    }

    *sz_values = (*SO)[*nslices];

    *J = (int *)malloc(*sz_values * sizeof(int));
    *values = (float *)malloc(*sz_values * sizeof(float));

    for (i = 0; i < *sz_values; i++)
    {
        (*J)[i] = -1;
        (*values)[i] = 0.0f;
    }

    for (int s = 0; s < *M; s += sz_slice)
    {
        int slice_id = s / sz_slice;
        int slice_end = s + sz_slice;

        if (slice_end > *M) slice_end = *M;

        int base = (*SO)[slice_id];

        for (i = s; i < slice_end; i++)
        {
            int local_row = i - s;

            for (j = O[i]; j < O[i + 1]; j++)
            {
                int k = j - O[i];
                int idx = base + sz_slice * k + local_row;

                (*J)[idx] = coo_J[j];
                (*values)[idx] = coo_val[j];
            }
        }
    }

    free(I);
    free(coo_J);
    free(coo_val);
    free(O);
}
