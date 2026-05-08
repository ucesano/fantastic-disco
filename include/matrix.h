#ifndef MATRIX_H
#define MATRIX_H

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
                      float **values);

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
                      float **values);


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
                      float **values);

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
                      const int sz_slice);

#endif
