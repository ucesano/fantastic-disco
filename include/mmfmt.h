#ifndef MM_FMT_H
#define MM_FMT_H

void mm_sort_coo(int *I, int *J, float *val, const int nz);

void mm_coo_to_csr_row_ptr(const int *I, const int nz, const int M, int *O);

void mm_csr_to_ellpack(const int *O, const int *J, const float *val, const int M, int *max_nz, int **ell_J, float **ell_val);

void mm_csr_to_sell_c(const int *O, const int *J, const float *val, const int M, const int C, int *num_slices, int **sell_O, int **sell_J, float **sell_val);

#endif