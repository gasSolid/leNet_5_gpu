/*************************************************************************
      > File Name: matrix.h
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Wed 08 Dec 2021 10:28:51 AM CST
 ************************************************************************/

#ifndef _MATRIX_H
#define _MATRIX_H
struct matrix{
	float *data;
	int rows;
	int cols;
};

void alloc_matrix_host(matrix *mat, int rows, int cols);

void reshap_matrix(matrix *mat, int rows, int cols);

void alloc_matrix_device(matrix *mat, int rows, int cols);

inline void free_matrix_host(matrix *mat)
{	
	free(mat->data);
}

inline void free_matrix_device(matrix *mat)
{	
	cudaFree(mat->data);
}

bool is_same_matrix(matrix *mat_A, matrix *mat_B);

void print_matrix(matrix *mat);

void matrix_dot_host(matrix *h_A, matrix *h_B, matrix *h_rets);

void matrix_dot_device(matrix *h_A, matrix *h_B, matrix *h_rets);

#endif
