/*************************************************************************
      > File Name: matrix.c
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Wed 08 Dec 2021 10:29:00 AM CST
 ************************************************************************/

#include<stdio.h>
#include "matrix.h"

#define TILE 16

void alloc_matrix_host(matrix *mat, int rows, int cols)
{
	mat->rows = rows;
	mat->cols = cols;
	mat->data = (float *)malloc(rows * cols * sizeof(float));
}

void alloc_matrix_device(matrix *mat, int rows, int cols)
{
	mat->rows = rows;
	mat->cols = cols;
	cudaMalloc((void **)&(mat->data), rows * cols * sizeof(float));
}

void reshap_matrix(matrix *mat, int rows, int cols)
{
	if (mat->rows * mat->cols != rows * cols){
		printf("dimesion is not consist in line %d of file %s\n", __LINE__, __FILE__);
		exit(0);
	}
	mat->rows = rows;
	mat->cols = cols;
}

void print_matrix(matrix *mat){
	printf("matrix rows: %d, cols %d\n", mat->rows, mat->cols);
	for (int i = 0; i < mat->rows; i++){
		for (int j = 0; j < mat->cols; j++){
			printf("%6.3f ", mat->data[i * mat->cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}
bool is_same_matrix(matrix *mat_A, matrix *mat_B)
{
	int rows_A = mat_A->rows;
	int cols_A = mat_A->cols;
	int rows_B = mat_B->rows;
	int cols_B = mat_B->cols;
	if (rows_A != rows_B || cols_A != cols_B){
		printf("dimension of matrix A is (%d, %d) and B is (%d, %d)!\n", rows_A, cols_A, rows_B, cols_B);
		return false;
	}
	for (int i = 0; i < rows_A * rows_B; i++){
		if (fabs(mat_A->data[i] - mat_B->data[i]) > 1.0E-6){
			printf("value difference in matrix A: %f, matrix B %f!\n", mat_A->data[i], mat_B->data[i]);
			return false;
		}
	}
	return true;
}

void matrix_dot_host(matrix *h_A, matrix *h_B, matrix *h_rets){
	int rows_A = h_A->rows;
	int cols_A = h_A->cols;
	int rows_B = h_B->rows;
	int cols_B = h_B->cols;
	if (cols_A != rows_B){
		printf("error in %d %s!\n", __LINE__, __FILE__);
		exit(0);
	}
	for (int i = 0; i < rows_A; i++){
		for (int j = 0; j < cols_B; j++){
			float val = 0.0;
			for (int k = 0; k < cols_A; k++){
				val += h_A->data[i * cols_A + k] * h_B->data[k * cols_B + j];
			}
			h_rets->data[i * cols_B + j] = val;
		}
	}
}

__global__ void matrix_dot_device_kernel_reg(float *d_A, int rows_A, int cols_A, float *d_B, int cols_B, float *d_C){

        int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ float shmm_B[TILE][TILE];
	int n_proc = (cols_A + TILE - 1) / TILE;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float tmp = 0.0;
	for (int i = 0; i < n_proc; i++){
		// replace shared memory using register file
		float reg_A = 0.0;
		if (row < rows_A)
			reg_A = d_A[row * cols_A + i * TILE + tx];
		if (col < cols_B)
			shmm_B[ty][tx] = d_B[(i * TILE + ty) * cols_B + col];
		else
			shmm_B[ty][tx] = 0;
		__syncthreads();

		for (int j = 0; j < TILE; j++){
			tmp += reg_A * shmm_B[(tx + j) % TILE][tx];
			reg_A = __shfl(reg_A, tx + 1, TILE);
		}
		__syncthreads();
	}
	if (row < rows_A && col < cols_B)
		d_C[row * cols_B + col] = tmp;
}

__global__ void matrix_dot_device_kernel_shmm(float *d_A, int rows_A, int cols_A, float *d_B, int cols_B, float *d_C){

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ float shmm_A[TILE][TILE];
	__shared__ float shmm_B[TILE][TILE];
	int n_proc = (cols_A + TILE - 1) / TILE;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float tmp = 0.0;
	for (int i = 0; i < n_proc; i++){
		if (row < rows_A)        
			shmm_A[ty][tx] = d_A[row * cols_A + i * TILE + tx];      
		else 
			shmm_A[ty][tx] = 0;     

		if (col < cols_B)
			shmm_B[ty][tx] = d_B[(i * TILE + ty) * cols_B + col];
		else 
			shmm_B[ty][tx] = 0; 

		__syncthreads();
		for (int j = 0; j < TILE; j++){
			tmp += shmm_A[ty][j] * shmm_B[j][tx];
		}
		__syncthreads();
	}
	if (row < rows_A && col < cols_B)
		d_C[row * cols_B + col] = tmp;
}

void matrix_dot_device(matrix *d_A, matrix *d_B, matrix *d_rets)
{
	int rows_A = d_A->rows;
	int cols_A = d_A->cols;
	int rows_B = d_B->rows;
	int cols_B = d_B->cols;
	if (cols_A != rows_B){
		printf("error in %d %s!\n", __LINE__, __FILE__);
		exit(0);
	}

	dim3 blocks(TILE, TILE);
	int block_x = (cols_B + TILE - 1) / TILE;
	int block_y = (rows_A + TILE - 1) / TILE;
	dim3 grids(block_x, block_y);
	printf("grid x: %d, y: %d\n", block_x, block_y);

	matrix_dot_device_kernel_reg<<<grids, blocks>>>(d_A->data, rows_A, cols_A, d_B->data, cols_B, d_rets->data);
}
