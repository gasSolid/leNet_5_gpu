/*************************************************************************
      > File Name: tensor.c
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Wed 08 Dec 2021 10:29:00 AM CST
 ************************************************************************/

#include<stdio.h>
#include "tensor.h"

#define TILE 16
#define E 2.718281828459

void alloc_tensor_host(tensor *mat, int rows, int cols, int height)
{
	mat->rows = rows;
	mat->cols = cols;
	mat->height = height;
	mat->data = (float *)malloc(rows * cols * height * sizeof(float));
}

void alloc_tensor_device(tensor *mat, int rows, int cols, int height)
{
	mat->rows = rows;
	mat->cols = cols;
	mat->height = height;
	cudaMalloc((void **)&(mat->data), rows * cols * height * sizeof(float));
}

void rand_init_tensor(tensor *ten){
	for (int i = 0; i < ten->height * ten->rows * ten->cols; i++){
		srand(i);		
		ten->data[i] = (rand() % 100) / 100.0;
	}
}

bool check_tensor_dim(tensor *A, tensor *B){
	return A->height == B->height && A->rows == B->rows && A->cols == B->cols;
}

bool tensor_memcpy_d2h(tensor *h_ten, tensor *d_ten){
	if (!check_tensor_dim(d_ten, h_ten)){
		printf("Error, dimesion is not consist in line %d of file %s\n", __LINE__, __FILE__);
		return false;
	}
	size_t bytes = sizeof(float) * h_ten->height * h_ten->rows * h_ten->cols;
	cudaMemcpy(h_ten->data, d_ten->data, bytes, cudaMemcpyDeviceToHost);
	return true;
}

bool tensor_memcpy_h2d(tensor *d_ten, tensor *h_ten){
	if (!check_tensor_dim(d_ten, h_ten)){
		printf("Error, dimesion is not consist in line %d of file %s\n", __LINE__, __FILE__);
		return false;
	}
	size_t bytes = sizeof(float) * h_ten->height * h_ten->rows * h_ten->cols;
	cudaMemcpy(d_ten->data, h_ten->data, bytes, cudaMemcpyHostToDevice);
	return true;
}

void reshap_tensor(tensor *mat, int rows, int cols, int height)
{
	if (mat->rows * mat->cols * mat->height != rows * cols * height){
		printf("Error, dimesion is not consist in line %d of file %s\n", __LINE__, __FILE__);
		exit(0);
	}
	mat->rows = rows;
	mat->cols = cols;
	mat->height = height;
}

void print_tensor(tensor *mat){
	printf("tensor rows: %d, cols %d, height: %d\n", mat->rows, mat->cols, mat->height);
	for (int i = 0; i < mat->height; i++){
		for (int j = 0; j < mat->rows; j++){
			for (int k = 0; k < mat->cols; k++){
				printf("%6.3f ", mat->data[i * mat->rows * mat->cols + j * mat->cols + k]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

void print_tensor_device(tensor *d_ten){
	tensor h_ten;
	alloc_tensor_host(&h_ten, d_ten->rows, d_ten->cols, d_ten->height);
	cudaMemcpy(h_ten.data, d_ten->data, sizeof(float) * h_ten.rows * h_ten.cols * h_ten.height, cudaMemcpyDeviceToHost);
	print_tensor(&h_ten);
	free_tensor_host(&h_ten);
}
bool is_same_tensor(tensor *mat_A, tensor *mat_B)
{
	int rows_A = mat_A->rows;
	int cols_A = mat_A->cols;
	int height_A = mat_A->height;
	int rows_B = mat_B->rows;
	int cols_B = mat_B->cols;
	int height_B = mat_B->height;
	if (rows_A != rows_B || cols_A != cols_B || height_A != height_B){
		printf("dimension of tensor A is (%d, %d, %d) and B is (%d, %d, %d)!\n", rows_A, cols_A, height_A, rows_B, cols_B, height_B);
		return false;
	}
	for (int i = 0; i < rows_A * cols_A * height_A; i++){
		if (fabs(mat_A->data[i] - mat_B->data[i]) > 1.0E-6){
			printf("value difference in tensor A: %f, tensor B %f!\n", mat_A->data[i], mat_B->data[i]);
			return false;
		}
	}
	return true;
}

void tensor_dot_host(tensor *h_A, tensor *h_B, tensor *h_rets){
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

void unrolling_conv_tensor_host(int kernel_size, tensor *h_in, tensor *h_rets){
	int unrolling_size_rows = h_in->rows - kernel_size + 1;
	int unrolling_size_cols = h_in->cols - kernel_size + 1;
	for (int i = 0; i < unrolling_size_rows; i++){
		for (int j = 0; j < unrolling_size_cols; j++){
			// unroling one convolution
			for (int x = 0; x< kernel_size; x++){
				for (int y = 0; y < kernel_size; y++){
					h_rets->data[(x * kernel_size + y) * h_rets->cols + i * unrolling_size_cols + j] = h_in->data[(i+x) * h_in->cols + j + y];	
				}
			}
			// bias 
			//h_rets->data[(kernel_size * kernel_size - 1 + 1) * h_rets->cols + i * unrolling_size + j] = 1.0;	
			h_rets->data[(kernel_size * kernel_size) * h_rets->cols + i * unrolling_size_cols + j] = 1.0;	
		}
	}
}

__global__ void unrolling_conv_tensor_kernel (float *d_in, int in_cols, float *d_rets, int rets_cols, int kernel_size, int unrolling_size_rows, int unrolling_size_cols){
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row >= unrolling_size_rows || col >= unrolling_size_cols){
		return ;
	}
	// unroling one convolution
	for (int x = 0; x< kernel_size; x++){
		for (int y = 0; y < kernel_size; y++){
			d_rets[(x * kernel_size + y) * rets_cols + row * unrolling_size_cols + col] = d_in[(row+x) * in_cols + col + y];	
		}
	}
	// bias 
	d_rets[(kernel_size * kernel_size) * rets_cols + row * unrolling_size_cols + col] = 1.0;	
}

void unrolling_conv_tensor_device(int kernel_size, tensor *d_in, tensor *d_rets){
	int unrolling_size_rows = d_in->rows - kernel_size + 1;
	int unrolling_size_cols = d_in->cols - kernel_size + 1;

	dim3 blocks(TILE, TILE);
	int block_x = (unrolling_size_cols + TILE - 1) / TILE;
	int block_y = (unrolling_size_rows + TILE - 1) / TILE;
	dim3 grids(block_x, block_y);
	printf("grid x: %d, y: %d\n", block_x, block_y);
	unrolling_conv_tensor_kernel<<<grids, blocks>>>(d_in->data, d_in->cols, d_rets->data, d_rets->cols, kernel_size, unrolling_size_rows, unrolling_size_cols);
}

__global__ void tensor_dot_device_kernel_reg(float *d_A, int rows_A, int cols_A, float *d_B, int cols_B, float *d_C){

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

__global__ void tensor_dot_device_kernel_shmm(float *d_A, int rows_A, int cols_A, float *d_B, int cols_B, float *d_C){

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

void tensor_dot_device(tensor *d_A, tensor *d_B, tensor *d_rets)
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

	tensor_dot_device_kernel_reg<<<grids, blocks>>>(d_A->data, rows_A, cols_A, d_B->data, cols_B, d_rets->data);
}

__device__ float tanh_device(float x){
	float a = powf(E, x);
	float b = powf(E, -x);
	return (a-b) / (a+b);
}

__global__ void sampling_device_kernel(float *d_c, float *d_channel_weight, float *d_channel_bias, float *d_s, int c_cols, int sum_rows, int cols, int rows, int window_size)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row >= sum_rows || col >= cols) return;
	int channel_id = row / rows;
	float rets = 0.0;
	int row_start = row * window_size;
	int row_end = (row + 1) * window_size;
	int col_start = col * window_size;
	int col_end = (col + 1) * window_size;
	for (int i = row_start; i < row_end; i++){
		for (int j = col_start; j < col_end; j++){
			rets += d_c[i * c_cols + j]; 
		}
	}
	rets = rets * d_channel_weight[channel_id] + d_channel_bias[channel_id];
	d_s[row * cols + col] = tanh_device(rets);
}
void sampling_device(tensor *d_c, tensor *d_channel_weight, tensor *d_channel_bias, tensor *d_s2, int window_size)
{
	int sum_rows = d_s2->rows * d_s2->height;	
 	int rows = d_s2->rows;	
	int cols = d_s2->cols;	

	dim3 blocks(TILE, TILE);
	int block_x = (cols + TILE - 1) / TILE;
	int block_y = (sum_rows + TILE - 1) / TILE;
	dim3 grids(block_x, block_y);
	printf("grid x: %d, y: %d\n", block_x, block_y);
	sampling_device_kernel<<<grids, blocks>>>(d_c->data, d_channel_weight->data, d_channel_bias->data, d_s2->data, d_c->cols, sum_rows, cols, rows, window_size);
}
