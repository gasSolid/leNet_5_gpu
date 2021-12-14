/*************************************************************************
      > File Name: tensor.h
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Wed 08 Dec 2021 10:28:51 AM CST
 ************************************************************************/

#ifndef _THNEOR_H
#define _THNEOR_H
struct tensor{
	float *data;
	int rows;
	int cols;
	int height;
};

void alloc_tensor_host(tensor *mat, int rows, int cols, int height = 1);

void reshap_tensor(tensor *mat, int rows, int cols, int height = 1);

void alloc_tensor_device(tensor *mat, int rows, int cols, int height = 1);

void rand_init_tensor(tensor *ten);

bool check_tensor_dim(tensor *A, tensor *B);

bool tensor_memcpy_d2h(tensor *h_ten, tensor *d_ten);

bool tensor_memcpy_h2d(tensor *d_ten, tensor *h_ten);

inline void free_tensor_host(tensor *mat)
{	
	free(mat->data);
}

inline void free_tensor_device(tensor *mat)
{	
	cudaFree(mat->data);
}

bool is_same_tensor(tensor *mat_A, tensor *mat_B);

void print_tensor(tensor *mat);

void print_tensor_device(tensor *d_ten);

void unrolling_conv_tensor_host(int c1_kernel_size, tensor *in, tensor *h_input_tensor);

void unrolling_conv_tensor_device(int kernel_size, tensor *d_in, tensor *d_rets);

void tensor_dot_host(tensor *h_A, tensor *h_B, tensor *h_rets);

void tensor_dot_device(tensor *h_A, tensor *h_B, tensor *h_rets);

void sampling_device(tensor *d_c1, tensor *d_channel_weight, tensor *d_channel_bias, tensor *d_s2, int window_size);

void unrolling_conv_kernel_host(tensor *h_kernel_tensor,  int kernel_size, const bool *connect, tensor *h_kernel_matrix);

void tensor_dot_device_bias(tensor *d_A, tensor *d_bias, tensor *d_B, tensor *d_rets);

#endif
