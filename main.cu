/*************************************************************************
      > File Name: main.cu
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Wed 01 Dec 2021 08:52:49 PM CST
 ************************************************************************/

#define USE_MNIST_LOADER
#define MNIST_DOUBLE 
#include<stdio.h>
#include "mnist.h"
#include "tensor.h"
#include "time.h"
#define INPUT_SIZE 28

static mnist_data *train_data_set, *test_data_set;
static unsigned int train_data_count, test_data_count;

void read_mnist_data(){
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
			&train_data_set, &train_data_count);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
			&test_data_set, &test_data_count);
}


static int minst_size = 28;
static int input_size = 32;

static int c1_kernel_size = 5;
static int c1_parallel_channel_size = 6;
static int c1_relate_channel_size = 1;
static int c1_size = input_size - c1_kernel_size + 1;
// include bias in kernel tensor
tensor d_input_image;
tensor h_input_image;

//c1
tensor d_input_tensor;
tensor d_c1_kernel_tensor;
tensor d_c1;

//s2
tensor d_s2_channel_weight;
tensor d_s2_channel_bias;
int s2_window_size = 2; 
tensor d_s2;

void init_all(){
	read_mnist_data();
	
	// 1 is the size of bias
	int input_tensor_rows = c1_kernel_size * c1_kernel_size * c1_relate_channel_size + (1 * c1_relate_channel_size);
	int input_tensor_cols = c1_size * c1_size;
	alloc_tensor_device(&d_input_tensor, input_tensor_rows, input_tensor_cols);

	alloc_tensor_host(&h_input_image, input_size, input_size);
	alloc_tensor_device(&d_input_image, input_size, input_size);


	// init c1
	tensor h_c1_kernel_tensor;
	int c1_kernel_tensor_col = c1_kernel_size * c1_kernel_size * c1_relate_channel_size + (1 * c1_relate_channel_size);
	alloc_tensor_host(&h_c1_kernel_tensor, c1_parallel_channel_size, c1_kernel_tensor_col);
	rand_init_tensor(&h_c1_kernel_tensor);
	alloc_tensor_device(&d_c1_kernel_tensor, c1_parallel_channel_size, c1_kernel_tensor_col);
	tensor_memcpy_h2d(&d_c1_kernel_tensor, &h_c1_kernel_tensor);
	free_tensor_host(&h_c1_kernel_tensor);
	alloc_tensor_device(&d_c1, c1_size, c1_size, c1_parallel_channel_size);
	
	// init s2
	tensor h_s2_channel_weight;
	tensor h_s2_channel_bias;
	alloc_tensor_host(&h_s2_channel_weight, c1_parallel_channel_size * c1_relate_channel_size, 1);
	alloc_tensor_host(&h_s2_channel_bias, c1_parallel_channel_size * c1_relate_channel_size, 1);
	rand_init_tensor(&h_s2_channel_weight);
	rand_init_tensor(&h_s2_channel_bias);

	alloc_tensor_device(&d_s2_channel_weight, c1_parallel_channel_size * c1_relate_channel_size, 1);
	alloc_tensor_device(&d_s2_channel_bias, c1_parallel_channel_size * c1_relate_channel_size, 1);
	tensor_memcpy_h2d(&d_s2_channel_weight, &h_s2_channel_weight);
	tensor_memcpy_h2d(&d_s2_channel_bias, &h_s2_channel_bias);
	free_tensor_host(&h_s2_channel_weight);
	free_tensor_host(&h_s2_channel_bias);
	
	if (d_c1.rows % s2_window_size != 0 || d_c1.cols % s2_window_size != 0){
		printf("Error! can not sampling in line: %d of file: %d\n", __LINE__, __FILE__);
		exit(0);
	}
	alloc_tensor_device(&d_s2, d_c1.rows / s2_window_size, d_c1.cols / s2_window_size, d_c1.height);
}

void free_all(){
	free_tensor_device(&d_input_image);
	free_tensor_host(&h_input_image);

	free_tensor_device(&d_input_tensor);
	free_tensor_device(&d_c1_kernel_tensor);
	free_tensor_device(&d_c1);

	free_tensor_device(&d_s2_channel_weight);
	free_tensor_device(&d_s2_channel_bias);
}

void print_input_image(const mnist_data *out){
	for (int i = 0; i < minst_size; i++){
		for (int j = 0; j < minst_size; j++){
			printf("%4.3f ", out->data[i][j]);
		}
		printf("\n");
	}
	printf("%d\n", out->label);
}


void forward_prop(mnist_data *input){
	for (int i = 0; i < minst_size; i++){
		for (int j = 0; j < minst_size; j++){
			h_input_image.data[(i + 2) * input_size + j + 2] = input->data[i][j];
		}
	}
	tensor_memcpy_h2d(&d_input_image, &h_input_image);

	// c1
	// unrolling input image in device 
	unrolling_conv_tensor_device(c1_kernel_size, &d_input_image, &d_input_tensor);
	tensor_dot_device(&d_c1_kernel_tensor, &d_input_tensor, &d_c1);
	print_tensor_device(&d_c1);
	
	// s2
	sampling_device(&d_c1, &d_s2_channel_weight, &d_s2_channel_bias, &d_s2, s2_window_size);
	print_tensor_device(&d_s2);
}

void learn(){
	for (int i = 0; i < 1; i++){
		forward_prop(test_data_set);
	}
}


int main(){

	init_all();
	//print_input_image(&(test_data_set[0]));
	learn();
	//test();
	free_all();
	printf("End\n");
	return 0;
}
