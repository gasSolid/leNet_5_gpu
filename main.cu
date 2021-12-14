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
tensor d_input_matrix;
tensor d_c1_kernel_matrix;
tensor d_c1;

//s2
tensor d_s2_channel_weight;
tensor d_s2_channel_bias;
int s2_window_size = 2; 
tensor d_s2;

//c3
static int c3_kernel_size = 5;
static int c3_parallel_channel_size = 16;
static int c3_relate_channel_size = 6;
tensor d_c3_kernel_matrix; // unrolling of kernel weight of c3
tensor d_c3_input_matrix; // unrolling of s2
tensor d_c3;

// s4
tensor d_s4_channel_weight;
tensor d_s4_channel_bias;
int s4_window_size = 2; 
tensor d_s4;
const bool c3_connect[16*6] =
{
	true, true, true, false, false, false,
	false, true, true, true, false, false,
	false, false, true, true, true, false,
	false, false, false, true, true, true,
	true, false, false, false, true, true,
	true, true, false, false, false, true,
	true, true, true, true, false, false,
	false, true, true, true, true, false,
	false, false, true, true, true, true,
	true, false, false, true, true, true,
	true, true, false, false, true, true,
	true, true, true, false, false, true,
	true, true, false, true, true, false,
	false, true, true, false, true, true,
	true, false, true, true, false, true,
	true, true, true, true, true, true};

// c5
static int c5_kernel_size = 5;
static int c5_parallel_channel_size = 120;
static int c5_relate_channel_size = 16;
tensor d_c5_kernel_matrix; // unrolling of kernel weight of c3
tensor d_c5_input_matrix; // unrolling of kernel weight of c3
tensor d_c5;


// f6 
static int f6_parallel_channel_size = 84;
tensor d_f6_kernel_matrix; 
tensor d_f6_kernel_bias_matrix; 
tensor d_f6; 

// fout
static int fout_parallel_channel_size = 10;
tensor d_fout_kernel_matrix; 
tensor d_fout_kernel_bias_matrix; 
tensor d_fout; 

void init_all(){
	read_mnist_data();

	// 1 is the size of bias
	int input_tensor_rows = c1_kernel_size * c1_kernel_size * c1_relate_channel_size + (1 * c1_relate_channel_size);
	int input_tensor_cols = c1_size * c1_size;
	alloc_tensor_device(&d_input_matrix, input_tensor_rows, input_tensor_cols);

	alloc_tensor_host(&h_input_image, input_size, input_size);
	alloc_tensor_device(&d_input_image, input_size, input_size);


	// init c1
	tensor h_c1_kernel_tensor;
	int c1_kernel_tensor_col = c1_kernel_size * c1_kernel_size * c1_relate_channel_size + (1 * c1_relate_channel_size);
	alloc_tensor_host(&h_c1_kernel_tensor, c1_parallel_channel_size, c1_kernel_tensor_col);
	rand_init_tensor(&h_c1_kernel_tensor);
	alloc_tensor_device(&d_c1_kernel_matrix, c1_parallel_channel_size, c1_kernel_tensor_col);
	tensor_memcpy_h2d(&d_c1_kernel_matrix, &h_c1_kernel_tensor);
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

	// c3
	tensor h_c3_kernel_tensor; //include bias
	tensor h_c3_kernel_matrix;
	alloc_tensor_host(&h_c3_kernel_tensor, 1, c3_kernel_size * c3_kernel_size + 1, c3_relate_channel_size);
	rand_init_tensor(&h_c3_kernel_tensor);
	alloc_tensor_host(&h_c3_kernel_matrix, c3_parallel_channel_size, (c3_kernel_size * c3_kernel_size + 1) * c3_relate_channel_size);
	memset(h_c3_kernel_matrix.data, 0x00, sizeof(float) * h_c3_kernel_matrix.rows * h_c3_kernel_matrix.cols);
	unrolling_conv_kernel_host(&h_c3_kernel_tensor, c3_kernel_size, c3_connect, &h_c3_kernel_matrix);
	//print_tensor(&h_c3_kernel_matrix);
	alloc_tensor_device(&d_c3_kernel_matrix, c3_parallel_channel_size, (c3_kernel_size * c3_kernel_size + 1) * c3_relate_channel_size);
	tensor_memcpy_h2d(&d_c3_kernel_matrix, &h_c3_kernel_matrix);
	alloc_tensor_device(&d_c3_input_matrix,  c3_relate_channel_size * (c3_kernel_size * c3_kernel_size + 1), (d_s2.rows - c3_kernel_size + 1) * (d_s2.cols - c3_kernel_size + 1));
	alloc_tensor_device(&d_c3, (d_s2.rows - c3_kernel_size + 1), (d_s2.cols - c3_kernel_size + 1), c3_parallel_channel_size);
	free_tensor_host(&h_c3_kernel_tensor);
	free_tensor_host(&h_c3_kernel_matrix);

	// init s4
	tensor h_s4_channel_weight;
	tensor h_s4_channel_bias;
	alloc_tensor_host(&h_s4_channel_weight, c3_parallel_channel_size, 1);
	alloc_tensor_host(&h_s4_channel_bias, c3_parallel_channel_size, 1);
	rand_init_tensor(&h_s4_channel_weight);
	rand_init_tensor(&h_s4_channel_bias);

	alloc_tensor_device(&d_s4_channel_weight, c3_parallel_channel_size, 1);
	alloc_tensor_device(&d_s4_channel_bias, c3_parallel_channel_size, 1);
	tensor_memcpy_h2d(&d_s4_channel_weight, &h_s4_channel_weight);
	tensor_memcpy_h2d(&d_s4_channel_bias, &h_s4_channel_bias);
	free_tensor_host(&h_s4_channel_weight);
	free_tensor_host(&h_s4_channel_bias);

	if (d_c3.rows % s4_window_size != 0 || d_c3.cols % s4_window_size != 0){
		printf("Error! can not sampling in line: %d of file: %d\n", __LINE__, __FILE__);
		exit(0);
	}
	alloc_tensor_device(&d_s4, d_c3.rows / s4_window_size, d_c3.cols / s4_window_size, d_c3.height);

	
	// c5 
	tensor h_c5_kernel_matrix;
	alloc_tensor_host(&h_c5_kernel_matrix, c5_parallel_channel_size, c5_relate_channel_size*(c5_kernel_size * c5_kernel_size + 1));
	rand_init_tensor(&h_c5_kernel_matrix);
	alloc_tensor_device(&d_c5_kernel_matrix, h_c5_kernel_matrix.rows, h_c5_kernel_matrix.cols);
	alloc_tensor_device(&d_c5_input_matrix, c5_relate_channel_size * (c5_kernel_size * c5_kernel_size + 1), (d_s4.rows - c5_kernel_size + 1) * (d_s4.cols - c5_kernel_size + 1));
	tensor_memcpy_h2d(&d_c5_kernel_matrix, &h_c5_kernel_matrix);
	alloc_tensor_device(&d_c5, c5_parallel_channel_size, 1);
	free_tensor_host(&h_c5_kernel_matrix);

	// f6
	tensor h_f6_kernel_matrix;
	tensor h_f6_kernel_bias_matrix;

	alloc_tensor_host(&h_f6_kernel_matrix, f6_parallel_channel_size, c5_parallel_channel_size);
	alloc_tensor_host(&h_f6_kernel_bias_matrix, f6_parallel_channel_size, 1);
	rand_init_tensor(&h_f6_kernel_matrix);
	rand_init_tensor(&h_f6_kernel_bias_matrix);
	alloc_tensor_device(&d_f6_kernel_matrix, h_f6_kernel_matrix.rows, h_f6_kernel_matrix.cols);
	alloc_tensor_device(&d_f6_kernel_bias_matrix, h_f6_kernel_bias_matrix.rows, h_f6_kernel_bias_matrix.cols);
	alloc_tensor_device(&d_f6, f6_parallel_channel_size, 1);
	tensor_memcpy_h2d(&d_f6_kernel_matrix, &h_f6_kernel_matrix);
	tensor_memcpy_h2d(&d_f6_kernel_bias_matrix, &h_f6_kernel_bias_matrix);

	// fout
	tensor h_fout_kernel_matrix;
	tensor h_fout_kernel_bias_matrix;
	alloc_tensor_host(&h_fout_kernel_matrix, fout_parallel_channel_size, f6_parallel_channel_size);
	alloc_tensor_host(&h_fout_kernel_bias_matrix, fout_parallel_channel_size, 1);
	rand_init_tensor(&h_fout_kernel_matrix);
	rand_init_tensor(&h_fout_kernel_bias_matrix);
	alloc_tensor_device(&d_fout_kernel_matrix, h_fout_kernel_matrix.rows, h_fout_kernel_matrix.cols);
	alloc_tensor_device(&d_fout_kernel_bias_matrix, h_fout_kernel_bias_matrix.rows, h_fout_kernel_bias_matrix.cols);
	alloc_tensor_device(&d_fout, fout_parallel_channel_size, 1);
	tensor_memcpy_h2d(&d_fout_kernel_matrix, &h_fout_kernel_matrix);
	tensor_memcpy_h2d(&d_fout_kernel_bias_matrix, &h_fout_kernel_bias_matrix);
}

void free_all(){
	free_tensor_device(&d_input_image);
	free_tensor_host(&h_input_image);

	//c1
	free_tensor_device(&d_input_matrix);
	free_tensor_device(&d_c1_kernel_matrix);
	free_tensor_device(&d_c1);

	//s2
	free_tensor_device(&d_s2_channel_weight);
	free_tensor_device(&d_s2_channel_bias);
	free_tensor_device(&d_s2);

	//c3
	free_tensor_device(&d_c3_kernel_matrix);
	free_tensor_device(&d_c3_input_matrix);
	free_tensor_device(&d_c3);

	//s4
	free_tensor_device(&d_s4_channel_weight);
	free_tensor_device(&d_s4_channel_bias);
	free_tensor_device(&d_s4);
	
	//c5
	free_tensor_device(&d_c5_kernel_matrix);
	free_tensor_device(&d_c5_input_matrix);
	free_tensor_device(&d_c5);
	
	// f6
	free_tensor_device(&d_f6_kernel_matrix);
	free_tensor_device(&d_f6_kernel_bias_matrix);
	free_tensor_device(&d_f6);

	// fout
	free_tensor_device(&d_fout_kernel_matrix);
	free_tensor_device(&d_fout_kernel_bias_matrix);
	free_tensor_device(&d_fout);
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
	unrolling_conv_tensor_device(c1_kernel_size, &d_input_image, &d_input_matrix);
	tensor_dot_device(&d_c1_kernel_matrix, &d_input_matrix, &d_c1);

	// s2
	sampling_device(&d_c1, &d_s2_channel_weight, &d_s2_channel_bias, &d_s2, s2_window_size);
	//print_tensor_device(&d_s2);

	//c3 
	//unrolling_conv_tensor_device(c3_kernel_size, &d_s2, &d_c3_input_matrix);
	tensor_dot_device(&d_c3_kernel_matrix, &d_c3_input_matrix, &d_c3);
	//print_tensor_device(&d_c3);
	
	// s4
	sampling_device(&d_c3, &d_s4_channel_weight, &d_s4_channel_bias, &d_s4, s4_window_size);
	//print_tensor_device(&d_s4);
	
	// c5
	unrolling_conv_tensor_device(c5_kernel_size, &d_s4, &d_c5_input_matrix);
	tensor_dot_device(&d_c5_kernel_matrix, &d_c5_input_matrix, &d_c5);
	//print_tensor_device(&d_c5);
	
	// f6
	tensor_dot_device_bias(&d_f6_kernel_matrix, &d_f6_kernel_bias_matrix, &d_c5, &d_f6);
	//print_tensor_device(&d_f6);

	// fout
	tensor_dot_device_bias(&d_fout_kernel_matrix, &d_fout_kernel_bias_matrix, &d_f6, &d_fout);
	print_tensor_device(&d_fout);
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
