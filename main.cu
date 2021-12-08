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
#include "matrix.h"
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
void rand_init_arr(float *arr, int arr_size){
	for (int i = 0; i < arr_size; i++){
		srand(i);		
		arr[i] = (rand() % 100) / 100.0;
		printf("%f ", arr[i]);
		//arr[i] = 0.0; 
	}
}

static int minst_size = 28;
static int input_size = 32;
matrix d_input_matrix;
matrix h_input_matrix;

static int c1_kernel_size = 5;
static int c1_channel_size = 3;
static int c1_size = input_size - c1_kernel_size + 1;
matrix d_c1_kernel_matrix;
matrix d_c1_bias_matrix;
matrix h_c1_kernel_matrix;

void init_all(){
	read_mnist_data();
	
	int input_matrix_rows = c1_kernel_size * c1_kernel_size;
	int input_matrix_cols = c1_size * c1_size;
	alloc_matrix_host(&h_input_matrix, input_matrix_rows, input_matrix_cols);
	alloc_matrix_device(&d_input_matrix, input_matrix_rows, input_matrix_cols);

	// init c1 kernel
	int c1_kernel_matrix_col = c1_kernel_size * c1_kernel_size;
	alloc_matrix_host(&h_c1_kernel_matrix, c1_channel_size, c1_kernel_matrix_col);
	rand_init_arr(h_c1_kernel_matrix.data, c1_channel_size * c1_kernel_matrix_col);
	alloc_matrix_device(&d_c1_kernel_matrix, c1_channel_size, c1_kernel_matrix_col);
	size_t bytes_c1_kernel_matrix = sizeof(float) * c1_channel_size * c1_kernel_matrix_col;
	cudaMemcpy(d_c1_kernel_matrix.data, h_c1_kernel_matrix.data, bytes_c1_kernel_matrix, cudaMemcpyHostToDevice);
	//free_matrix_host(&h_c1_kernel_matrix);
}

void free_all(){
	free_matrix_device(&d_input_matrix);
	free_matrix_device(&d_c1_kernel_matrix);

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
	// c1
	float *tmp_in = (float *)malloc(input_size * input_size * sizeof(float));
	memset(tmp_in, 0x00, input_size * input_size * sizeof(float));
	for (int i = 0; i < minst_size; i++){
		for (int j = 0; j < minst_size; j++){
			tmp_in[(i + 2) * input_size + j + 2] = input->data[i][j];
		}
	}

	// unrolling input image in host
	int unrolling_size = input_size - c1_kernel_size + 1;
	for (int i = 0; i < unrolling_size; i++){
		for (int j = 0; j < unrolling_size; j++){
			for (int x = 0; x< c1_kernel_size; x++){
				for (int y = 0; y < c1_kernel_size; y++){
					h_input_matrix.data[(x * c1_kernel_size + y) * h_input_matrix.cols + i * unrolling_size + j] = tmp_in[(i+x) * input_size + j + y];	
				}
			}
		}
	}
	size_t bytes_input_matrix = sizeof(float) * d_input_matrix.rows * d_input_matrix.cols;
	cudaMemcpy(d_input_matrix.data, h_input_matrix.data, bytes_input_matrix, cudaMemcpyHostToDevice);

	//matrix h_c1;
	//alloc_matrix_host(&h_c1, h_c1_kernel_matrix.rows, h_input_matrix.cols);
	//matrix_dot_host(&h_c1_kernel_matrix, &h_input_matrix, &h_c1);
	//reshap_matrix(&h_c1, 30, 30);
	//print_matrix(&h_c1);

	matrix d_c1;
	alloc_matrix_device(&d_c1, h_c1_kernel_matrix.rows, h_input_matrix.cols);
	matrix_dot_device(&d_c1_kernel_matrix, &d_input_matrix, &d_c1);

	matrix h_c2;
	alloc_matrix_host(&h_c2, h_c1_kernel_matrix.rows, h_input_matrix.cols);
	cudaMemcpy(h_c2.data, d_c1.data, sizeof(float) * h_c2.rows*h_c2.cols, cudaMemcpyDeviceToHost);
	reshap_matrix(&h_c2, 28 * 3, 28);
	print_matrix(&h_c2);
	//if(!is_same_matrix(&h_c1, &h_c2)){
	//	exit(0);
	//}

	free(tmp_in);
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
	return 0;
}
