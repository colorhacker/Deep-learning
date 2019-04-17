#ifndef _user_nn_initialization_H
#define _user_nn_initialization_H

#include "../user_config.h"
#include "user_nn_matrix.h"

#if defined lecun_uniform
	#define user_nn_init_rand(x,y) user_nn_init_lecun_uniform(x,y)
#elif defined glorot_normal
	#define user_nn_init_rand(x,y) user_nn_init_glorot_normal(x,y)
#elif defined glorot_uniform
	#define user_nn_init_rand(x,y) user_nn_init_glorot_uniform(x,y)
#elif defined he_normal
	#define user_nn_init_rand(x,y) user_nn_init_he_normal(x,y)
#else
	#define user_nn_init_rand(x,y) user_nn_init_he_uniform(x,y)
#endif

float user_nn_init_normal(void);
float user_nn_init_uniform(void);

float user_nn_init_lecun_uniform(int input_count, int output_count);
float user_nn_init_glorot_normal(int input_count, int output_count);
float user_nn_init_glorot_uniform(int input_count, int output_count);
float user_nn_init_he_normal(int input_count, int output_count);
float user_nn_init_he_uniform(int input_count, int output_count);

void user_nn_matrix_init_vaule(user_nn_matrix *src_matrix, int input, int output);//随机设置矩阵值给定参数
void user_nn_matrices_init_vaule(user_nn_list_matrix *list_matrix, int input, int output);//随机设置连续矩阵list_matrix值，post是系数

#endif
