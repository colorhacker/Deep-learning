#ifndef _user_nn_initialization_H
#define _user_nn_initialization_H

#include "../user_config.h"
#include "user_nn_matrix.h"

#define user_nn_init_rand(x,y) user_nn_init_lecun_uniform(x,y)

float user_nn_init_lecun_uniform(int input_count, int output_count);
float user_nn_init_glorot_normal(int input_count, int output_count);
float user_nn_init_glorot_uniform(int input_count, int output_count);
float user_nn_init_he_normal(int input_count, int output_count);
float user_nn_init_he_uniform(int input_count, int output_count);

void user_nn_matrix_init_vaule(user_nn_matrix *src_matrix, int input, int output);//������þ���ֵ��������
void user_nn_matrices_init_vaule(user_nn_list_matrix *list_matrix, int input, int output);//���������������list_matrixֵ��post��ϵ��

#endif
