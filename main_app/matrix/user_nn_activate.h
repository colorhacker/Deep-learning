#ifndef _user_nn_activate_H
#define _user_nn_activate_H

#include "../user_config.h"
#include "user_nn_matrix.h"

typedef enum _activation_type {
	activation_sigmoid = 0,//
	activation_tanh = 1//
}activation_type;


float user_nn_activate_softmax(float value, activation_type type);//对float参数进行softmax激活
float user_nn_activate_softmax_d(float value, activation_type type);//对float参数进行softmax求导

void user_nn_activate_matrix(user_nn_matrix *dest_matrix, activation_type type);//采用激活函数对矩阵进行激活处理
void user_nn_activate_matrix_d(user_nn_matrix *dest_matrix, activation_type type);//采用激活函数对矩阵进行求导处理

void user_nn_activate_matrix_sum_constant(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, float constant, activation_type type);//采用激活函数对矩阵加上一个值进行激活处理
void user_nn_activate_matrix_sum_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type);//采用激活函数对矩阵加上一个矩阵进行激活处理

void user_nn_activate_matrix_d_mult_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type);//对矩阵进行求导数，结果乘矩阵，进行返回
void user_nn_activate_matrices_d_mult_matrices(user_nn_list_matrix *save_matrices, user_nn_list_matrix *src_matrices, user_nn_list_matrix *sub_matrices, activation_type type);//对连续矩阵进行求导数，结果乘矩阵，进行返回

#endif