#ifndef _user_nn_activate_H
#define _user_nn_activate_H

#include "../user_config.h"
#include "user_nn_matrix.h"

typedef enum _activation_type {
	activation_sigmoid = 0,//
	activation_tanh = 1,//
	activation_prelu = 2
}activation_type;


float user_nn_activate_softmax(float value, activation_type type);//��float��������softmax����
float user_nn_activate_softmax_d(float value, activation_type type);//��float��������softmax��

void user_nn_activate_matrix(user_nn_matrix *dest_matrix, activation_type type);//���ü�����Ծ�����м����
void user_nn_activate_matrix_d(user_nn_matrix *dest_matrix, activation_type type);//���ü�����Ծ�������󵼴���

void user_nn_activate_matrix_sum_constant(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, float constant, activation_type type);//���ü�����Ծ������һ��ֵ���м����
void user_nn_activate_matrix_sum_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type);//���ü�����Ծ������һ��������м����

void user_nn_activate_matrix_d_mult_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type);//�Ծ����������������˾��󣬽��з���
void user_nn_activate_matrices_d_mult_matrices(user_nn_list_matrix *save_matrices, user_nn_list_matrix *src_matrices, user_nn_list_matrix *sub_matrices, activation_type type);//�����������������������˾��󣬽��з���

#endif