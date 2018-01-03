#ifndef _user_nn_matrix_file_H
#define _user_nn_matrix_file_H

#include "user_nn_matrix.h"
long user_nn_model_save_matrix(FILE *file, long offset, user_nn_matrix *matrix);//写入矩阵到文件对象中
long user_nn_model_save_matrices(FILE *file, long offset, user_nn_list_matrix *matrices);//写入连续矩阵到文件对象中
long user_nn_model_read_matrix(FILE *file, long offset, user_nn_matrix *matrix);//读取一个矩阵从文件对象中
long user_nn_model_read_matrices(FILE *file, long offset, user_nn_list_matrix *matrices);//读取连续矩阵从文件对象中

bool user_nn_model_file_save_matrices(const char *path, long offset, user_nn_list_matrix *matrices);//保存连续矩阵到文件中
user_nn_list_matrix *user_nn_model_file_read_matrices(const char *path, long offset);//从文件中读取连续矩阵

#endif