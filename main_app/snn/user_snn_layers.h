

#ifndef _user_snn_layers_H
#define _user_snn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _thred_type {
	thred_none = 0,//
	thred_heighten = 1,//期待值高
	thred_lower = 2//期待值低
}thred_type;

void user_snn_data_softmax(user_nn_matrix *src_matrix);
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix * max_matrix);
user_nn_matrix *user_nn_matrix_thred_process(user_nn_matrix *src_matrix, user_nn_matrix *target_matrix);
user_nn_matrix *user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix);//矩阵阈值累加
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *thred_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, float avg_value, float step_value);//更新阈值


#endif