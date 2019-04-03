#ifndef _user_nn_rw_model_H
#define _user_nn_rw_model_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix_file.h"
#include "../nn/user_nn_layers.h"

bool user_nn_model_save_model(user_nn_layers *layers, int id);//保存模型
user_nn_layers	*user_nn_model_load_model(int id);//载入层模型

#endif