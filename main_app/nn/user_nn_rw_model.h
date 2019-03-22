#ifndef _user_nn_rw_model_H
#define _user_nn_rw_model_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix_file.h"
#include "../nn/user_nn_layers.h"

bool user_nn_model_save_model(const char *path,user_nn_layers *layers);//保存模型
user_nn_layers	*user_nn_model_load_model(const char *path);//载入层模型

#endif