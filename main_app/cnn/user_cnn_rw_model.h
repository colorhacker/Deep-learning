#ifndef _user_cnn_rw_model_H
#define _user_cnn_rw_model_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_matrix_file.h"
#include "../cnn/user_cnn_layers.h"
#include "../other/user_nn_debug.h"

bool user_cnn_model_save_model(user_cnn_layers *layers,int id);//保存模型
user_cnn_layers	*user_cnn_model_load_model(int id);//载入层模型

#endif