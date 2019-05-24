#ifndef _user_snn_rw_model_H
#define _user_snn_rw_model_H

#include "user_snn_layers.h"
#include "../user_config.h"
#include "../matrix/user_nn_matrix_file.h"

bool user_snn_model_save_model(user_snn_layers *layers, int id);//保存模型
user_snn_layers	*user_snn_model_load_model(int id);//载入层模型

#endif