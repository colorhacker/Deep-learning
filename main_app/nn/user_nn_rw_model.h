#ifndef _user_nn_rw_model_H
#define _user_nn_rw_model_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix_file.h"
#include "../nn/user_nn_layers.h"

#define		user_nn_model_nn_file_name		"nn_model.bin"
#define		user_nn_model_nn_layer_addr		 0x0		//保存层的基地址
#define		user_nn_model_nn_content_addr	 0x800		//保存层对象的基地址
#define		user_nn_model_nn_data_addr		 0x1000		//保存数据的基地址

bool user_nn_model_save_model(const char *path,user_nn_layers *layers);//保存模型
user_nn_layers	*user_nn_model_load_model(const char *path);//载入层模型

#endif