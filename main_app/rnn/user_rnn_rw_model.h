#ifndef _user_rnn_rw_model_H
#define _user_rnn_rw_model_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix_file.h"
#include "../rnn/user_rnn_layers.h"



bool user_rnn_model_save_model(user_rnn_layers *layers,int id);//����ģ��
user_rnn_layers	*user_rnn_model_load_model(int id);//�����ģ��

#endif