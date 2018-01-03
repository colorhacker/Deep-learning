#ifndef _user_rnn_model_H
#define _user_rnn_model_H

#include "../user_config.h"
#include "user_rnn_layers.h"


user_rnn_layers *user_rnn_model_create(int *layer_infor);//创建rnn模型
user_rnn_layers *user_rnn_model_return_layer(user_rnn_layers *layers, user_rnn_layer_type type);
void user_rnn_model_load_input_feature(user_rnn_layers *layers, user_nn_list_matrix *src_matrix);
void user_rnn_model_load_target_feature(user_rnn_layers *layers, user_nn_list_matrix *src_matrix);
void user_rnn_model_ffp(user_rnn_layers *layers);
void user_rnn_model_bp(user_rnn_layers *layers, float alpha);
float user_rnn_model_return_loss(user_rnn_layers *layers);
user_nn_list_matrix *user_rnn_model_return_result(user_rnn_layers *layers);

#endif