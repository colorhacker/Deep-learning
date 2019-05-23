#ifndef _user_snn_model_H
#define _user_snn_model_H

#include "../user_config.h"
#include "user_snn_layers.h"
#include "user_snn_ffp_bp.h"

user_snn_layers *user_snn_model_create(int *layer_infor);//创建nn模型
user_snn_layers *user_snn_model_return_layer(user_snn_layers *layers, user_snn_layer_type type);
void user_snn_model_info_layer(user_snn_layers *layers);//显示层信息
void user_snn_model_load_input_feature(user_snn_layers *layers, user_nn_matrix *src_matrix);
void user_snn_model_load_target_feature(user_snn_layers *layers, user_nn_matrix *src_matrix);
void user_snn_model_ffp(user_snn_layers *layers);
void user_snn_model_bp(user_snn_layers *layers, float alpha);
float user_snn_model_return_loss(user_snn_layers *layers);
user_nn_matrix *user_snn_model_return_result(user_snn_layers *layers);
void user_snn_model_display_feature(user_snn_layers *layers);

#endif