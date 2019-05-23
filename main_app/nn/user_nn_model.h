#ifndef _user_nn_model_H
#define _user_nn_model_H

#include "../user_config.h"
#include "user_nn_layers.h"
#include "user_nn_ffp.h"
#include "user_nn_bp.h"
#include "user_nn_grads.h"

user_nn_layers *user_nn_model_create(int *layer_infor);//创建nn模型
user_nn_layers *user_nn_model_return_layer(user_nn_layers *layers, user_nn_layer_type type);
void user_nn_model_info_layer(user_nn_layers *layers);//显示层信息
void user_nn_model_load_input_feature(user_nn_layers *layers, user_nn_matrix *src_matrix);
void user_nn_model_load_target_feature(user_nn_layers *layers, user_nn_matrix *src_matrix);
void user_nn_model_ffp(user_nn_layers *layers);
void user_nn_model_bp(user_nn_layers *layers, float alpha);
float user_nn_model_return_loss(user_nn_layers *layers);
user_nn_matrix *user_nn_model_return_result(user_nn_layers *layers);
void user_nn_model_display_feature(user_nn_layers *layers);

#endif