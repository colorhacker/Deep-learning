#ifndef _user_cnn_model_H
#define _user_cnn_model_H

#include "../user_config.h"

char *user_cnn_model_get_exe_path(void);

user_nn_matrix *user_cnn_model_matrices_splice(user_nn_list_matrix *src_matrix);//拼接图像连续矩阵为单个矩阵
void user_cnn_model_display_matrix(char *window_name, user_nn_matrix  *src_matrix);//显示矩阵数据
void user_cnn_model_display_matrices(char *window_name, user_nn_list_matrix  *src_matrices, int gain);//显示连续的矩阵
void user_cnn_model_display_feature(user_cnn_layers *layers);//使用opencv显示特征数据

user_cnn_layers *user_cnn_model_create(int *layer_infor);//创建一个模型
void user_cnn_model_load_input_feature(user_cnn_layers *layers, user_nn_matrix *src_matrix, int index);//加载输入特征数据
void user_cnn_model_load_input_image(user_cnn_layers *layers, char *path, int index);//加载图像数据
void user_cnn_model_load_target_feature(user_cnn_layers *layers, user_nn_matrix *src_matrix);//载入目标矩阵
void user_cnn_model_ffp(user_cnn_layers *layers);//正向计算一次
void user_cnn_model_bp(user_cnn_layers *layers,float alpha);//反向计算一次

int user_cnn_model_return_class(user_cnn_layers *layers);//获取识别的类
float user_cnn_model_return_loss(user_cnn_layers *layers);//获取代价函数
user_cnn_layers *user_cnn_model_return_layer(user_cnn_layers *layers, user_cnn_layer_type type);
void user_cnn_model_info_layer(user_cnn_layers *layers);

#endif