#ifndef _user_cnn_model_H
#define _user_cnn_model_H

#include "../user_config.h"

char *user_cnn_model_get_exe_path(void);

user_nn_matrix *user_cnn_model_matrices_splice(user_nn_list_matrix *src_matrix);//ƴ��ͼ����������Ϊ��������
void user_cnn_model_display_matrix(char *window_name, user_nn_matrix  *src_matrix);//��ʾ��������
void user_cnn_model_display_matrices(char *window_name, user_nn_list_matrix  *src_matrices, int gain);//��ʾ�����ľ���
void user_cnn_model_display_feature(user_cnn_layers *layers);//ʹ��opencv��ʾ��������

user_cnn_layers *user_cnn_model_create(int *layer_infor);//����һ��ģ��
void user_cnn_model_load_input_feature(user_cnn_layers *layers, user_nn_matrix *src_matrix, int index);//����������������
void user_cnn_model_load_input_image(user_cnn_layers *layers, char *path, int index);//����ͼ������
void user_cnn_model_load_target_feature(user_cnn_layers *layers, user_nn_matrix *src_matrix);//����Ŀ�����
void user_cnn_model_ffp(user_cnn_layers *layers);//�������һ��
void user_cnn_model_bp(user_cnn_layers *layers,float alpha);//�������һ��

int user_cnn_model_return_class(user_cnn_layers *layers);//��ȡʶ�����
float user_cnn_model_return_loss(user_cnn_layers *layers);//��ȡ���ۺ���
user_cnn_layers *user_cnn_model_return_layer(user_cnn_layers *layers, user_cnn_layer_type type);
void user_cnn_model_info_layer(user_cnn_layers *layers);

#endif