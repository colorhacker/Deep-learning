

#ifndef _user_rnn_layers_H
#define _user_rnn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _rnn_layer_type {
	u_rnn_layer_type_null = 0,
	u_rnn_layer_type_input,
	u_rnn_layer_type_hidden,
	u_rnn_layer_type_output
}user_rnn_layer_type;

typedef struct _rnn_layers {
	struct _rnn_layers *prior;//��һ��ceng
	int index;//ָ��
	user_rnn_layer_type type;//����
	void *content;//����
	struct _rnn_layers *next;//��һ��
}user_rnn_layers;

typedef struct _rnn_input_layers {
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	int time_number;//���������������
	user_nn_list_matrix	*deltas_matrices;//���汾������ݲв�
	user_nn_list_matrix *feature_matrices;//����������ݾ���
}user_rnn_input_layers;

typedef struct _rnn_hidden_layers {
	int input_feature_number;//�������ݸ���
	int time_number;//������ݸ���
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�

	user_nn_matrix		*kernel_matrix;//��ű����������������ز������	
	user_nn_matrix		*kernel_matrix_t;//����ϸ�ʱ��ڵ����ز����ݵ���ʱʱ��ڵ���񾭺�
	user_nn_matrix		*biases_matrix;//ƫ�ò���
	user_nn_matrix		*feature_matrix_t;//time-1ʱ������ز�����
	user_nn_list_matrix *feature_matrices;//��ż������������� ����ʱ����������ж��

	user_nn_list_matrix	*deltas_matrices;//���汾������ݲв�
	user_nn_matrix		*deltas_kernel_matrix;//����в��ǰһ��feture maps�Ľ��Ҳ���Ǧ�W��ֵ	
	user_nn_matrix		*deltas_matrix_t;//���汾������ݲв�
	user_nn_matrix		*deltas_kernel_matrix_t;//����в��ǰһ��feture maps�Ľ��Ҳ���Ǧ�W��ֵ	
	user_nn_matrix		*deltas_biases_matrix;//����bias����
}user_rnn_hidden_layers;//

//�����
typedef struct _rnn_output_layers {
	int input_feature_number;//�������ݸ���
	int time_number;//������ݸ���
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	float loss_function;//���ۺ���

	user_nn_matrix		*kernel_matrix;//������ز�� �ں˹���
	user_nn_matrix		*biases_matrix;//ƫ�ò���  ƫ�ò�������
	user_nn_list_matrix *feature_matrices;//��ż������������� ����ʱ����������ж��
	user_nn_list_matrix *target_matrices;//Ŀ�����
	user_nn_list_matrix	*deltas_matrices;//���汾������ݲв�
	user_nn_matrix		*deltas_kernel_matrix;//����в��ǰһ��feture maps�Ľ��Ҳ���Ǧ�W��ֵ
	user_nn_matrix		*deltas_biases_matrix;//���Ԧ�bias����
	user_nn_matrix		*error_matrix;//��������ֵ���󱣴�	
}user_rnn_output_layers;//�����

user_rnn_layers *user_rnn_layers_get(user_rnn_layers *dest, int index);
user_rnn_layers *user_rnn_layers_create(user_rnn_layer_type type, int index);
void user_rnn_layers_delete(user_rnn_layers *layers);

user_rnn_input_layers *user_rnn_layers_input_create(user_rnn_layers *rnn_layers, int feature_width, int feature_height, int time_number);
user_rnn_hidden_layers *user_rnn_layers_hidden_create(user_rnn_layers *rnn_layers, int feature_number, int time_number);
user_rnn_output_layers *user_rnn_layers_output_create(user_rnn_layers *rnn_layers, int feature_number, int time_number);


#endif