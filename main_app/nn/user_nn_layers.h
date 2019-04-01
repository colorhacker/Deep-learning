

#ifndef _user_nn_layers_H
#define _user_nn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _nn_layer_type {
	u_nn_layer_type_null = 0,
	u_nn_layer_type_input,
	u_nn_layer_type_hidden,
	u_nn_layer_type_output
}user_nn_layer_type;

typedef struct _nn_layers {
	struct _nn_layers *prior;//��һ��ceng
	int index;//ָ��
	user_nn_layer_type type;//����
	void *content;//����
	struct _nn_layers *next;//��һ��
}user_nn_layers;

typedef struct _nn_input_layers {
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	user_nn_matrix	*deltas_matrix;//���汾������ݲв�
	user_nn_matrix	*feature_matrix;//����������ݾ���
}user_nn_input_layers;

typedef struct _nn_hidden_layers {
	int input_feature_number;//�������ݸ���
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�

	user_nn_matrix		*kernel_matrix;//��Ԫw	
	user_nn_matrix		*biases_matrix;//ƫ�ò���
	user_nn_matrix		*feature_matrix;//��ż������������� 

	user_nn_matrix		*deltas_matrix;//���汾������ݲв�
	user_nn_matrix		*deltas_kernel_matrix;//����в��ǰһ��feture maps�Ľ��Ҳ���Ǧ�W��ֵ	
	user_nn_matrix		*deltas_biases_matrix;//����bias����
}user_nn_hidden_layers;//

//�����
typedef struct _nn_output_layers {
	int input_feature_number;//�������ݸ���
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	float loss_function;//���ۺ���

	user_nn_matrix		*kernel_matrix;//������ز�� �ں˹���
	user_nn_matrix		*biases_matrix;//ƫ�ò���  ƫ�ò�������
	user_nn_matrix		*feature_matrix;//��ż������������� ����ʱ����������ж��
	user_nn_matrix		*target_matrix;//Ŀ�����
	user_nn_matrix		*deltas_matrix;//���汾������ݲв�
	user_nn_matrix		*deltas_kernel_matrix;//����в��ǰһ��feture maps�Ľ��Ҳ���Ǧ�W��ֵ
	user_nn_matrix		*deltas_biases_matrix;//���Ԧ�bias����
	user_nn_matrix		*error_matrix;//��������ֵ���󱣴�	
}user_nn_output_layers;//�����

user_nn_layers *user_nn_layers_get(user_nn_layers *dest, int index);
user_nn_layers *user_nn_layers_create(user_nn_layer_type type, int index);
void user_nn_layers_delete(user_nn_layers *layers);

user_nn_input_layers *user_nn_layers_input_create(user_nn_layers *nn_layers, int feature_width, int feature_height);
user_nn_hidden_layers *user_nn_layers_hidden_create(user_nn_layers *nn_layers, int feature_number);
user_nn_output_layers *user_nn_layers_output_create(user_nn_layers *nn_layers, int feature_number);


#endif