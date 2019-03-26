

#ifndef _user_cnn_layers_H
#define _user_cnn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _cnn_layer_type{
	u_cnn_layer_type_null = 0,
	u_cnn_layer_type_input,
	u_cnn_layer_type_conv,
	u_cnn_layer_type_pool,
	u_cnn_layer_type_full,
	u_cnn_layer_type_output
}user_cnn_layer_type;

typedef struct _cnn_layers{
	struct _cnn_layers *prior;//��һ����
	int index;//ָ��
	user_cnn_layer_type type;//����
	void *content;//����
	struct _cnn_layers *next;//��һ��
}user_cnn_layers;

typedef struct _cnn_input_layers{
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	int feature_number;//���������������
	user_nn_list_matrix *feature_matrices;//����������ݾ���
}user_cnn_input_layers;
typedef struct _cnn_conv_layers{
	int input_feature_number;//�������ݸ���
	int feature_number;//������ݸ���
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	int kernel_width;//����˵Ŀ��
	int kernel_height;//����˵Ŀ��

	user_nn_matrix		*biases_matrix;//ƫ�ò���
	user_nn_list_matrix *feature_matrices;//��ž�������������
	user_nn_list_matrix *kernel_matrices;//��ž����
	user_nn_list_matrix *deltas_matrices;//�в��
	user_nn_list_matrix *deltas_kernel_matrices;//����в��ǰһ��feture maps�ľ�����Ҳ���Ǧ�W��ֵ
	user_nn_matrix		*deltas_biases_matrix;//����bias����
}user_cnn_conv_layers;//�����
typedef struct _cnn_pool_layers{
	int input_feature_number;//�������ݸ���
	int feature_number;//������ݸ���
	int pool_width;//�ػ����
	int pool_height;//�ػ��߶�
	user_nn_pooling_type pool_type;//�ػ���ʽ
	int feature_width;//poolingģ�����ݿ��
	int feature_height;//poolingģ�����ݸ߶�
	user_nn_matrix *kernel_matrix;//pooling�������� ��ֵ�������
	user_nn_list_matrix *feature_matrices;//��ž�������������
	user_nn_list_matrix *deltas_matrices;//�в��
}user_cnn_pool_layers;//�ػ���
typedef struct _cnn_full_layers{
	int feature_number;//�����������ݸ���
	//int feature_width;//fullģ�����ݿ��
	//int feature_height;//fullģ�����ݸ߶�
	user_nn_matrix *input_feature_matrix;//������������ ���ϲ��������������һ������
	user_nn_matrix *kernel_matrix;//��������˻���
	user_nn_matrix *biases_matrix;//�����ƫ�ò���
	user_nn_matrix *feature_matrix;//�������������
	user_nn_matrix *deltas_matrix;//�в��
	user_nn_matrix *deltas_kernel_matrix;//����в��ǰһ��feture maps�ľ�����Ҳ���Ǧ�W��ֵ
}user_cnn_full_layers;//ȫ����
//�����
typedef struct _cnn_output_layers{
	int feature_number;//���������������
	int class_number;//�������---��ʱ����
	float loss_function;//���ۺ���

	user_nn_matrix *input_feature_matrix;//������������ ���ϲ��������������һ������
	user_nn_matrix *kernel_matrix;//��������˻���
	user_nn_matrix *biases_matrix;//�����ƫ�ò���
	user_nn_matrix *feature_matrix;//�������������
	user_nn_matrix *target_matrix;//Ŀ�������������
	user_nn_matrix *error_matrix;//��������ֵ���󱣴�
	user_nn_matrix *deltas_matrix;//�����в�
	user_nn_matrix *deltas_kernel_matrix;//����в��ǰһ��feture maps�ľ�����Ҳ���Ǧ�W��ֵ
}user_cnn_output_layers;//�����

user_cnn_layers			*user_cnn_layers_get(user_cnn_layers *dest, int index);
user_cnn_layers			*user_cnn_layers_create(user_cnn_layer_type type, int index);
void user_cnn_layers_delete(user_cnn_layers *layers);
user_cnn_input_layers	*user_cnn_layers_input_create(user_cnn_layers *cnn_layers, int feature_width, int feature_height, int feature_number);
user_cnn_conv_layers	*user_cnn_layers_convolution_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height ,int feature_number);
user_cnn_pool_layers	*user_cnn_layers_pooling_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height, user_nn_pooling_type pool_type);
user_cnn_full_layers	*user_cnn_layers_fullconnect_create(user_cnn_layers *cnn_layers);
user_cnn_output_layers	*user_cnn_layers_output_create(user_cnn_layers *cnn_layers, int class_number);

#endif