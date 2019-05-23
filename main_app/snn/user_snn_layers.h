

#ifndef _user_snn_layers_H
#define _user_snn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _snn_layer_type {
	u_snn_layer_type_null = 0,
	u_snn_layer_type_input,
	u_snn_layer_type_hidden,
	u_snn_layer_type_output
}user_snn_layer_type;

typedef struct _snn_layers {
	struct _snn_layers *prior;//��һ��ceng
	int index;//ָ��
	user_snn_layer_type type;//����
	void *content;//����
	struct _snn_layers *next;//��һ��
}user_snn_layers;

typedef struct _snn_input_layers {
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�
	user_nn_matrix	*feature_matrix;//����������ݾ���
	user_nn_matrix	*thred_matrix;//���汾��ı仯����
}user_snn_input_layers;

typedef struct _snn_hidden_layers {
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�

	user_nn_matrix		*min_kernel_matrix;//��Ԫw	
	user_nn_matrix		*max_kernel_matrix;//ƫ�ò���
	user_nn_matrix		*feature_matrix;//��ż������������� 
	user_nn_matrix		*thred_matrix;//���汾��ı仯����
}user_snn_hidden_layers;//

					   //�����
typedef struct _snn_output_layers {
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�

	user_nn_matrix		*min_kernel_matrix;//��Ԫw	
	user_nn_matrix		*max_kernel_matrix;//ƫ�ò���
	user_nn_matrix		*feature_matrix;//��ż������������� 
	user_nn_matrix		*target_matrix;//Ŀ�����
	user_nn_matrix		*thred_matrix;//���汾��ı仯����
}user_snn_output_layers;//�����

#define snn_avg_vaule		1.0f
#define snn_step_vaule		0.001f

#define snn_thred_none		1.0f	//����Ҫ�仯
#define snn_thred_add		0.9f	//Ŀ��ֵ�������ֵ
#define snn_thred_acc		1.1f	//Ŀ��ֵС�����ֵ

void user_snn_data_softmax(user_nn_matrix *src_matrix);
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix * max_matrix);
void user_nn_matrix_thred_process(user_nn_matrix *thred_matrix, user_nn_matrix *src_matrix, user_nn_matrix *target_matrix);
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *output_matrix);//������ֵ�ۼ�
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value);//������ֵ


user_snn_layers *user_snn_layers_create(user_snn_layer_type type, int index);
user_snn_input_layers *user_snn_layers_input_create(user_snn_layers *nn_layers, int feature_width, int feature_height);
user_snn_hidden_layers *user_snn_layers_hidden_create(user_snn_layers *snn_layers, int feature_number);
user_snn_output_layers *user_snn_layers_output_create(user_snn_layers *nn_layers, int feature_number);


#endif