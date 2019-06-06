

#ifndef _user_snn_layers_H
#define _user_snn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _snn_layer_type {
	u_snn_layer_type_null = 0,
	u_snn_layer_type_input,
	u_snn_layer_type_flat,
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

typedef struct _snn_flat_layers {
	int feature_width;//���ݿ�� �������ݵĿ��
	int feature_height;//���ݸ߶� �������ݵĸ߶�

	user_nn_matrix		*min_kernel_matrix;//��Ԫw	
	user_nn_matrix		*max_kernel_matrix;//ƫ�ò���
	user_nn_matrix		*feature_matrix;//��ż������������� 
	user_nn_matrix		*thred_matrix;//���汾��ı仯����
}user_snn_flat_layers;//

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
	float loss_function;//��ʧ����

	user_nn_matrix		*min_kernel_matrix;//��Ԫw	
	user_nn_matrix		*max_kernel_matrix;//ƫ�ò���
	user_nn_matrix		*feature_matrix;//��ż������������� 
	user_nn_matrix		*target_matrix;//Ŀ�����
	user_nn_matrix		*thred_matrix;//���汾��ı仯����
}user_snn_output_layers;//�����



void user_snn_data_softmax(user_nn_matrix *src_matrix);
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix * max_matrix);
void user_nn_matrix_thred_process(user_nn_matrix *thred_matrix, user_nn_matrix *src_matrix, user_nn_matrix *target_matrix);
void user_nn_matrix_thred_flat(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *output_matrix);
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *output_matrix);//������ֵ�ۼ�
void user_nn_matrix_update_flat(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value);
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value);//������ֵ


user_snn_layers *user_snn_layers_get(user_snn_layers *dest, int index);
user_snn_layers *user_snn_layers_create(user_snn_layer_type type, int index);
void user_snn_layers_delete(user_snn_layers *layers);
void user_snn_layers_all_delete(user_snn_layers *layers);
user_snn_input_layers *user_snn_layers_input_create(user_snn_layers *nn_layers, int feature_width, int feature_height);
user_snn_flat_layers *user_snn_layers_flat_create(user_snn_layers *snn_layers);
user_snn_hidden_layers *user_snn_layers_hidden_create(user_snn_layers *snn_layers, int feature_number);
user_snn_output_layers *user_snn_layers_output_create(user_snn_layers *nn_layers, int feature_number);


#endif

/*
//float src[] = { -0.5f };
//float min[] = { -0.5f,0.1f };
//float max[] = { 1.6f,1.0f };

//user_nn_matrix *src_matrix = user_nn_matrix_create_memset(1, 1, src);

//user_nn_matrix *min_matrix = user_nn_matrix_create_memset(1, 2, min);
//user_nn_matrix *max_matrix = user_nn_matrix_create_memset(1, 2, max);

//user_nn_matrix *res_matrix = user_nn_matrix_create(1, 2);

//user_nn_matrix_thred_acc(src_matrix, min_matrix, max_matrix, res_matrix);//

//if (res_matrix != NULL) {
//	user_nn_matrix_printf(NULL, res_matrix);//��ӡ����
//}
//else {
//	printf("null\n");
//}
//printf("\nend");
//system("pause");
//return;
//float data[] = { -1.0f,0.9f};
//user_nn_matrix *matrix = user_nn_matrix_create_memset(1, sizeof(data)/ sizeof(float), data);
//user_snn_data_softmax(matrix);
//user_snn_data_softmax(matrix);
//user_snn_data_softmax(matrix);
//user_nn_matrix_printf(NULL, matrix);
//system("pause");
//return;
srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
int layers[] = {
'i', 1, 2,
//'f',
'h', 100,
//'f',
'o', 2
};
int io = 0;
user_snn_layers *layer = user_snn_model_create(layers);//����ģ��

float input[][2] = { { -0.1f,0.1f },{ 0.15f,-0.15f } ,{ -0.2f,0.2f },{ 0.3f,-0.3f } };
float output[][2] = { { -1.0f,1.0f },{ 1.5f,-1.5f } ,{ -2.0f,2.0f },{ -3.0f,3.0f } };
user_nn_matrix *input1 = user_nn_matrix_create_memset(1, 2, input[0]);
user_nn_matrix *input2 = user_nn_matrix_create_memset(1, 2, input[1]);
user_nn_matrix *input3 = user_nn_matrix_create_memset(1, 2, input[2]);
user_nn_matrix *input4 = user_nn_matrix_create_memset(1, 2, input[3]);
user_nn_matrix *output1 = user_nn_matrix_create_memset(1, 2, output[0]);
user_nn_matrix *output2 = user_nn_matrix_create_memset(1, 2, output[1]);
user_nn_matrix *output3 = user_nn_matrix_create_memset(1, 2, output[2]);
user_nn_matrix *output4 = user_nn_matrix_create_memset(1, 2, output[3]);

//user_snn_data_softmax(output4);//��������
//user_nn_matrix_printf(NULL, output4);
for (int count = 0; count < 500000; count++) {
user_snn_model_load_input_feature(layer, input1);//������������
user_snn_model_load_target_feature(layer, output1);//����Ŀ������
user_snn_model_ffp(layer);
user_snn_model_bp(layer);
user_snn_model_load_input_feature(layer, input2);//������������
user_snn_model_load_target_feature(layer, output2);//����Ŀ������
user_snn_model_ffp(layer);
user_snn_model_bp(layer);
user_snn_model_load_input_feature(layer, input3);//������������
user_snn_model_load_target_feature(layer, output3);//����Ŀ������
user_snn_model_ffp(layer);
user_snn_model_bp(layer);
user_snn_model_load_input_feature(layer, input4);//������������
user_snn_model_load_target_feature(layer, output4);//����Ŀ������
user_snn_model_ffp(layer);
user_snn_model_bp(layer);

if (io++ >= 1000) {
io = 0;
printf("\n--->:%f",  user_snn_model_return_loss(layer));
}
//user_snn_model_display_feature(snn_layers);

}


user_snn_model_load_input_feature(layer, input1);//������������
user_snn_model_load_target_feature(layer, output1);//����Ŀ������
user_snn_model_ffp(layer);
user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));
user_snn_model_load_input_feature(layer, input2);//������������
user_snn_model_load_target_feature(layer, output2);//����Ŀ������
user_snn_model_ffp(layer);
user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));
user_snn_model_load_input_feature(layer, input3);//������������
user_snn_model_load_target_feature(layer, output3);//����Ŀ������
user_snn_model_ffp(layer);
user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));
user_snn_model_load_input_feature(layer, input4);//������������
user_snn_model_load_target_feature(layer, output4);//����Ŀ������
user_snn_model_ffp(layer);
user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));

user_nn_matrix_printf(NULL, (((user_snn_output_layers *)user_snn_model_return_layer(layer, u_snn_layer_type_output)->content)->min_kernel_matrix));
user_nn_matrix_printf(NULL, (((user_snn_output_layers *)user_snn_model_return_layer(layer, u_snn_layer_type_output)->content)->max_kernel_matrix));
user_nn_matrix_delete(input1);
user_nn_matrix_delete(input2);
user_nn_matrix_delete(input3);
user_nn_matrix_delete(input4);
user_nn_matrix_delete(output1);
user_nn_matrix_delete(output2);
user_nn_matrix_delete(output3);
user_nn_matrix_delete(output4);
system("pause");
return;
*/