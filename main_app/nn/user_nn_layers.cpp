
#include "user_nn_layers.h"

//����ָ����
user_nn_layers *user_nn_layers_get(user_nn_layers *dest, int index) {
	while (index--) {
		if (dest->next != NULL) {
			dest = dest->next;
		}
		else {
		}
	}
	return dest;
}

//����һ����
//������
//type��������
//index��ָ��
//���� ������Ĳ�
user_nn_layers *user_nn_layers_create(user_nn_layer_type type, int index) {
	user_nn_layers *nn_layers = NULL;

	nn_layers = (user_nn_layers *)malloc(sizeof(user_nn_layers));//�����ڴ�
	nn_layers->prior = NULL;//ָ����һ��
	nn_layers->type = type;//�������
	nn_layers->index = index;//ָ��
	nn_layers->content = NULL;//ָ������
	nn_layers->next = NULL;//ָ����һ��

	return nn_layers;
}
//ɾ����
void user_nn_layers_delete(user_nn_layers *layers) {
	if (layers != NULL) {
		if (layers->content != NULL) {
			free(layers->content);
		}
		free(layers);
	}
}
//���������
//����
//feature_width���������ݵĿ��
//feature_height���������ݵĸ߶�
//feature_number���������ݵ�����
//���أ��ɹ���ʧ��
user_nn_input_layers *user_nn_layers_input_create(user_nn_layers *nn_layers, int feature_width, int feature_height) {
	user_nn_layers			*last_layers =nn_layers;
	user_nn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_nn_layers_create(u_nn_layer_type_input, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_nn_input_layers));//�����ڴ������Ķ���ռ�
	input_layers = (user_nn_input_layers *)last_layers->next->content;//ת����ǰ���ֵ �������ò���

	input_layers->feature_width = feature_width;//�����������ݵĿ��
	input_layers->feature_height = feature_height;//�����������ݵĸ߶�
	input_layers->deltas_matrix = user_nn_matrix_create(input_layers->feature_width, input_layers->feature_height);//��һ�㷴�������Ĳв�
	input_layers->feature_matrix = user_nn_matrix_create(input_layers->feature_width, input_layers->feature_height);//����������������ݾ��� 

	return input_layers;
}
//�������ز�
//����
//width���������ݵĿ��
//height���������ݵĸ߶�
//���� �ɹ���ʧ��
user_nn_hidden_layers *user_nn_layers_hidden_create(user_nn_layers *nn_layers,int feature_number) {
	user_nn_layers			*last_layers =nn_layers;
	user_nn_hidden_layers	*hidden_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_nn_layers_create(u_nn_layer_type_hidden, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_nn_hidden_layers));//����ռ�
	hidden_layers = (user_nn_hidden_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ

	if (last_layers->type == u_nn_layer_type_input) {
		user_nn_input_layers	*temp_layers = (user_nn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_nn_layer_type_hidden) {
		user_nn_hidden_layers	*temp_layers = (user_nn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->feature_width  = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//��̬����Ȩ�ؾ����С
	hidden_layers->biases_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//���ƫ�ò���

	hidden_layers->deltas_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);
	hidden_layers->deltas_biases_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);

	hidden_layers->deltas_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//��һ�㷴�������Ĳв�
	hidden_layers->feature_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//��������

	user_nn_matrix_init_vaule(hidden_layers->kernel_matrix, intput_featrue_width*intput_feature_height, hidden_layers->feature_width*hidden_layers->feature_height);//��ʼ��ȫ���ӵ�Ȩ��ֵ
	user_nn_matrix_init_vaule(hidden_layers->biases_matrix, intput_featrue_width*intput_feature_height, hidden_layers->feature_width*hidden_layers->feature_height);//��ʼ��ȫ���ӵ�Ȩ��ֵ

	return hidden_layers;
}
//���������
//����
//count���������
//���� �ɹ���ʧ��
user_nn_output_layers *user_nn_layers_output_create(user_nn_layers *nn_layers, int feature_number) {
	user_nn_layers			*last_layers =nn_layers;
	user_nn_output_layers	*output_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_nn_layers_create(u_nn_layer_type_output, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_nn_output_layers));//���������ڴ��������ռ�
	output_layers = (user_nn_output_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ
	 //�����ǵ�ǰ��Ĳ���
	if (last_layers->type == u_nn_layer_type_input) {
		user_nn_input_layers	*temp_layers = (user_nn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_nn_layer_type_hidden) {
		user_nn_hidden_layers	*temp_layers = (user_nn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;

	output_layers->loss_function = 0.0f;
	output_layers->kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//ȫ���Ӳ��Ȩ��ֵ
	output_layers->biases_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//���N��ƫ�ò��� ����ʹ��softmat�ع��ƫ�ò���
	
	output_layers->feature_matrix		= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//
	output_layers->target_matrix		= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//
	output_layers->error_matrix			= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//��Ӵ���ֵ

	output_layers->deltas_matrix		= user_nn_matrix_create( output_layers->feature_width, output_layers->feature_height);//����в�
	output_layers->deltas_kernel_matrix	= user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//����в���ϲ�Ľ����W
	output_layers->deltas_biases_matrix	= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//

	user_nn_matrix_init_vaule(output_layers->kernel_matrix, intput_featrue_width*intput_feature_height, output_layers->feature_width*output_layers->feature_height);//��ʼ��ȫ���ӵ�Ȩ��ֵ
	user_nn_matrix_init_vaule(output_layers->biases_matrix, intput_featrue_width*intput_feature_height, output_layers->feature_width*output_layers->feature_height);//��ʼ��ȫ���ӵ�Ȩ��ֵ

	return output_layers;
}

