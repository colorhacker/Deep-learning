
#include "user_rnn_layers.h"

//����ָ����
user_rnn_layers *user_rnn_layers_get(user_rnn_layers *dest, int index) {
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
user_rnn_layers *user_rnn_layers_create(user_rnn_layer_type type, int index) {
	user_rnn_layers *rnn_layers = NULL;

	rnn_layers = (user_rnn_layers *)malloc(sizeof(user_rnn_layers));//�����ڴ�
	rnn_layers->prior = NULL;//ָ����һ��
	rnn_layers->type = type;//�������
	rnn_layers->index = index;//ָ��
	rnn_layers->content = NULL;//ָ������
	rnn_layers->next = NULL;//ָ����һ��

	return rnn_layers;
}
//ɾ����
void user_rnn_layers_delete(user_rnn_layers *layers) {
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
user_rnn_input_layers *user_rnn_layers_input_create(user_rnn_layers *rnn_layers, int feature_width, int feature_height, int time_number) {
	user_rnn_layers			*last_layers = rnn_layers;
	user_rnn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����rnn_layers�ն���
	}
	last_layers->next = user_rnn_layers_create(u_rnn_layer_type_input, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->content = malloc(sizeof(user_rnn_input_layers));//�����ڴ������Ķ���ռ�
	input_layers = (user_rnn_input_layers *)last_layers->next->content;//ת����ǰ���ֵ �������ò���

	input_layers->feature_width = feature_width;//�����������ݵĿ��
	input_layers->feature_height = feature_height;//�����������ݵĸ߶�
	input_layers->time_number = time_number;//�����������ݵĸ���
	input_layers->deltas_matrices = user_nn_matrices_create(1, input_layers->time_number, input_layers->feature_width, input_layers->feature_height);//��һ�㷴�������Ĳв�
	input_layers->feature_matrices = user_nn_matrices_create(1, input_layers->time_number, input_layers->feature_width, input_layers->feature_height);//����������������ݾ��� 

	return input_layers;
}
//�������ز�
//����
//width���������ݵĿ��
//height���������ݵĸ߶�
//���� �ɹ���ʧ��
user_rnn_hidden_layers *user_rnn_layers_hidden_create(user_rnn_layers *rnn_layers,int feature_number, int time_number) {
	user_rnn_layers			*last_layers = rnn_layers;
	user_rnn_hidden_layers	*hidden_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����rnn_layers�ն���
	}
	last_layers->next = user_rnn_layers_create(u_rnn_layer_type_hidden, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_rnn_hidden_layers));//����ռ�
	hidden_layers = (user_rnn_hidden_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ

	if (last_layers->type == u_rnn_layer_type_input) {
		user_rnn_input_layers	*temp_layers = (user_rnn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_rnn_layer_type_hidden) {
		user_rnn_hidden_layers	*temp_layers = (user_rnn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->time_number = time_number;//����ʱ��Ƭ
	hidden_layers->feature_width  = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//��̬����Ȩ�ؾ����С
	hidden_layers->kernel_matrix_t = user_nn_matrix_create(hidden_layers->feature_height, hidden_layers->feature_height);//����ʱ�������㵽�������Ȩ��
	hidden_layers->biases_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//���ƫ�ò���

	hidden_layers->deltas_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);
	hidden_layers->deltas_kernel_matrix_t = user_nn_matrix_create(hidden_layers->feature_height, hidden_layers->feature_height);
	hidden_layers->deltas_biases_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);

	hidden_layers->deltas_matrix_t = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);
	hidden_layers->deltas_matrices = user_nn_matrices_create(1, hidden_layers->time_number, hidden_layers->feature_width, hidden_layers->feature_height);//��һ�㷴�������Ĳв�
	hidden_layers->feature_matrix_t = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//�����ϸ�ʱ��Ƭ������������
	hidden_layers->feature_matrices = user_nn_matrices_create(1, hidden_layers->time_number, hidden_layers->feature_width, hidden_layers->feature_height);//��������

	user_nn_matrix_init_vaule(hidden_layers->kernel_matrix, hidden_layers->time_number, hidden_layers->time_number);//��ʼ��ȫ���ӵ�Ȩ��ֵ
	user_nn_matrix_init_vaule(hidden_layers->kernel_matrix_t, hidden_layers->time_number, hidden_layers->time_number);//��ʼ��ȫ���ӵ�Ȩ��ֵ
	user_nn_matrix_init_vaule(hidden_layers->biases_matrix, hidden_layers->time_number, hidden_layers->time_number);//��ʼ��ȫ���ӵ�Ȩ��ֵ

	return hidden_layers;
}
//���������
//����
//count���������
//���� �ɹ���ʧ��
user_rnn_output_layers *user_rnn_layers_output_create(user_rnn_layers *rnn_layers, int feature_number, int time_number) {
	user_rnn_layers			*last_layers = rnn_layers;
	user_rnn_output_layers	*output_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����rnn_layers�ն���
	}
	last_layers->next = user_rnn_layers_create(u_rnn_layer_type_output, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_rnn_output_layers));//���������ڴ��������ռ�
	output_layers = (user_rnn_output_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ
	 //�����ǵ�ǰ��Ĳ���
	if (last_layers->type == u_rnn_layer_type_input) {
		user_rnn_input_layers	*temp_layers = (user_rnn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_rnn_layer_type_hidden) {
		user_rnn_hidden_layers	*temp_layers = (user_rnn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->time_number = time_number;//���õ�ǰ����������ݸ���
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;

	output_layers->loss_function = 0.0f;
	output_layers->kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//ȫ���Ӳ��Ȩ��ֵ
	output_layers->biases_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//���N��ƫ�ò��� ����ʹ��softmat�ع��ƫ�ò���
	
	output_layers->feature_matrices		= user_nn_matrices_create(1, output_layers->time_number, output_layers->feature_width, output_layers->feature_height);//
	output_layers->target_matrices		= user_nn_matrices_create(1, output_layers->time_number, output_layers->feature_width, output_layers->feature_height);//
	output_layers->error_matrix	= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//��Ӵ���ֵ

	output_layers->deltas_matrices		= user_nn_matrices_create(1, output_layers->time_number, output_layers->feature_width, output_layers->feature_height);//����в�
	output_layers->deltas_kernel_matrix	= user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//����в���ϲ�Ľ����W
	output_layers->deltas_biases_matrix	= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//

	user_nn_matrix_init_vaule(output_layers->kernel_matrix, output_layers->time_number, output_layers->time_number);//��ʼ��ȫ���ӵ�Ȩ��ֵ
	user_nn_matrix_init_vaule(output_layers->biases_matrix, output_layers->time_number, output_layers->time_number);//��ʼ��ȫ���ӵ�Ȩ��ֵ

	return output_layers;
}

