
#include "user_snn_layers.h"

//����ָ����
user_snn_layers *user_snn_layers_get(user_snn_layers *dest, int index) {
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
user_snn_layers *user_snn_layers_create(user_snn_layer_type type, int index) {
	user_snn_layers *snn_layers = NULL;

	snn_layers = (user_snn_layers *)malloc(sizeof(user_snn_layers));//�����ڴ�
	snn_layers->prior = NULL;//ָ����һ��
	snn_layers->type = type;//�������
	snn_layers->index = index;//ָ��
	snn_layers->content = NULL;//ָ������
	snn_layers->next = NULL;//ָ����һ��

	return snn_layers;
}
//ɾ����
void user_snn_layers_delete(user_snn_layers *layers) {
	if (layers != NULL) {
		if (layers->content != NULL) {
			if (layers->type == u_snn_layer_type_input) {
				user_nn_matrix_delete(((user_snn_input_layers *)layers->content)->thred_matrix);
				user_nn_matrix_delete(((user_snn_input_layers *)layers->content)->feature_matrix);
			}
			else if (layers->type == u_snn_layer_type_flat) {
				user_nn_matrix_delete(((user_snn_flat_layers *)layers->content)->thred_kernel_matrix);
				user_nn_matrix_delete(((user_snn_flat_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_flat_layers *)layers->content)->thred_matrix);
			}
			else if (layers->type == u_snn_layer_type_hidden) {
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_kernel_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_matrix);
			}
			else if (layers->type == u_snn_layer_type_output) {
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_kernel_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_matrix);
			}
			free(layers->content);
		}
		free(layers);
	}
}
//ɾ��������
void user_snn_layers_all_delete(user_snn_layers *layers) {
	user_snn_layers *layer = layers;
	user_snn_layers *layer_next = NULL;
	while (layer != NULL) {
		layer_next = layer->next;
		user_snn_layers_delete(layer);//ɾ����ǰ����
		layer = layer_next;//���¾���
	}
}
//���������
//����
//feature_width���������ݵĿ��
//feature_height���������ݵĸ߶�
//feature_number���������ݵ�����
//���أ��ɹ���ʧ��
user_snn_input_layers *user_snn_layers_input_create(user_snn_layers *nn_layers, int feature_width, int feature_height) {
	user_snn_layers			*last_layers = nn_layers;
	user_snn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_input, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_snn_input_layers));//�����ڴ������Ķ���ռ�
	input_layers = (user_snn_input_layers *)last_layers->next->content;//ת����ǰ���ֵ �������ò���

	input_layers->feature_width = feature_width;//�����������ݵĿ��
	input_layers->feature_height = feature_height;//�����������ݵĸ߶�
	input_layers->feature_matrix = user_nn_matrix_create(input_layers->feature_width, input_layers->feature_height);//����������������ݾ���
	input_layers->thred_matrix = user_nn_matrix_create(input_layers->feature_width, input_layers->feature_height);//����������������ݾ���
	//input_layers->feature_matrix = NULL;//Ԥ��ָ�����ݵ�ַ

	return input_layers;
}
//����ƽ��
//����
//width���������ݵĿ��
//height���������ݵĸ߶�
//���� �ɹ���ʧ��
user_snn_flat_layers *user_snn_layers_flat_create(user_snn_layers *snn_layers) {
	user_snn_layers			*last_layers = snn_layers;
	user_snn_flat_layers	*flat_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_flat, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_snn_flat_layers));//����ռ�
	flat_layers = (user_snn_flat_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ

	if (last_layers->type == u_snn_layer_type_input) {
		user_snn_input_layers	*temp_layers = (user_snn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_flat) {
		user_snn_flat_layers	*temp_layers = (user_snn_flat_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	flat_layers->feature_width = intput_featrue_width;
	flat_layers->feature_height = intput_feature_height;

	flat_layers->thred_kernel_matrix = user_nn_matrix_create(flat_layers->feature_width, flat_layers->feature_height);//��Ԫ����

	flat_layers->feature_matrix = user_nn_matrix_create(flat_layers->feature_width, flat_layers->feature_height);//�����������
	flat_layers->thred_matrix = user_nn_matrix_create(flat_layers->feature_width, flat_layers->feature_height);//�����仯����

	user_snn_init_matrix(flat_layers->thred_kernel_matrix);//��ʼ������

	return flat_layers;
}
//�������ز�
//����
//width���������ݵĿ��
//height���������ݵĸ߶�
//���� �ɹ���ʧ��
user_snn_hidden_layers *user_snn_layers_hidden_create(user_snn_layers *snn_layers, int feature_number) {
	user_snn_layers			*last_layers = snn_layers;
	user_snn_hidden_layers	*hidden_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_hidden, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_snn_hidden_layers));//����ռ�
	hidden_layers = (user_snn_hidden_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ

	if (last_layers->type == u_snn_layer_type_input) {
		user_snn_input_layers	*temp_layers = (user_snn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_flat) {
		user_snn_flat_layers	*temp_layers = (user_snn_flat_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->feature_width = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->thred_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//��Ԫ����

	hidden_layers->feature_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//�����������
	hidden_layers->thred_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//�����仯����

	user_snn_init_matrix(hidden_layers->thred_kernel_matrix);//��ʼ������

	return hidden_layers;
}
//���������
//����
//count���������
//���� �ɹ���ʧ��
user_snn_output_layers *user_snn_layers_output_create(user_snn_layers *nn_layers, int feature_number) {
	user_snn_layers			*last_layers = nn_layers;
	user_snn_output_layers	*output_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//��ѯ����nn_layers�ն���
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_output, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_snn_output_layers));//���������ڴ��������ռ�
	output_layers = (user_snn_output_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ
																		//�����ǵ�ǰ��Ĳ���
	if (last_layers->type == u_snn_layer_type_input) {
		user_snn_input_layers	*temp_layers = (user_snn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_flat) {
		user_snn_flat_layers	*temp_layers = (user_snn_flat_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;

	output_layers->loss_function = 0.0f;
	output_layers->thred_kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//��Ԫ����

	output_layers->feature_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//�����������
	output_layers->thred_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//�����仯����
	output_layers->target_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//����Ŀ�����

	user_snn_init_matrix(output_layers->thred_kernel_matrix);//��ʼ������

	return output_layers;
}


//������ֵ����
void user_snn_data_softmax(user_nn_matrix *src_matrix) {
	float ave_value = user_nn_matrix_cum_element(src_matrix) / (float)(src_matrix->height * src_matrix->width);
	user_nn_matrix_sub_constant(src_matrix, ave_value);//ƽ��ֵ����Ϊ0.0f
}
//��ʼ����ֵ����
//������� ��С������
//��� ��
void user_snn_init_matrix(user_nn_matrix *thred_matrix) {
	int count = thred_matrix->height * thred_matrix->width;
	float *thred_data = thred_matrix->data;
	while (count--) {
		*thred_data++ = user_nn_init_uniform();
	}
}

//����ֵ����
//src_matrix �������
//target_matrix Ŀ�����
//���� �������
void user_nn_matrix_thred_process(user_nn_matrix *thred_matrix,user_nn_matrix *src_matrix, user_nn_matrix *target_matrix) {
	int count = src_matrix->width * src_matrix->height;
	float *src_data = src_matrix->data;
	float *target_data = target_matrix->data;
	float *thred_data = thred_matrix->data;
	while (count--) {
		if (*target_data > *src_data) {
			*thred_data = user_nn_snn_thred_add;
		}else if (*target_data < *src_data) {
			*thred_data = user_nn_snn_thred_acc;
		}else {
			*thred_data = user_nn_snn_thred_none;
		}
		src_data++;
		target_data++;
		thred_data++;
	}
}

//����ͨ����ֵ�������
//src_matrix �������
//min_matrix ����ֵ
//max_matrix ����ֵ
//��� ���ؽ������
void user_nn_matrix_thred_flat(user_nn_matrix *src_matrix, user_nn_matrix *thred_matrix, user_nn_matrix *output_matrix) {
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *output_data = output_matrix->data;

	for (int count = 0; count < src_matrix->height * src_matrix->width; count++) {
		if ((0.0f < *src_data) && (*src_data <= *thred_data)) {
			*output_data += 1.0f;
		}
		if ((*thred_data <= *src_data) && (*src_data < 0.0f)) {
			*output_data -= 1.0f;
		}
		src_data++;
		thred_data++;
		output_data++;
	}
}
//���������õ���ֵ�����ۼӼ��� �;���˷����� ֻ��ֵ�ǽ����ж�
//src_matrix �������
//min_matrix ����ֵ
//max_matrix ����ֵ
//��� ���ؽ������
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *thred_matrix,user_nn_matrix *output_matrix) {
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *output_data = output_matrix->data;
	if (thred_matrix->width != src_matrix->height) {//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return;
	}
	if ((output_matrix->width != src_matrix->width) || (output_matrix->height != thred_matrix->height)) {
		return;
	}

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for 
	for (int height = 0; height < output_matrix->height; height++) {
		for (int width = 0; width < output_matrix->width; width++) {
			for (int point = 0; point < src_matrix->height; point++) {
				if ((0.0f < src_data[width + point*src_matrix->width]) && (src_data[width + point*src_matrix->width] <= thred_data[height * thred_matrix->width + point])) {
					output_data[height*output_matrix->width + width] += 1.0f;
				}
				if ((thred_data[height * thred_matrix->width + point] <= src_data[width + point*src_matrix->width]) && (src_data[width + point*src_matrix->width] < 0.0f)) {
					output_data[height*output_matrix->width + width] -= 1.0f;
				}

			}
		}
	}
#else
	for (int height = 0; height < output_matrix->height; height++) {
		for (int width = 0; width < output_matrix->width; width++) {
			min_data = min_matrix->data + height * min_matrix->width;//ָ���п�ͷ
			max_data = max_matrix->data + height * max_matrix->width;//ָ���п�ͷ
			src_data = src_matrix->data + width;//ָ���п�ͷ
			for (int point = 0; point < src_matrix->height; point++) {
				if ((0.0f < *src_data) && (*src_data <= *thred_data)) {
					*output_data += 1.0f;
				}
				if ((*thred_data <= *src_data) && (*src_data < 0.0f)) {
					*output_data -= 1.0f;
				}
				src_data += src_matrix->width;
				thred_data++;
				output_data++;
				}
			output_data++;
			}
		}
#endif

}
//������ֵ���и���
//src_matrix �������
//min_matrix ����ֵ
//max_matrix ����ֵ
//��� ���ؽ������
void user_nn_matrix_update_flat(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *thred_matrix,user_nn_matrix *target_matrix ,float step_value){
	float *src_exp_data = src_exp_matrix->data;
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *target_data = target_matrix->data;//
	
	for (int count = 0; count < src_matrix->height * src_matrix->width; count++) {
		if (*target_data == user_nn_snn_thred_add) {//�������ֵ
			if ((0.0f < *thred_data) && (*thred_data < *src_data)) {
				*thred_data += step_value;
				*src_exp_data = *src_data <= *thred_data ? *src_exp_data : (*src_exp_data - user_nn_snn_add_value);
			}
			if ((*thred_data < *src_data) && (*src_data < 0.0f)) {
				*thred_data += step_value;
				*src_exp_data = *src_data >= *thred_data ? (*src_exp_data - user_nn_snn_add_value) : *src_exp_data;
			}
		}
		else if (*target_data == user_nn_snn_thred_acc) {//�������ֵ
			if ((0.0f < *src_data) && (*src_data < *thred_data)) {
				*thred_data -= step_value;
				*src_exp_data = *src_data <= *thred_data ? (*src_exp_data + user_nn_snn_add_value) : *src_exp_data;
			}
			if ((*src_data < *thred_data) && (*thred_data < 0.0f)) {
				*thred_data -= step_value;
				*src_exp_data = *src_data >= *thred_data ? *src_exp_data : (*src_exp_data + user_nn_snn_add_value);
			}
		}
		else {

		}
		src_data++;
		src_exp_data++;
		target_data++;
		thred_data++;
	}
}
//���ߵͽ�����ֵ����
//src_matrix ǰһ����������
//src_exp_matrix ǰһ���Ŀ��ֵ
//src_target_matrix ǰһ����Ҫ�ı��Ŀ��ֵ
//min_matrix ��Ԫ����ֵ
//max_matrix ��Ԫ����ֵ
//thred_matrix �������ֵ��Ҫ�ı��Ŀ��ֵ����
//��� ��
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *thred_matrix, user_nn_matrix *target_matrix, float step_value) {
	float *src_exp_data = src_exp_matrix->data;
	float *src_data = src_matrix->data;//
	float *target_data = target_matrix->data;//
	float *thred_data = thred_matrix->data;//
	
	//float avg_value = 1.0f;
	//float step_value = 0.001f;
	if (thred_matrix->width != src_matrix->height) {//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return;
	}
	if ((target_matrix->width != src_matrix->width) || (target_matrix->height != thred_matrix->height)) {
		return;
	}
	for (int height = 0; height < target_matrix->height; height++) {
		for (int width = 0; width < target_matrix->width; width++) {
			thred_data = thred_matrix->data + height * thred_matrix->width;//ָ���п�ͷ
			src_data = src_matrix->data + width;//ָ���п�ͷ
			src_exp_data = src_exp_matrix->data + width;
			for (int point = 0; point < src_matrix->height; point++) {
				if (*target_data == user_nn_snn_thred_add) {//�������ֵ
					if ((0.0f < *thred_data) && (*thred_data < *src_data)) {
						*thred_data += step_value;
						*src_exp_data = *src_data <= *thred_data ? *src_exp_data : (*src_exp_data - user_nn_snn_add_value);
					}
					if ((*thred_data < *src_data) && (*src_data < 0.0f)) {
						*thred_data += step_value;
						*src_exp_data = *src_data >= *thred_data ? (*src_exp_data - user_nn_snn_add_value) : *src_exp_data;
					}
				}
				else if (*target_data == user_nn_snn_thred_acc) {//�������ֵ
					if ((0.0f < *src_data) && (*src_data < *thred_data)) {
						*thred_data -= step_value;
						*src_exp_data = *src_data <= *thred_data ? (*src_exp_data + user_nn_snn_add_value) : *src_exp_data;
					}
					if ((*src_data < *thred_data) && (*thred_data < 0.0f)) {
						*thred_data -= step_value;
						*src_exp_data = *src_data >= *thred_data ? *src_exp_data : (*src_exp_data + user_nn_snn_add_value);
					}
				}
				else {

				}
				src_data += src_matrix->width;
				src_exp_data += src_exp_matrix->width;
				thred_data++;
			}
			thred_data++;
		}
	}
}
