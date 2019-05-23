
#include "user_snn_layers.h"


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
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->feature_width = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->min_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//��Ԫ����
	hidden_layers->max_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//��Ԫ����

	hidden_layers->feature_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//�����������
	hidden_layers->thred_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//�����仯����

	user_snn_init_matrix(hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix);//��ʼ������

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
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;


	output_layers->min_kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//��Ԫ����
	output_layers->max_kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//��Ԫ����

	output_layers->feature_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//�����������
	output_layers->thred_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//�����仯����
	output_layers->target_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//����Ŀ�����

	user_snn_init_matrix(output_layers->min_kernel_matrix, output_layers->max_kernel_matrix);//��ʼ������

	return output_layers;
}


//������ֵ����
void user_snn_data_softmax(user_nn_matrix *src_matrix) {
	float count = (float)(src_matrix->height * src_matrix->width);
	float *src_data = src_matrix->data;
	float *max_value = user_nn_matrix_return_max_addr(src_matrix);
	if (*max_value > 1.0f) {
		user_nn_matrix_divi_constant(src_matrix, *max_value);//��һ
	}
	user_nn_matrix_sum_constant(src_matrix, (count - user_nn_matrix_cum_element(src_matrix)) / count);//ƽ��ֵ����Ϊ1.0f
	user_nn_matrix_divi_constant(src_matrix, 0.0001f);//����
	user_nn_matrxi_floor(src_matrix);//ȡ��
	user_nn_matrix_mult_constant(src_matrix, 0.0001f);//�˷�
	//*max_value += count - user_nn_matrix_cum_element(src_matrix);
	//printf("%-10.6f\n", user_nn_matrix_cum_element(src_matrix));
}
//��ʼ����ֵ����
//������� ��С������
//��� ��
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix *max_matrix) {
	int count = min_matrix->height * min_matrix->width;
	float *min_data = min_matrix->data;
	float *max_data = max_matrix->data;
	while (count--) {
		*min_data = user_nn_init_normal();
		*max_data = *min_data + user_nn_init_normal() + 1.0f;
		min_data++;
		max_data++;
	}
	//user_snn_data_softmax(min_matrix);
	//user_snn_data_softmax(max_matrix);
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
			*thred_data = snn_thred_add;
		}else if (*target_data < *src_data) {
			*thred_data = snn_thred_acc;
		}else {
			*thred_data = snn_thred_none;
		}
		src_data++;
		target_data++;
		thred_data++;
	}
}
//���������õ���ֵ�����ۼӼ��� �;���˷����� ֻ��ֵ�ǽ����ж�
//src_matrix �������
//min_matrix ����ֵ
//max_matrix ����ֵ
//��� ���ؽ������
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix,user_nn_matrix *output_matrix) {
	user_nn_matrix *result = NULL;//�������
	float *min_data = min_matrix->data;//
	float *src_data = src_matrix->data;//
	float *max_data = max_matrix->data;//
	float *output_data = output_matrix->data;
	if (min_matrix->width != src_matrix->height) {//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return;
	}
	if ((output_matrix->width != src_matrix->width) || (output_matrix->height != min_matrix->height)) {
		return;
	}

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for 
	for (int height = 0; height < output_matrix->height; height++) {
		for (int width = 0; width < output_matrix->width; width++) {
			for (int point = 0; point < src_matrix->height; point++) {
				if ((min_data[ height * min_matrix->width + point] <= src_data[width + point*src_matrix->width]) && (src_data[width + point*src_matrix->width] <= max_data[height * max_matrix->width + point])) {
					//if ((min_data[width + point*src_matrix->width] <= src_data[height * min_matrix->width + point]) && (src_data[height * min_matrix->width + point] <= max_data[width + point*src_matrix->width])) {
					output_data[height*output_matrix->width + width] += 1.0f;//����������ֵ����ۼ�
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
				if ((*min_data <= *src_data) && (*src_data <= *max_data)) {
					*output_data += 1.0f;//����������ֵ����ۼ�
				}
				src_data += src_matrix->width;
				min_data++;
				max_data++;
				}
			output_data++;
			}
		}
#endif

}

//���ߵͽ�����ֵ����
//src_matrix ǰһ����������
//src_target_matrix ǰһ����Ҫ�ı��Ŀ��ֵ
//min_matrix ��Ԫ����ֵ
//max_matrix ��Ԫ����ֵ
//thred_matrix �������ֵ��Ҫ�ı��Ŀ��ֵ����
//��� ��
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value) {
	float *src_exp_data = src_exp_matrix->data;
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *min_data = min_matrix->data;//
	float *max_data = max_matrix->data;//
	
	//float avg_value = 1.0f;
	//float step_value = 0.001f;
	if (min_matrix->width != src_matrix->height) {//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return;
	}
	if ((thred_matrix->width != src_matrix->width) || (thred_matrix->height != min_matrix->height)) {
		return;
	}
#if defined _OPENMP && _USER_API_OPENMP && false
#pragma omp parallel for 
	for (int height = 0; height < src_matrix->height; height++) {
		for (int width = 0; width < min_matrix->width; width++) {
			for (int point = 0; point < min_matrix->height; point++) {
				min_data = min_matrix->data + height * min_matrix->width + point;//ָ���п�ͷ
				max_data = max_matrix->data + height * max_matrix->width + point;//ָ���п�ͷ
				src_data = src_matrix->data + width + point*src_matrix->width;;//ָ���п�ͷ
				thred_data = thred_matrix->data + height*thred_matrix->width + width;

				if (*thred_data == snn_thred_heighten) {
					if (*src_data >= avg_value) {
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? *max_data : (*max_data + step_value);
					}
					else {
						*min_data = *min_data > *src_data ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else if (*thred_data == snn_thred_lower) {
					if (*src_data >= avg_value) {
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? (*max_data - step_value) : *max_data;
					}
					else {
						*min_data = *min_data > *src_data ? *min_data : (*min_data + step_value);
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else {

				}
			}
		}
	}
#else
	for (int height = 0; height < thred_matrix->height; height++) {
		for (int width = 0; width < thred_matrix->width; width++) {
			min_data = min_matrix->data + height * min_matrix->width;//ָ���п�ͷ
			max_data = max_matrix->data + height * max_matrix->width;//ָ���п�ͷ
			src_data = src_matrix->data + width;//ָ���п�ͷ
			src_exp_data = src_exp_matrix->data + width;
			for (int point = 0; point < src_matrix->height; point++) {
				if (*thred_data == snn_thred_add) {
					if (*src_data >= avg_value) {
						//avg_value = *src_data;
						//�ڱ���ǰһ��������������������ƶ���ֵ
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? *max_data : (*max_data + step_value);
						//�ڱ��ֱ�����ֵ����������ƶ�����ֵ
						*src_exp_data = *src_data < *min_data ? (*src_exp_data + 0.1f) : *src_exp_data;
						*src_exp_data = *src_data > *max_data ? (*src_exp_data - 0.1f) : *src_exp_data;
					}
					else {
						//avg_value = *src_data;
						*min_data = *min_data > *src_data ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);

						*src_exp_data = *src_data < *min_data ? (*src_exp_data + 0.1f) : *src_exp_data;
						*src_exp_data = *src_data > *max_data ? (*src_exp_data - 0.1f) : *src_exp_data;
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else if (*thred_data == snn_thred_acc) {
					if (*src_data >= avg_value) {
						//avg_value = *src_data;
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? (*max_data - step_value) : *max_data;

						if (*min_data < *src_data && *src_data < *max_data) {
							*src_exp_data += 0.1f;
						}
					}
					else {
						//avg_value = *src_data;
						*min_data = *min_data > *src_data ? *min_data : (*min_data + step_value);
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);

						if (*min_data < *src_data && *src_data < *max_data) {
							*src_exp_data -= 0.1f;
						}
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else {

				}
				src_data += src_matrix->width;
				src_exp_data += src_exp_matrix->width;
				min_data++;
				max_data++;
			}
			thred_data++;
		}
	}
#endif
}

/*
float min[] = { 0.5f,0.1f };
float max[] = { 1.6f,1.0f };

user_nn_matrix *src_matrix = user_nn_matrix_create_memset(1, 1, src);

user_nn_matrix *min_matrix = user_nn_matrix_create_memset(1, 2, min);
user_nn_matrix *max_matrix = user_nn_matrix_create_memset(1, 2, max);

user_nn_matrix *res_matrix = user_nn_matrix_create(1, 2);

user_nn_matrix_thred_acc(src_matrix, min_matrix, max_matrix, res_matrix);//

if (res_matrix != NULL) {
	user_nn_matrix_printf(NULL, res_matrix);//��ӡ����
}
else {
	printf("null\n");
}
printf("\nend");
*/