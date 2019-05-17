
#include "user_snn_layers.h"

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
//
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
	user_snn_data_softmax(min_matrix);
	user_snn_data_softmax(max_matrix);
}

//����ֵ����
//src_matrix �������
//target_matrix Ŀ�����
//���� �������
user_nn_matrix *user_nn_matrix_thred_process(user_nn_matrix *src_matrix, user_nn_matrix *target_matrix) {
	user_nn_matrix *result = NULL;//�������
	int count = src_matrix->width * src_matrix->height;
	float *src_data = src_matrix->data;
	float *target_data = target_matrix->data;
	float *result_data = NULL;
	result = user_nn_matrix_create(src_matrix->width, src_matrix->height);//�����µľ���
	result_data = result->data;//��ȡ����ָ��
	while (count--) {
		if (*target_data > *src_data) {
			*result_data = thred_heighten;
		}
		else if (*target_data < *src_data) {
			*result_data = thred_lower;
		}
		else {
			*result_data = thred_none;
		}
		src_data++;
		target_data++;
		result_data++;
	}
	return result;
}
//���������õ���ֵ�����ۼӼ��� �;���˷����� ֻ��ֵ�ǽ����ж�
//src_matrix �������
//min_matrix ����ֵ
//max_matrix ����ֵ
//��� ���ؽ������
user_nn_matrix *user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix) {
	user_nn_matrix *result = NULL;//�������
	float *src_data = src_matrix->data;//
	float *min_data = min_matrix->data;//
	float *max_data = max_matrix->data;//
	float *result_data = NULL;
	//int width, height, point;//��������
	if (src_matrix->width != min_matrix->height) {//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return NULL;
	}
	result = user_nn_matrix_create(min_matrix->width, src_matrix->height);//�����µľ���
	result_data = result->data;//��ȡ����ָ��
#if defined _OPENMP && _USER_API_OPENMP && false
#pragma omp parallel for 
	for (int height = 0; height < result->height; height++) {
		for (int width = 0; width < result->width; width++) {
			for (int point = 0; point < min_matrix->height; point++) {
				if ((min_data[width + point*min_matrix->width] <= src_data[height * src_matrix->width + point]) && (src_data[height * src_matrix->width + point] <= max_data[width + point*max_matrix->width])) {
					result_data[height*result->width + width] += 1.0f;//����������ֵ����ۼ�
				}
			}
		}
	}
#else
	for (int height = 0; height < result->height; height++) {
		for (int width = 0; width < result->width; width++) {
			src_data = src_matrix->data + height * src_matrix->width;//ָ���п�ͷ
			min_data = min_matrix->data + width;//ָ���п�ͷ
			max_data = max_matrix->data + width;//ָ���п�ͷ
			for (int point = 0; point < min_matrix->height; point++) {
				if ((*min_data <= *src_data) && (*src_data <= *max_data)) {
					*result_data += 1.0f;//����������ֵ����ۼ�
				}
				max_data += max_matrix->width;
				min_data += min_matrix->width;
				src_data++;
			}
			result_data++;
		}
	}
#endif
	return result;
}

//���ߵͽ�����ֵ����
//src_matrix �������
//min_matrix ����ֵ
//max_matrix ����ֵ
//��� ���ؽ������
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *thred_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, float avg_value, float step_value) {
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *min_data = min_matrix->data;//
	float *max_data = max_matrix->data;//
	//float avg_value = 1.0f;
	//float step_value = 0.001f;
	if (src_matrix->width != min_matrix->height) {//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return;
	}
#if defined _OPENMP && _USER_API_OPENMP && false
#pragma omp parallel for 
	for (int height = 0; height < src_matrix->height; height++) {
		for (int width = 0; width < min_matrix->width; width++) {
			for (int point = 0; point < min_matrix->height; point++) {
				src_data = src_matrix->data + height * src_matrix->width + point;
				thred_data = thred_matrix->data + height*thred_matrix->width + width;
				min_data = min_matrix->data + width + point*min_matrix->width;
				max_data = max_matrix->data + width + point*max_matrix->width;

				if (*thred_data == thred_heighten) {
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
				else if (*thred_data == thred_lower) {
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
	for (int height = 0; height < src_matrix->height; height++) {
		for (int width = 0; width < min_matrix->width; width++) {
			src_data = src_matrix->data + height * src_matrix->width;//ָ���п�ͷ
			min_data = min_matrix->data + width;//ָ���п�ͷ
			max_data = max_matrix->data + width;//ָ���п�ͷ
			for (int point = 0; point < min_matrix->height; point++) {
				if (*thred_data == thred_heighten) {
					if (*src_data >= avg_value) {
						//avg_value = *src_data;
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? *max_data : (*max_data + step_value);
					}
					else {
						//avg_value = *src_data;
						*min_data = *min_data > *src_data ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else if (*thred_data == thred_lower) {
					if (*src_data >= avg_value) {
						//avg_value = *src_data;
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? (*max_data - step_value) : *max_data;
					}
					else {
						//avg_value = *src_data;
						*min_data = *min_data > *src_data ? *min_data : (*min_data + step_value);
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else {

				}
				max_data += max_matrix->width;
				min_data += min_matrix->width;
				src_data++;
			}
			thred_data++;
		}
	}
#endif
}