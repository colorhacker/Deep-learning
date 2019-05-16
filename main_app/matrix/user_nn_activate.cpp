#include "user_nn_activate.h"

//��float��������softmax����
//value���������
//type����������
//���� �����
float user_nn_activate(float value, activation_type type) {
	switch (type) {
		case activation_sigmoid:
			return 1.0f / (1.0f + exp(-value));//S�����ͺ���
		case activation_tanh:
			return tanh(value);//˫������
		case activation_prelu:
			return value >= 0.0f? value : 0.0f;//relu����
		default:break;
	}
	return 0;
}
//��float��������softmax��
//value���������
//type����������
//���� �����
float user_nn_activate_d(float value, activation_type type) {
	switch (type) {
	case activation_sigmoid:
		return value*(1.0f - value);//S�����ͺ���
	case activation_tanh:
		return (1.0f - value*value);//˫������
	case activation_prelu:
		return value >= 0.0f ? 1.0f : 0.0f;//relu����
	default:break;
	}
	return 0;
}

//���ü�����Ծ�����м����
//dest_matrix:������Ķ�������ͬʱ����˾�����
//���أ���
void user_nn_activate_matrix(user_nn_matrix *dest_matrix, activation_type type) {
	int count = dest_matrix->width * dest_matrix->height;//��ȡ�������ݴ�С
	float *dest_data = dest_matrix->data;

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for
	for (int index = 0; index < count; index++) {
		dest_data[index] = user_nn_activate((float)dest_data[index], type);
	}
#else
	while (count--) {
		*dest_data++ = user_nn_activate_softmax(*dest_data, type);
	}
#endif
}

//���ü�����Ծ�������󵼴���
//dest_matrix:���󵼵Ķ�������ͬʱ����˾�����
//���أ���
void user_nn_activate_matrix_d(user_nn_matrix *dest_matrix, activation_type type) {
	int count = dest_matrix->width * dest_matrix->height;//��ȡ�������ݴ�С
	float *dest_data = dest_matrix->data;

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for
	for (int index = 0; index < count; index++) {
		dest_data[index] = user_nn_activate_d((float)dest_data[index], type);
	}
#else
	while (count--) {
		*dest_data++ = user_nn_activate_softmax_d(*dest_data, type);
	}
#endif
}

//���ü�����Ծ������һ��ֵ���м����
//��ʽ��save_matrix=sigmoid(src_matrix + constant);
//����
//save_matrix���������
//src_matrix��������ľ���
//constant��������ľ������һ��ֵ
//type����������
//���� ��
void user_nn_activate_matrix_sum_constant(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, float constant, activation_type type) {
	int count = src_matrix->width * src_matrix->height;//��ȡ�������ݴ�С
	float *src_data = src_matrix->data;
	float *save_data = save_matrix->data;
#if defined _OPENMP && _USER_API_OPENMP
	#pragma omp parallel for
	for (int index = 0; index < count; index++) {
		save_data[index] = user_nn_activate((float)src_data[index] + constant, type);
	}
#else
	while (count--) {
		*save_data++ = user_nn_activate_softmax(*src_data++ + constant, type);
	}
#endif
}
//���ü�����Ծ������һ��������м����
//��ʽ��save_matrix=sigmoid(src_matrix + sub_matrix);
//����
//save_matrix���������
//src_matrix��������ľ���
//sub_matrix��������ľ������һ������
//type����������
//���� ��
void user_nn_activate_matrix_sum_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type) {
	int count = src_matrix->width * src_matrix->height;//��ȡ�������ݴ�С
	float *src_data = src_matrix->data;
	float *save_data = save_matrix->data;
	float *sub_data = sub_matrix->data;

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for
	for (int index = 0; index < count; index++) {
		save_data[index] = user_nn_activate(src_data[index] + sub_data[index], type);
	}
#else
	while (count--) {
		*save_data++ = user_nn_activate_softmax(*src_data++ + *sub_data++, type);
	}
#endif

}
//�Ծ����������������˾��󣬽��з���
//����
//save_matrix���������
//src_matrix������;���
//sub_matrix���󵼾���
//���� ��
void user_nn_activate_matrix_d_mult_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type) {
	int count = src_matrix->width * src_matrix->height;//��ȡ�������ݴ�С
	float *src_data = src_matrix->data;
	float *save_data = save_matrix->data;
	float *sub_data = sub_matrix->data;

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for
	for (int index = 0; index < count; index++) {
		save_data[index] = src_data[index] * user_nn_activate_d(sub_data[index], type);
	}
#else
	while (count--) {
		*save_data++ = *src_data++ * user_nn_activate_softmax_d(*sub_data++, type);
	}
#endif
}
//�����������������������˾��󣬽��з���
//����
//save_matrices��������������
//src_matrices���������������
//sub_matrices������������
//���� ��
void user_nn_activate_matrices_d_mult_matrices(user_nn_list_matrix *save_matrices, user_nn_list_matrix *src_matrices, user_nn_list_matrix *sub_matrices, activation_type type) {
	int count = src_matrices->width * src_matrices->height;//��ȡ�������ݴ�С
	user_nn_matrix *save_data = save_matrices->matrix;
	user_nn_matrix *src_data = src_matrices->matrix;
	user_nn_matrix *sub_data = sub_matrices->matrix;
	while (count--) {
		user_nn_activate_matrix_d_mult_matrix(save_data, src_data, sub_data,type);
		save_data = save_data->next;
		src_data = src_data->next;
		sub_data = sub_data->next;
	}
}
