#include "./user_nn_initialization.h"

//����0~+1�������
float user_nn_init_normal(void) {
	return (float)((float)rand() / RAND_MAX);
}
//���� -1~+1�������
float user_nn_init_uniform(void) {
	return (float)(((float)(rand()*2.0 / RAND_MAX) - 1.0f));
}
//lecun��ʼ��
float user_nn_init_lecun_uniform(int input_count, int output_count) {
	return (float)(user_nn_init_uniform()*sqrt(3.0 / input_count));
}
//glorot Xavier��ʼ�� 
float user_nn_init_glorot_normal(int input_count, int output_count) {
	return (float)(user_nn_init_normal()*sqrt(2.0 / (input_count + output_count)));
}
//glorot��ʼ��
float user_nn_init_glorot_uniform(int input_count, int output_count) {
	return (float)(user_nn_init_uniform()*sqrt(6.0 / (input_count + output_count)));
}
//he��ʼ��
float user_nn_init_he_normal(int input_count, int output_count) {
	return (float)(user_nn_init_normal()*sqrt(2.0 / input_count));
}
//he��ʼ��
float user_nn_init_he_uniform(int input_count, int output_count) {
	return (float)(user_nn_init_normal()*sqrt(6.0 / input_count));
}
//orthogonal��ʼ��
float user_nn_init_orthogonal(int input_count, int output_count) {
	return 0;
}
//identity��ʼ��
float user_nn_init_identity(int input_count, int output_count) {
	//scale*����Խ���ȫΪ1
	return 0;
}


//���õ������������ֵ �������
void user_nn_matrix_init_vaule(user_nn_matrix *src_matrix, int input, int output) {
	int total = src_matrix->height * src_matrix->width;//���ø߶�����
	float *data = src_matrix->data;//��ȡ�ڴ�ָ��
	while (total--) {
		*data++ = user_nn_init_rand(input, output);//����ֵ
	}
}
//�����������������ֵ �������
void user_nn_matrices_init_vaule(user_nn_list_matrix *list_matrix, int input, int output) {
	user_nn_matrix *matrix = list_matrix->matrix;//��ȡ��һ���������
	float *data = 0;
	int total = 0;

	if (matrix == NULL) {
		return;
	}
	while (matrix != NULL) {
		user_nn_matrix_init_vaule(matrix, input, output);
		matrix = matrix->next;//����������һ������
	}
}
