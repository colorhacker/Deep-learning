#include "./user_nn_initialization.h"

//返回0~+1的随机数
float user_nn_init_normal(void) {
	return (float)((float)rand() / RAND_MAX);
}
//返回 -1~+1的随机数
float user_nn_init_uniform(void) {
	return (float)(((float)(rand()*2.0 / RAND_MAX) - 1.0f));
}
//lecun初始化
float user_nn_init_lecun_uniform(int input_count, int output_count) {
	return (float)(user_nn_init_uniform()*sqrt(3.0 / input_count));
}
//glorot Xavier初始化 
float user_nn_init_glorot_normal(int input_count, int output_count) {
	return (float)(user_nn_init_normal()*sqrt(2.0 / (input_count + output_count)));
}
//glorot初始化
float user_nn_init_glorot_uniform(int input_count, int output_count) {
	return (float)(user_nn_init_uniform()*sqrt(6.0 / (input_count + output_count)));
}
//he初始化
float user_nn_init_he_normal(int input_count, int output_count) {
	return (float)(user_nn_init_normal()*sqrt(2.0 / input_count));
}
//he初始化
float user_nn_init_he_uniform(int input_count, int output_count) {
	return (float)(user_nn_init_normal()*sqrt(6.0 / input_count));
}
//orthogonal初始化
float user_nn_init_orthogonal(int input_count, int output_count) {
	return 0;
}
//identity初始化
float user_nn_init_identity(int input_count, int output_count) {
	//scale*矩阵对角线全为1
	return 0;
}


//设置单个居中里面的值 随机设置
void user_nn_matrix_init_vaule(user_nn_matrix *src_matrix, int input, int output) {
	int total = src_matrix->height * src_matrix->width;//设置高度与宽度
	float *data = src_matrix->data;//获取内存指针
	while (total--) {
		*data++ = user_nn_init_rand(input, output);//设置值
	}
}
//设置连续矩阵里面的值 随机设置
void user_nn_matrices_init_vaule(user_nn_list_matrix *list_matrix, int input, int output) {
	user_nn_matrix *matrix = list_matrix->matrix;//获取第一个矩阵对象
	float *data = 0;
	int total = 0;

	if (matrix == NULL) {
		return;
	}
	while (matrix != NULL) {
		user_nn_matrix_init_vaule(matrix, input, output);
		matrix = matrix->next;//继续处理下一个数据
	}
}
