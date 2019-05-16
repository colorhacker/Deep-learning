#include "user_nn_activate.h"

//对float参数进行softmax激活
//value：激活对象
//type：激活类型
//返回 ：结果
float user_nn_activate(float value, activation_type type) {
	switch (type) {
		case activation_sigmoid:
			return 1.0f / (1.0f + exp(-value));//S生长型函数
		case activation_tanh:
			return tanh(value);//双曲函数
		case activation_prelu:
			return value >= 0.0f? value : 0.0f;//relu激活
		default:break;
	}
	return 0;
}
//对float参数进行softmax求导
//value：激活对象
//type：激活类型
//返回 ：结果
float user_nn_activate_d(float value, activation_type type) {
	switch (type) {
	case activation_sigmoid:
		return value*(1.0f - value);//S生长型函数
	case activation_tanh:
		return (1.0f - value*value);//双曲函数
	case activation_prelu:
		return value >= 0.0f ? 1.0f : 0.0f;//relu激活
	default:break;
	}
	return 0;
}

//采用激活函数对矩阵进行激活处理
//dest_matrix:被激活的对象，数据同时保存此矩阵中
//返回：无
void user_nn_activate_matrix(user_nn_matrix *dest_matrix, activation_type type) {
	int count = dest_matrix->width * dest_matrix->height;//获取矩阵数据大小
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

//采用激活函数对矩阵进行求导处理
//dest_matrix:被求导的对象，数据同时保存此矩阵中
//返回：无
void user_nn_activate_matrix_d(user_nn_matrix *dest_matrix, activation_type type) {
	int count = dest_matrix->width * dest_matrix->height;//获取矩阵数据大小
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

//采用激活函数对矩阵加上一个值进行激活处理
//公式：save_matrix=sigmoid(src_matrix + constant);
//参数
//save_matrix：保存矩阵
//src_matrix：被激活的矩阵
//constant：被激活的矩阵加上一个值
//type：激活类型
//返回 无
void user_nn_activate_matrix_sum_constant(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, float constant, activation_type type) {
	int count = src_matrix->width * src_matrix->height;//获取矩阵数据大小
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
//采用激活函数对矩阵加上一个矩阵进行激活处理
//公式：save_matrix=sigmoid(src_matrix + sub_matrix);
//参数
//save_matrix：保存矩阵
//src_matrix：被激活的矩阵
//sub_matrix：被激活的矩阵加上一个矩阵
//type：激活类型
//返回 无
void user_nn_activate_matrix_sum_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type) {
	int count = src_matrix->width * src_matrix->height;//获取矩阵数据大小
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
//对矩阵进行求导数，结果乘矩阵，进行返回
//参数
//save_matrix：保存矩阵
//src_matrix：被求和矩阵
//sub_matrix：求导矩阵
//返回 无
void user_nn_activate_matrix_d_mult_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, activation_type type) {
	int count = src_matrix->width * src_matrix->height;//获取矩阵数据大小
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
//对连续矩阵进行求导数，结果乘矩阵，进行返回
//参数
//save_matrices：保存连续矩阵
//src_matrices：被求和连续矩阵
//sub_matrices：求导连续矩阵
//返回 无
void user_nn_activate_matrices_d_mult_matrices(user_nn_list_matrix *save_matrices, user_nn_list_matrix *src_matrices, user_nn_list_matrix *sub_matrices, activation_type type) {
	int count = src_matrices->width * src_matrices->height;//获取矩阵数据大小
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
