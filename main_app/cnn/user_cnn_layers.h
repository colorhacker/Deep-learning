

#ifndef _user_cnn_layers_H
#define _user_cnn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _cnn_layer_type{
	u_cnn_layer_type_null = 0,
	u_cnn_layer_type_input,
	u_cnn_layer_type_conv,
	u_cnn_layer_type_pool,
	u_cnn_layer_type_full,
	u_cnn_layer_type_output
}user_cnn_layer_type;

typedef struct _cnn_layers{
	struct _cnn_layers *prior;//上一个层
	int index;//指数
	user_cnn_layer_type type;//类型
	void *content;//对象
	struct _cnn_layers *next;//下一层
}user_cnn_layers;

typedef struct _cnn_input_layers{
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	int feature_number;//本层具有特征个数
	user_nn_list_matrix *feature_matrices;//存放输入数据矩阵
}user_cnn_input_layers;
typedef struct _cnn_conv_layers{
	int input_feature_number;//输入数据个数
	int feature_number;//输出数据个数
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	int kernel_width;//卷积核的宽度
	int kernel_height;//卷积核的宽度

	user_nn_matrix		*biases_matrix;//偏置参数
	user_nn_list_matrix *feature_matrices;//存放卷积后的特征数据
	user_nn_list_matrix *kernel_matrices;//存放卷积核
	user_nn_list_matrix *deltas_matrices;//残差保存
	user_nn_list_matrix *deltas_kernel_matrices;//本层残差对前一层feture maps的卷积结果也就是ΔW的值
	user_nn_matrix		*deltas_biases_matrix;//误差对bias导数
}user_cnn_conv_layers;//卷积层
typedef struct _cnn_pool_layers{
	int input_feature_number;//输入数据个数
	int feature_number;//输出数据个数
	int pool_width;//池化宽度
	int pool_height;//池化高度
	user_nn_pooling_type pool_type;//池化方式
	int feature_width;//pooling模板数据宽度
	int feature_height;//pooling模板数据高度
	user_nn_matrix *kernel_matrix;//pooling矩阵数据 均值化矩阵核
	user_nn_list_matrix *feature_matrices;//存放卷积后的特征数据
	user_nn_list_matrix *deltas_matrices;//残差保存
}user_cnn_pool_layers;//池化层
typedef struct _cnn_full_layers{
	int feature_number;//本层特征数据个数
	//int feature_width;//full模板数据宽度
	//int feature_height;//full模板数据高度
	user_nn_matrix *input_feature_matrix;//输入特征数据 是上层的所有特征拉成一个矩阵
	user_nn_matrix *kernel_matrix;//输出层矩阵乘积核
	user_nn_matrix *biases_matrix;//输出层偏置参数
	user_nn_matrix *feature_matrix;//输出层特征数据
	user_nn_matrix *deltas_matrix;//残差保存
	user_nn_matrix *deltas_kernel_matrix;//本层残差对前一层feture maps的卷积结果也就是ΔW的值
}user_cnn_full_layers;//全连接
//输出层
typedef struct _cnn_output_layers{
	int feature_number;//本层具有特征个数
	int class_number;//分类个数---临时保存
	float loss_function;//代价函数

	user_nn_matrix *input_feature_matrix;//输入特征数据 是上层的所有特征拉成一个矩阵
	user_nn_matrix *kernel_matrix;//输出层矩阵乘积核
	user_nn_matrix *biases_matrix;//输出层偏置参数
	user_nn_matrix *feature_matrix;//输出层特征数据
	user_nn_matrix *target_matrix;//目标矩阵特征数据
	user_nn_matrix *error_matrix;//输出层错误值矩阵保存
	user_nn_matrix *deltas_matrix;//输出层残差
	user_nn_matrix *deltas_kernel_matrix;//本层残差对前一层feture maps的卷积结果也就是ΔW的值
}user_cnn_output_layers;//输出层

user_cnn_layers			*user_cnn_layers_get(user_cnn_layers *dest, int index);
user_cnn_layers			*user_cnn_layers_create(user_cnn_layer_type type, int index);
void user_cnn_layers_delete(user_cnn_layers *layers);
user_cnn_input_layers	*user_cnn_layers_input_create(user_cnn_layers *cnn_layers, int feature_width, int feature_height, int feature_number);
user_cnn_conv_layers	*user_cnn_layers_convolution_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height ,int feature_number);
user_cnn_pool_layers	*user_cnn_layers_pooling_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height, user_nn_pooling_type pool_type);
user_cnn_full_layers	*user_cnn_layers_fullconnect_create(user_cnn_layers *cnn_layers);
user_cnn_output_layers	*user_cnn_layers_output_create(user_cnn_layers *cnn_layers, int class_number);

#endif