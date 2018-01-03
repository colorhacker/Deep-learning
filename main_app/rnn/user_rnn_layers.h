

#ifndef _user_rnn_layers_H
#define _user_rnn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
//#include "user_rnn_create.h"

typedef enum _rnn_layer_type {
	u_rnn_layer_type_null = 0,
	u_rnn_layer_type_input,
	u_rnn_layer_type_hidden,
	u_rnn_layer_type_output
}user_rnn_layer_type;

typedef struct _rnn_layers {
	struct _rnn_layers *prior;//上一个ceng
	int index;//指数
	user_rnn_layer_type type;//类型
	void *content;//对象
	struct _rnn_layers *next;//下一层
}user_rnn_layers;

typedef struct _rnn_input_layers {
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	int time_number;//本层具有特征个数
	user_nn_list_matrix	*deltas_matrices;//保存本层的数据残差
	user_nn_list_matrix *feature_matrices;//存放输入数据矩阵
}user_rnn_input_layers;

typedef struct _rnn_hidden_layers {
	int input_feature_number;//输入数据个数
	int time_number;//输出数据个数
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度

	user_nn_matrix		*kernel_matrix;//存放本次数据如数到隐藏层的数据	
	user_nn_matrix		*kernel_matrix_t;//存放上个时间节点隐藏层数据到此时时间节点的神经核
	user_nn_matrix		*biases_matrix;//偏置参数
	user_nn_matrix		*feature_matrix_t;//time-1时间的隐藏层特征
	user_nn_list_matrix *feature_matrices;//存放计算后的特征数据 按照时间序列因此有多个

	user_nn_list_matrix	*deltas_matrices;//保存本层的数据残差
	user_nn_matrix		*deltas_kernel_matrix;//本层残差对前一层feture maps的结果也就是ΔW的值	
	user_nn_matrix		*deltas_matrix_t;//保存本层的数据残差
	user_nn_matrix		*deltas_kernel_matrix_t;//本层残差对前一层feture maps的结果也就是ΔW的值	
	user_nn_matrix		*deltas_biases_matrix;//误差对bias导数
}user_rnn_hidden_layers;//

//输出层
typedef struct _rnn_output_layers {
	int input_feature_number;//输入数据个数
	int time_number;//输出数据个数
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	float loss_function;//代价函数

	user_nn_matrix		*kernel_matrix;//存放隐藏层核 内核共用
	user_nn_matrix		*biases_matrix;//偏置参数  偏置参数共用
	user_nn_list_matrix *feature_matrices;//存放计算后的特征数据 按照时间序列因此有多个
	user_nn_list_matrix *target_matrices;//目标矩阵
	user_nn_list_matrix	*deltas_matrices;//保存本层的数据残差
	user_nn_matrix		*deltas_kernel_matrix;//本层残差对前一层feture maps的结果也就是ΔW的值
	user_nn_matrix		*deltas_biases_matrix;//误差对Δbias导数
	user_nn_matrix		*error_matrix;//输出层错误值矩阵保存	
}user_rnn_output_layers;//输出层

user_rnn_layers *user_rnn_layers_get(user_rnn_layers *dest, int index);
user_rnn_layers *user_rnn_layers_create(user_rnn_layer_type type, int index);
void user_rnn_layers_delete(user_rnn_layers *layers);

user_rnn_input_layers *user_rnn_layers_input_create(user_rnn_layers *rnn_layers, int feature_width, int feature_height, int time_number);
user_rnn_hidden_layers *user_rnn_layers_hidden_create(user_rnn_layers *rnn_layers, int feature_number, int time_number);
user_rnn_output_layers *user_rnn_layers_output_create(user_rnn_layers *rnn_layers, int feature_number, int time_number);


#endif