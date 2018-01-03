

#ifndef _user_nn_layers_H
#define _user_nn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
//#include "user_nn_create.h"

typedef enum _nn_layer_type {
	u_nn_layer_type_null = 0,
	u_nn_layer_type_input,
	u_nn_layer_type_hidden,
	u_nn_layer_type_output
}user_nn_layer_type;

typedef struct _nn_layers {
	struct _nn_layers *prior;//上一个ceng
	int index;//指数
	user_nn_layer_type type;//类型
	void *content;//对象
	struct _nn_layers *next;//下一层
}user_nn_layers;

typedef struct _nn_input_layers {
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	user_nn_matrix	*deltas_matrix;//保存本层的数据残差
	user_nn_matrix	*feature_matrix;//存放输入数据矩阵
}user_nn_input_layers;

typedef struct _nn_hidden_layers {
	int input_feature_number;//输入数据个数
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度

	user_nn_matrix		*kernel_matrix;//存放本次数据如数到隐藏层的数据	
	user_nn_matrix		*biases_matrix;//偏置参数
	user_nn_matrix		*feature_matrix;//存放计算后的特征数据 按照时间序列因此有多个

	user_nn_matrix		*deltas_matrix;//保存本层的数据残差
	user_nn_matrix		*deltas_kernel_matrix;//本层残差对前一层feture maps的结果也就是ΔW的值	
	user_nn_matrix		*deltas_biases_matrix;//误差对bias导数
}user_nn_hidden_layers;//

//输出层
typedef struct _nn_output_layers {
	int input_feature_number;//输入数据个数
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	float loss_function;//代价函数

	user_nn_matrix		*kernel_matrix;//存放隐藏层核 内核共用
	user_nn_matrix		*biases_matrix;//偏置参数  偏置参数共用
	user_nn_matrix		*feature_matrix;//存放计算后的特征数据 按照时间序列因此有多个
	user_nn_matrix		*target_matrix;//目标矩阵
	user_nn_matrix		*deltas_matrix;//保存本层的数据残差
	user_nn_matrix		*deltas_kernel_matrix;//本层残差对前一层feture maps的结果也就是ΔW的值
	user_nn_matrix		*deltas_biases_matrix;//误差对Δbias导数
	user_nn_matrix		*error_matrix;//输出层错误值矩阵保存	
}user_nn_output_layers;//输出层

user_nn_layers *user_nn_layers_get(user_nn_layers *dest, int index);
user_nn_layers *user_nn_layers_create(user_nn_layer_type type, int index);
void user_nn_layers_delete(user_nn_layers *layers);

user_nn_input_layers *user_nn_layers_input_create(user_nn_layers *nn_layers, int feature_width, int feature_height);
user_nn_hidden_layers *user_nn_layers_hidden_create(user_nn_layers *nn_layers, int feature_number);
user_nn_output_layers *user_nn_layers_output_create(user_nn_layers *nn_layers, int feature_number);


#endif