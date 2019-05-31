

#ifndef _user_snn_layers_H
#define _user_snn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _snn_layer_type {
	u_snn_layer_type_null = 0,
	u_snn_layer_type_input,
	u_snn_layer_type_flat,
	u_snn_layer_type_hidden,
	u_snn_layer_type_output
}user_snn_layer_type;

typedef struct _snn_layers {
	struct _snn_layers *prior;//上一个ceng
	int index;//指数
	user_snn_layer_type type;//类型
	void *content;//对象
	struct _snn_layers *next;//下一层
}user_snn_layers;

typedef struct _snn_input_layers {
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	user_nn_matrix	*feature_matrix;//存放输入数据矩阵
	user_nn_matrix	*thred_matrix;//保存本层的变化矩阵
}user_snn_input_layers;

typedef struct _snn_flat_layers {
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度

	user_nn_matrix		*min_kernel_matrix;//神经元w	
	user_nn_matrix		*max_kernel_matrix;//偏置参数
	user_nn_matrix		*feature_matrix;//存放计算后的特征数据 
	user_nn_matrix		*thred_matrix;//保存本层的变化矩阵
}user_snn_flat_layers;//

typedef struct _snn_hidden_layers {
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度

	user_nn_matrix		*min_kernel_matrix;//神经元w	
	user_nn_matrix		*max_kernel_matrix;//偏置参数
	user_nn_matrix		*feature_matrix;//存放计算后的特征数据 
	user_nn_matrix		*thred_matrix;//保存本层的变化矩阵
}user_snn_hidden_layers;//

					   //输出层
typedef struct _snn_output_layers {
	int feature_width;//数据宽度 特征数据的宽度
	int feature_height;//数据高度 特征数据的高度
	float loss_function;//损失函数

	user_nn_matrix		*min_kernel_matrix;//神经元w	
	user_nn_matrix		*max_kernel_matrix;//偏置参数
	user_nn_matrix		*feature_matrix;//存放计算后的特征数据 
	user_nn_matrix		*target_matrix;//目标矩阵
	user_nn_matrix		*thred_matrix;//保存本层的变化矩阵
}user_snn_output_layers;//输出层



void user_snn_data_softmax(user_nn_matrix *src_matrix);
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix * max_matrix);
void user_nn_matrix_thred_process(user_nn_matrix *thred_matrix, user_nn_matrix *src_matrix, user_nn_matrix *target_matrix);
void user_nn_matrix_thred_flat(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *output_matrix);
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *output_matrix);//矩阵阈值累加
void user_nn_matrix_update_flat(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value);
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value);//更新阈值


user_snn_layers *user_snn_layers_get(user_snn_layers *dest, int index);
user_snn_layers *user_snn_layers_create(user_snn_layer_type type, int index);
void user_snn_layers_delete(user_snn_layers *layers);
void user_snn_layers_all_delete(user_snn_layers *layers);
user_snn_input_layers *user_snn_layers_input_create(user_snn_layers *nn_layers, int feature_width, int feature_height);
user_snn_flat_layers *user_snn_layers_flat_create(user_snn_layers *snn_layers);
user_snn_hidden_layers *user_snn_layers_hidden_create(user_snn_layers *snn_layers, int feature_number);
user_snn_output_layers *user_snn_layers_output_create(user_snn_layers *nn_layers, int feature_number);


#endif