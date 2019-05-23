

#ifndef _user_snn_layers_H
#define _user_snn_layers_H


#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../matrix/user_nn_initialization.h"

typedef enum _snn_layer_type {
	u_snn_layer_type_null = 0,
	u_snn_layer_type_input,
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

	user_nn_matrix		*min_kernel_matrix;//神经元w	
	user_nn_matrix		*max_kernel_matrix;//偏置参数
	user_nn_matrix		*feature_matrix;//存放计算后的特征数据 
	user_nn_matrix		*target_matrix;//目标矩阵
	user_nn_matrix		*thred_matrix;//保存本层的变化矩阵
}user_snn_output_layers;//输出层

#define snn_avg_vaule		1.0f
#define snn_step_vaule		0.001f

#define snn_thred_none		1.0f	//不需要变化
#define snn_thred_add		0.9f	//目标值大于输出值
#define snn_thred_acc		1.1f	//目标值小于输出值

void user_snn_data_softmax(user_nn_matrix *src_matrix);
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix * max_matrix);
void user_nn_matrix_thred_process(user_nn_matrix *thred_matrix, user_nn_matrix *src_matrix, user_nn_matrix *target_matrix);
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *output_matrix);//矩阵阈值累加
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value);//更新阈值


user_snn_layers *user_snn_layers_create(user_snn_layer_type type, int index);
user_snn_input_layers *user_snn_layers_input_create(user_snn_layers *nn_layers, int feature_width, int feature_height);
user_snn_hidden_layers *user_snn_layers_hidden_create(user_snn_layers *snn_layers, int feature_number);
user_snn_output_layers *user_snn_layers_output_create(user_snn_layers *nn_layers, int feature_number);


#endif