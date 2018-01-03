
#include "user_rnn_model.h"
#include "user_rnn_ffp.h"
#include "user_rnn_bptt.h"
#include "user_rnn_grads.h"

//通过输入层的信息创建模型
//layer_infor 网络模型参数
//返回 所创建的网络模型对象
user_rnn_layers *user_rnn_model_create(int *layer_infor) {
	user_rnn_layers			*rnn_layers = NULL;
	rnn_layers = user_rnn_layers_create(u_rnn_layer_type_null, 0);//创建一个空层

	while (1) {
		switch (*layer_infor) {
		case 'i':
			user_rnn_layers_input_create(rnn_layers, *(layer_infor + 1), *(layer_infor + 2), *(layer_infor + 3));	//输入层
			layer_infor += 4;
			break;
		case 'h':
			user_rnn_layers_hidden_create(rnn_layers, *(layer_infor + 1), *(layer_infor + 2));//隐含层
			layer_infor += 3;
			break;
		case 'o':
			user_rnn_layers_output_create(rnn_layers, *(layer_infor + 1), *(layer_infor + 2));//输出层
			layer_infor += 3;
			return rnn_layers;
		default:
			printf("set error\n"); while (1);
			break;
		}
	}
	return NULL;
}
//加载特征数据到指定到输入特征数据中
//layers 加载对象层
//src_matrix 目标数据
//返回 无
void user_rnn_model_load_input_feature(user_rnn_layers *layers, user_nn_list_matrix *src_matrix) {
	user_rnn_layers *rnn_input_layer = user_rnn_layers_get(layers, 1);//获取输入层
	user_nn_matrices_cpy_matrices(((user_rnn_input_layers *)rnn_input_layer->content)->feature_matrices, src_matrix);
}
//加载特征数据到指定到期望特征数据中
//layers 加载对象层
//src_matrix 目标数据
//返回 无
void user_rnn_model_load_target_feature(user_rnn_layers *layers, user_nn_list_matrix *src_matrix) {
	user_rnn_layers *rnn_output_layer = user_rnn_model_return_layer(layers, u_rnn_layer_type_output);//获取输入层
	user_nn_matrices_cpy_matrices(((user_rnn_output_layers *)rnn_output_layer->content)->target_matrices, src_matrix);
}
//正向执行一次迭代
//layers 所创建的层
//返回值 无
void user_rnn_model_ffp(user_rnn_layers *layers) {
	while (1) {
		switch (layers->type) {
		case u_rnn_layer_type_null:
			break;
		case u_rnn_layer_type_input:
			break;
		case u_rnn_layer_type_hidden:
			user_rnn_ffp_hidden(layers->prior, layers);//子采样处理
			break;
		case u_rnn_layer_type_output:
			user_rnn_ffp_output(layers->prior, layers);//输出层计算
			break;
		default:
			break;
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
}

//反向传播一次
//layers：层起始位置
//index：当前标签位置
//alpha：更新系数
//返回值：无
void user_rnn_model_bp(user_rnn_layers *layers, float alpha) {
	//取得指向最后一层数据指针
	while (layers->next != NULL) {
		layers = layers->next;
	}
	//反向计算残差
	while (1) {
		switch (layers->type) {
		case u_rnn_layer_type_null:
			break;
		case u_rnn_layer_type_input:
			break;
		case u_rnn_layer_type_hidden:
			user_rnn_bp_hidden_back_prior(layers->prior, layers);
			break;
		case u_rnn_layer_type_output:
			user_rnn_bp_output_back_prior(layers->prior, layers);
			break;
		default:
			break;
		}
		if (layers->prior == NULL) {
			break;
		}
		else {
			layers = layers->prior;
		}
	}
	//求解权重残差值
	while (1) {
		switch (layers->type) {
		case u_rnn_layer_type_null:
			break;
		case u_rnn_layer_type_input:
			break;
		case u_rnn_layer_type_hidden:
			user_rnn_grads_hidden(layers, alpha);//更新权重
			break;
		case u_rnn_layer_type_output:
			user_rnn_grads_output(layers, alpha);//更新权重
			break;
		default:
			break;
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
}
//获取loss损失值
//layers 获取对象层
//返回 损失值的大小
float user_rnn_model_return_loss(user_rnn_layers *layers) {
	static float loss_function = 0;//全局变量的loss值
	while (1) {
		if (layers->type == u_rnn_layer_type_output) {
			if (loss_function == 0) {
				loss_function = ((user_rnn_output_layers *)layers->content)->loss_function;
			}
			else {
				loss_function = (float)0.99f * loss_function + 0.01f * ((user_rnn_output_layers *)layers->content)->loss_function;
			}
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return loss_function;
}

//从整个网络中获取一个指定层 按顺序查找
//layers 查找的对象层
//type 目标层类型
//返回 结果对象层
user_rnn_layers *user_rnn_model_return_layer(user_rnn_layers *layers, user_rnn_layer_type type) {
	while (1) {
		if (layers->type == type) {
			return layers;//返回最大值
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return NULL;
}
//获取输出矩阵
//layers 获取对象层
//返回 矩阵值
user_nn_list_matrix *user_rnn_model_return_result(user_rnn_layers *layers) {
	user_nn_list_matrix *result = NULL;//
	while (1) {
		if (layers->type == u_rnn_layer_type_output) {
			result = ((user_rnn_output_layers *)layers->content)->feature_matrices;
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return result;
}