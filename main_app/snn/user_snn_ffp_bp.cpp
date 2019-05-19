
#include "user_snn_ffp_bp.h"

//intput->hidden->output
//hidden=tanh(input*w1+hidden_t*w2+bias1)
//
void user_snn_ffp_hidden(user_snn_layers *prior_layer, user_snn_layers *hidden_layer) {
	user_snn_output_layers *hidden_layers = (user_snn_output_layers  *)hidden_layer->content;//获取本层的参数
	user_nn_matrix *input_feature_matrix = NULL;
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_output_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_memset(hidden_layers->feature_matrix, 0.0f);
	user_nn_matrix_thred_acc(input_feature_matrix, hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix, hidden_layers->feature_matrix);//计算值
	user_nn_matrix_cpy_matrix(hidden_layers->softmax_feature_matrix, hidden_layers->feature_matrix);
	user_snn_data_softmax(hidden_layers->softmax_feature_matrix);//数据归一化处理
}

void user_snn_ffp_output(user_snn_layers *prior_layer, user_snn_layers *output_layer) {
	user_snn_output_layers *output_layers = (user_snn_output_layers  *)output_layer->content;//获取本层的参数
	user_nn_matrix *input_feature_matrix = NULL;
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_output_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_memset(output_layers->feature_matrix,0.0f);//清空特征数据
	user_nn_matrix_thred_acc(input_feature_matrix, output_layers->min_kernel_matrix, output_layers->max_kernel_matrix, output_layers->feature_matrix);//计算值
	user_nn_matrix_cpy_matrix(output_layers->softmax_feature_matrix, output_layers->feature_matrix);
	//user_snn_data_softmax(output_layers->softmax_feature_matrix);//数据归一化处理
}

//反向传播进行求取更新梯度值
//
void user_snn_bp_output_back_prior(user_snn_layers *prior_layer, user_snn_layers *output_layer) {
	user_nn_matrix   *input_feature_matrix = NULL;//
	user_nn_matrix   *input_softmax_feature_matrix = NULL;//
	user_nn_matrix   *input_thred_matrix = NULL;//
	user_nn_matrix   *input_thred_matrix_temp = NULL;//

	user_snn_output_layers  *output_layers = (user_snn_output_layers  *)output_layer->content;//获取本层池化层数据
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_softmax_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_softmax_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->softmax_feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_hidden_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else {
		return ;
	}
	//**** 使用归一化处理与未归一化处理的数据进行阈值设置
	user_nn_matrix_thred_process(output_layers->thred_matrix,output_layers->softmax_feature_matrix, output_layers->target_matrix);//计算出阈值变化趋势
	user_nn_matrix_update_thred(input_softmax_feature_matrix, output_layers->thred_matrix, output_layers->min_kernel_matrix, output_layers->max_kernel_matrix, snn_avg_vaule, snn_step_vaule);//更新阈值

	//user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
	input_thred_matrix_temp = user_nn_matrix_mult_matrix(input_feature_matrix, output_layers->thred_matrix);//矩阵乘法
	//user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
	user_nn_matrix_cpy_matrix(input_thred_matrix, input_thred_matrix_temp);//更新矩阵

	user_nn_matrix_delete(input_thred_matrix_temp);//删除矩阵
}

//反向传播进行求取更新梯度值
void user_snn_bp_hidden_back_prior(user_snn_layers *prior_layer, user_snn_layers *hidden_layer) {
	user_snn_hidden_layers  *hidden_layers = (user_snn_hidden_layers  *)hidden_layer->content;//获取本层池化层数据
	user_nn_matrix	 *input_feature_matrix = NULL;
	user_nn_matrix   *input_softmax_feature_matrix = NULL;//
	user_nn_matrix   *input_thred_matrix = NULL;//
	user_nn_matrix   *input_thred_matrix_temp = NULL;//
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_softmax_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_softmax_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->softmax_feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else {
		return;
	}
	//hidden_layers->thred_matrix 这个是复用
	user_nn_matrix_thred_process(hidden_layers->thred_matrix, hidden_layers->feature_matrix, hidden_layers->thred_matrix);//计算出阈值变化趋势

	user_nn_matrix_update_thred(input_softmax_feature_matrix, hidden_layers->thred_matrix, hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix, snn_avg_vaule, snn_step_vaule);//更新阈值

	//user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
	input_thred_matrix_temp = user_nn_matrix_mult_matrix(input_feature_matrix, hidden_layers->thred_matrix);//矩阵乘法
	//user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
	user_nn_matrix_cpy_matrix(input_thred_matrix, input_thred_matrix_temp);//更新矩阵

	user_nn_matrix_delete(input_thred_matrix_temp);//删除矩阵
}

