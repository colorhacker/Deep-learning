
#include "user_snn_ffp_bp.h"

void user_snn_ffp_flat(user_snn_layers *prior_layer, user_snn_layers *flat_layer) {
	user_snn_flat_layers *flat_layers = (user_snn_flat_layers  *)flat_layer->content;//获取本层的参数
	user_nn_matrix *input_feature_matrix = NULL;
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_flat) {
		input_feature_matrix = ((user_snn_flat_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_memset(flat_layers->feature_matrix, 0.0f);
	user_nn_matrix_thred_flat(input_feature_matrix, flat_layers->min_kernel_matrix, flat_layers->max_kernel_matrix, flat_layers->feature_matrix);//计算值
	user_snn_data_softmax(flat_layers->feature_matrix);//数据归一化处理
}
void user_snn_ffp_hidden(user_snn_layers *prior_layer, user_snn_layers *hidden_layer) {
	user_snn_hidden_layers *hidden_layers = (user_snn_hidden_layers  *)hidden_layer->content;//获取本层的参数
	user_nn_matrix *input_feature_matrix = NULL;
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_flat) {
		input_feature_matrix = ((user_snn_flat_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_memset(hidden_layers->feature_matrix, 0.0f);
	user_nn_matrix_thred_acc(input_feature_matrix, hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix, hidden_layers->feature_matrix);//计算值
	user_snn_data_softmax(hidden_layers->feature_matrix);//数据归一化处理
}

void user_snn_ffp_output(user_snn_layers *prior_layer, user_snn_layers *output_layer) {
	user_snn_output_layers *output_layers = (user_snn_output_layers  *)output_layer->content;//获取本层的参数
	user_nn_matrix *input_feature_matrix = NULL;
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_flat) {
		input_feature_matrix = ((user_snn_flat_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_memset(output_layers->feature_matrix,0.0f);//清空特征数据
	user_nn_matrix_thred_acc(input_feature_matrix, output_layers->min_kernel_matrix, output_layers->max_kernel_matrix, output_layers->feature_matrix);//计算值
	user_snn_data_softmax(output_layers->feature_matrix);//数据归一化处理
}

//反向传播进行求取更新梯度值
void user_snn_bp_output_back_prior(user_snn_layers *prior_layer, user_snn_layers *output_layer) {
	user_nn_matrix   *input_feature_matrix = NULL;//
	user_nn_matrix   *input_thred_matrix = NULL;//
	user_snn_output_layers  *output_layers = (user_snn_output_layers  *)output_layer->content;//获取本层池化层数据
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_flat) {
		input_feature_matrix = ((user_snn_flat_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_flat_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_hidden_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else {
		return ;
	}

	user_nn_matrix_cpy_matrix(input_thred_matrix, input_feature_matrix);//拷贝值
	//user_nn_matrix_memset(input_thred_matrix,0.0f);
	//
	//**** 使用归一化处理与未归一化处理的数据进行阈值设置
	user_nn_matrix_thred_process(output_layers->thred_matrix,output_layers->feature_matrix, output_layers->target_matrix);//计算出阈值变化趋势
	user_nn_matrix_update_thred(input_feature_matrix, input_thred_matrix, output_layers->min_kernel_matrix, output_layers->max_kernel_matrix, output_layers->thred_matrix, user_nn_snn_avg_vaule, user_nn_snn_step_vaule);//更新阈值
	user_snn_data_softmax(input_thred_matrix);
	user_nn_matrix_thred_process(input_thred_matrix, input_feature_matrix, input_thred_matrix);//计算出前一层阈值变化趋势
}

//反向传播进行求取更新梯度值
void user_snn_bp_hidden_back_prior(user_snn_layers *prior_layer, user_snn_layers *hidden_layer) {
	user_snn_hidden_layers  *hidden_layers = (user_snn_hidden_layers  *)hidden_layer->content;//获取本层池化层数据
	user_nn_matrix	 *input_feature_matrix = NULL;
	user_nn_matrix   *input_thred_matrix = NULL;//
	//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_flat) {
		input_feature_matrix = ((user_snn_flat_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_flat_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_hidden_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_cpy_matrix(input_thred_matrix, input_feature_matrix);//拷贝值
	//user_nn_matrix_memset(input_thred_matrix, 0.0f);
	user_nn_matrix_update_thred(input_feature_matrix, input_thred_matrix, hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix, hidden_layers->thred_matrix, user_nn_snn_avg_vaule, user_nn_snn_step_vaule);//更新阈值
	user_snn_data_softmax(input_thred_matrix);
	user_nn_matrix_thred_process(input_thred_matrix, input_feature_matrix, input_thred_matrix);//计算出阈值变化趋势
}
//
void user_snn_bp_flat_back_prior(user_snn_layers *prior_layer, user_snn_layers *flat_layer) {
	user_snn_flat_layers  *flat_layers = (user_snn_flat_layers  *)flat_layer->content;//获取本层池化层数据
	user_nn_matrix	 *input_feature_matrix = NULL;
	user_nn_matrix   *input_thred_matrix = NULL;//
												//提取前一层的数据
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_flat) {
		input_feature_matrix = ((user_snn_flat_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_flat_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
		input_thred_matrix = ((user_snn_hidden_layers *)prior_layer->content)->thred_matrix;//转化输入层对象
	}
	else {
		return;
	}
	user_nn_matrix_cpy_matrix(input_thred_matrix, input_feature_matrix);//拷贝值
	//user_nn_matrix_memset(input_thred_matrix, 0.0f);
	user_nn_matrix_update_thred(input_feature_matrix, input_thred_matrix, flat_layers->min_kernel_matrix, flat_layers->max_kernel_matrix, flat_layers->thred_matrix, user_nn_snn_avg_vaule, user_nn_snn_step_vaule);//更新阈值
	user_snn_data_softmax(input_thred_matrix);
	user_nn_matrix_thred_process(input_thred_matrix, input_feature_matrix, input_thred_matrix);//计算出阈值变化趋势
}

