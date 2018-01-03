
#include "user_rnn_bptt.h"


//反向传播进行求取更新梯度值
//
float user_rnn_bp_output_back_prior(user_rnn_layers *prior_layer, user_rnn_layers *output_layer) {
	user_nn_list_matrix   *input_feature_matrices = NULL;//输入特征数据链表
	user_nn_list_matrix	  *input_deltas_matrices = NULL;
	user_rnn_output_layers  *output_layers = (user_rnn_output_layers  *)output_layer->content;//获取本层池化层数据
	user_nn_matrix *target_feature_matrix = NULL;
	user_nn_matrix *output_feature_matrix = NULL;
	user_nn_matrix *error_matrix = NULL;

	user_nn_matrix *deltas_biase_matrix = NULL;
	user_nn_matrix *deltas_kernel_matrix = NULL;
	user_nn_matrix *deltas_matrix = NULL;//残差数据
	user_nn_matrix *input_feature_matrix = NULL;//获取输入特征

	user_nn_matrix *deltas_kernel_matrix_temp = NULL;

	user_nn_matrix *output_kernel_matrix = NULL;
	user_nn_matrix *input_deltas_matrix = NULL;
	user_nn_matrix *input_deltas_matrix_temp = NULL;
	float *loss_vaule = NULL;
	int time_index = 0;
	//提取前一层的数据
	if (prior_layer->type == u_rnn_layer_type_input) {
		input_feature_matrices = ((user_rnn_input_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
		input_deltas_matrices = ((user_rnn_input_layers *)prior_layer->content)->deltas_matrices;
	}
	else if (prior_layer->type == u_rnn_layer_type_hidden) {
		input_feature_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
		input_deltas_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->deltas_matrices;//获取上一层的残差权重
	}
	else {
		return 0;
	}

	output_layers->loss_function = 0;//损失函数清零
	for (time_index = output_layers->time_number - 1; time_index >= 0; time_index--) {
		error_matrix = output_layers->error_matrix;//获取错误矩阵
		target_feature_matrix = user_nn_matrices_ext_matrix_index(output_layers->target_matrices, time_index);//获取目标矩阵值
		output_feature_matrix = user_nn_matrices_ext_matrix_index(output_layers->feature_matrices,time_index);//获取特征矩阵
		loss_vaule = &output_layers->loss_function;//获取损失函数
		//(Lo, Er) = output_loss_error(Oh, Ta)
		user_nn_matrix_cum_matrix_mult_alpha(error_matrix, output_feature_matrix, target_feature_matrix, -1.0f);//减法
		*loss_vaule = *loss_vaule + user_nn_matrix_get_rms(error_matrix);//获取军方误差
		//dOh=np.multiply(Er,act_function_d(Oh))
		deltas_matrix = user_nn_matrices_ext_matrix_index(output_layers->deltas_matrices, time_index);//获取残差矩阵
		user_nn_activate_matrix_d(output_feature_matrix, user_nn_rnn_softmax);//求导输出数据
		user_nn_matrix_poit_mult_matrix(deltas_matrix, error_matrix, output_feature_matrix);//残差=输出数据*错误值
		//ΔWo=ΔWo+np.dot(dOh,Hi.T)
		deltas_kernel_matrix = output_layers->deltas_kernel_matrix;//残差矩阵
		input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, time_index);//获取特征矩阵
		user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
		deltas_kernel_matrix_temp = user_nn_matrix_mult_matrix(deltas_matrix, input_feature_matrix);//矩阵乘法
		user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
		user_nn_matrix_cum_matrix(deltas_kernel_matrix, deltas_kernel_matrix, deltas_kernel_matrix_temp);//累加权重更新值 ΔWo
		//Δbo=Δbo+dOh
		deltas_biase_matrix = output_layers->deltas_biases_matrix;
		user_nn_matrix_cum_matrix(deltas_biase_matrix, deltas_biase_matrix, deltas_matrix);//累加偏置参数更新值Δbo
		//dHh=np.dot(Wo.T,dOh)#残差返回前一层
		output_kernel_matrix = output_layers->kernel_matrix;//获取权值参数
		input_deltas_matrix = user_nn_matrices_ext_matrix_index(input_deltas_matrices, time_index);//获取残差指针
		user_nn_matrix_transpose(output_kernel_matrix);//矩阵转置
		input_deltas_matrix_temp = user_nn_matrix_mult_matrix(output_kernel_matrix, deltas_matrix);//矩阵乘法
		user_nn_matrix_transpose(output_kernel_matrix);//矩阵转置
		user_nn_matrix_cpy_matrix(input_deltas_matrix, input_deltas_matrix_temp);//保存残差

		user_nn_matrix_delete(deltas_kernel_matrix_temp);//删除矩阵
		user_nn_matrix_delete(input_deltas_matrix_temp);//删除矩阵
	}
	return *loss_vaule;
}

//反向传播进行求取更新梯度值
void user_rnn_bp_hidden_back_prior(user_rnn_layers *prior_layer, user_rnn_layers *hidden_layer) {
	user_nn_list_matrix   *input_feature_matrices = NULL;//输入特征数据链表
	user_nn_list_matrix	  *input_deltas_matrices = NULL;
	user_rnn_hidden_layers  *hidden_layers = (user_rnn_hidden_layers  *)hidden_layer->content;//获取本层池化层数据
	user_nn_matrix *hidden_deltas_matrix = NULL;
	user_nn_matrix *hidden_feature_matrix = NULL;
	user_nn_matrix *hidden_deltas_matrix_t = NULL;

	user_nn_matrix *deltas_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *deltas_kernel_matrix_temp = NULL;

	user_nn_matrix *deltas_kernel_matrix_t = NULL;
	user_nn_matrix *input_feature_matrix_t = NULL;
	user_nn_matrix *deltas_kernel_matrix_t_temp = NULL;

	user_nn_matrix *deltas_biases_matrix = NULL;

	user_nn_matrix *kernel_matrix_t = NULL;
	user_nn_matrix *hidden_deltas_matrix_t_temp = NULL;

	user_nn_matrix *kernel_matrix = NULL;
	user_nn_matrix *input_deltas_matrix = NULL;
	user_nn_matrix *input_deltas_matrix_temp = NULL;
	int time_index = 0;
	//提取前一层的数据
	if (prior_layer->type == u_rnn_layer_type_input) {
		input_feature_matrices = ((user_rnn_input_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
		input_deltas_matrices = ((user_rnn_input_layers *)prior_layer->content)->deltas_matrices;
	}
	else if (prior_layer->type == u_rnn_layer_type_hidden) {
		input_feature_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
		input_deltas_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->deltas_matrices;//获取上一层的残差权重
	}
	else {
		return;
	}

	for (time_index = hidden_layers->time_number - 1;time_index >= 0; time_index--) {
		hidden_deltas_matrix = user_nn_matrices_ext_matrix_index(hidden_layers->deltas_matrices, time_index);//获取残差矩阵
		hidden_feature_matrix= user_nn_matrices_ext_matrix_index(hidden_layers->feature_matrices, time_index);//获取输出特征
		hidden_deltas_matrix_t = hidden_layers->deltas_matrix_t;
		//dHh=np.multiply(dOh,act_function_d(Hi))+dHt_1
		user_nn_activate_matrix_d(hidden_feature_matrix, user_nn_rnn_softmax);//求导输出数据
		user_nn_matrix_poit_mult_matrix(hidden_deltas_matrix, hidden_deltas_matrix, hidden_feature_matrix);//残差=输出数据*错误值
		user_nn_matrix_cum_matrix(hidden_deltas_matrix, hidden_deltas_matrix, hidden_deltas_matrix_t);//得到本层总残差值
		//ΔWi=ΔWi+np.dot(dHh,i.T)
		deltas_kernel_matrix = hidden_layers->deltas_kernel_matrix;//获取残差矩阵
		input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, time_index);//获取输入特征数据
		user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
		deltas_kernel_matrix_temp = user_nn_matrix_mult_matrix(hidden_deltas_matrix, input_feature_matrix);//矩阵乘法
		user_nn_matrix_transpose(input_feature_matrix);//矩阵转置
		user_nn_matrix_cum_matrix(deltas_kernel_matrix, deltas_kernel_matrix, deltas_kernel_matrix_temp);//累加输入层到隐含层的残差变化值
		//ΔWh = ΔWh + np.dot(dHh, dHt_1.T)
		deltas_kernel_matrix_t = hidden_layers->deltas_kernel_matrix_t;//获取残差矩阵
		user_nn_matrix_transpose(hidden_deltas_matrix_t);//矩阵转置
		deltas_kernel_matrix_t_temp = user_nn_matrix_mult_matrix(hidden_deltas_matrix, hidden_deltas_matrix_t);//矩阵乘法
		user_nn_matrix_transpose(hidden_deltas_matrix_t);//矩阵转置
		user_nn_matrix_cum_matrix(deltas_kernel_matrix_t, deltas_kernel_matrix_t, deltas_kernel_matrix_t_temp);//累加输入层到隐含层的残差变化值
		//Δbh=Δbh+dHh
		deltas_biases_matrix = hidden_layers->deltas_biases_matrix;
		user_nn_matrix_cum_matrix(deltas_biases_matrix, deltas_biases_matrix, hidden_deltas_matrix);//累加偏置参数残差变化值
		//dHt_1=np.dot(Wh.T,dHh)
		kernel_matrix_t = hidden_layers->kernel_matrix_t;
		user_nn_matrix_transpose(kernel_matrix_t);//矩阵转置
		hidden_deltas_matrix_t_temp = user_nn_matrix_mult_matrix(kernel_matrix_t, hidden_deltas_matrix);//矩阵乘法
		user_nn_matrix_transpose(kernel_matrix_t);//矩阵转置
		user_nn_matrix_cpy_matrix(hidden_deltas_matrix_t, hidden_deltas_matrix_t_temp);//保存残差
		//dIi = np.dot(Wi.T, dHh)#残差返回前一层
		kernel_matrix = hidden_layers->kernel_matrix;
		input_deltas_matrix = user_nn_matrices_ext_matrix_index(input_deltas_matrices, time_index);//获取输入残差矩阵
		user_nn_matrix_transpose(kernel_matrix);//矩阵转置
		input_deltas_matrix_temp = user_nn_matrix_mult_matrix(kernel_matrix, hidden_deltas_matrix);//矩阵乘法
		user_nn_matrix_transpose(kernel_matrix);//矩阵转置
		user_nn_matrix_cpy_matrix(input_deltas_matrix, input_deltas_matrix_temp);//保存残差

		user_nn_matrix_delete(deltas_kernel_matrix_temp);//删除矩阵
		user_nn_matrix_delete(deltas_kernel_matrix_t_temp);//删除矩阵
		user_nn_matrix_delete(hidden_deltas_matrix_t_temp);//删除矩阵
		user_nn_matrix_delete(input_deltas_matrix_temp);//删除矩阵
	}
}

