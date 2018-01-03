
#include "user_rnn_ffp.h"

//intput->hidden->output
//hidden=tanh(input*w1+hidden_t*w2+bias1)
//
void user_rnn_ffp_hidden(user_rnn_layers *prior_layer, user_rnn_layers *hidden_layer) {
	user_nn_list_matrix   *input_feature_matrices = NULL;//输入特征数据链表
	user_rnn_hidden_layers *hidden_layers = (user_rnn_hidden_layers  *)hidden_layer->content;//获取本层的参数
	user_nn_matrix *hidden_kernel_matrix = NULL;
	user_nn_matrix *intput_feature_matrix = NULL;
	user_nn_matrix *hidden_kernel_matrix_t = NULL;
	user_nn_matrix *hidden_feature_matrix_t = NULL;
	user_nn_matrix *hidden_bias_matrix = NULL;
	user_nn_matrix *hidden_feature_matrix = NULL;

	user_nn_matrix *intput_to_hidden_feature = NULL;
	user_nn_matrix *hidden_to_hidden_feature = NULL;
	int time_index = 0;

	//提取前一层的数据
	if (prior_layer->type == u_rnn_layer_type_input) {
		input_feature_matrices = ((user_rnn_input_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else if (prior_layer->type == u_rnn_layer_type_hidden) {
		input_feature_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else {
		return;
	}
	
	for (time_index = 0; time_index < hidden_layers->time_number; time_index++) {
		hidden_feature_matrix = user_nn_matrices_ext_matrix_index(hidden_layers->feature_matrices, time_index);//特征矩阵	
		hidden_kernel_matrix = hidden_layers->kernel_matrix;//获取前一层的疏导本层数据的权重
		intput_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, time_index);//获取前一层输入到本层的特征数据
		hidden_kernel_matrix_t = hidden_layers->kernel_matrix_t;//隐藏层到隐藏层的权重
		hidden_feature_matrix_t = hidden_layers->feature_matrix_t;//上个时间片隐藏层的数据
		hidden_bias_matrix = hidden_layers->biases_matrix;//获取偏置参数矩阵
		//Hi=act_function(np.dot(Wi,i)+np.dot(Wh,Ht_1)+bh)
		
		intput_to_hidden_feature = user_nn_matrix_mult_matrix(hidden_kernel_matrix, intput_feature_matrix);//np.dot(Wi,i)
		hidden_to_hidden_feature = user_nn_matrix_mult_matrix(hidden_kernel_matrix_t, hidden_feature_matrix_t);//np.dot(Wh,Ht_1)
		user_nn_matrix_cum_matrix(hidden_feature_matrix, intput_to_hidden_feature, hidden_to_hidden_feature);//权重相加
		user_nn_matrix_cum_matrix(hidden_feature_matrix, hidden_feature_matrix, hidden_bias_matrix);//+bh加上偏置参数
		user_nn_activate_matrix(hidden_feature_matrix, user_nn_rnn_softmax);//采用激活函数进行激活
		//Ht_1=Hi
		user_nn_matrix_cpy_matrix(hidden_feature_matrix_t, hidden_feature_matrix);//保存最后一个时间片的隐藏层特征值

		user_nn_matrix_delete(intput_to_hidden_feature);//删除矩阵
		user_nn_matrix_delete(hidden_to_hidden_feature);//删除矩阵
	}
}
//intput->hidden->output
//output=tanh(hidden*w2+bias2)
//
void user_rnn_ffp_output(user_rnn_layers *prior_layer, user_rnn_layers *output_layer) {
	user_nn_list_matrix   *input_feature_matrices = NULL;//输入特征数据链表
	user_rnn_output_layers *output_layers = (user_rnn_output_layers  *)output_layer->content;//获取本层的参数
	user_nn_matrix *output_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *output_bias_matrix = NULL;
	user_nn_matrix *output_feature_matrix = NULL;

	user_nn_matrix *intput_to_output_feature = NULL;
	int time_index = 0;

	//提取前一层的数据
	if (prior_layer->type == u_rnn_layer_type_input) {
		input_feature_matrices = ((user_rnn_input_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else if (prior_layer->type == u_rnn_layer_type_hidden) {
		input_feature_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else {
		return;
	}

	for (time_index = 0; time_index < output_layers->time_number; time_index++) {
		output_kernel_matrix = output_layers->kernel_matrix;//取得权重
		input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, time_index);//获取上层的输入特征数据
		output_bias_matrix = output_layers->biases_matrix;//取得偏置参数
		output_feature_matrix = user_nn_matrices_ext_matrix_index(output_layers->feature_matrices, time_index);//获取输出特征
		//Oh=act_function(np.dot(Wo,Hi)+bo)
		intput_to_output_feature = user_nn_matrix_mult_matrix(output_kernel_matrix,input_feature_matrix);//np.dot(Wo,Hi)		
		user_nn_matrix_cum_matrix(output_feature_matrix, intput_to_output_feature, output_bias_matrix);//+bo加上偏置参数
		user_nn_activate_matrix(output_feature_matrix, user_nn_rnn_softmax);//数据进行激活,保存本层的输出数据里面

		user_nn_matrix_delete(intput_to_output_feature);//删除临时矩阵	
	}
}