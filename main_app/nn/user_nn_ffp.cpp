
#include "user_nn_ffp.h"

//intput->hidden->output
//hidden=tanh(input*w1+hidden_t*w2+bias1)
//
void user_nn_ffp_hidden(user_nn_layers *prior_layer, user_nn_layers *hidden_layer) {
	user_nn_hidden_layers *hidden_layers = (user_nn_hidden_layers  *)hidden_layer->content;//获取本层的参数
	user_nn_matrix *hidden_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *hidden_bias_matrix = NULL;
	user_nn_matrix *hidden_feature_matrix = NULL;

	user_nn_matrix *intput_to_hidden_feature = NULL;

	//提取前一层的数据
	if (prior_layer->type == u_nn_layer_type_input) {
		input_feature_matrix = ((user_nn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_nn_layer_type_hidden) {
		input_feature_matrix = ((user_nn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}
	
	hidden_feature_matrix = hidden_layers->feature_matrix;//特征矩阵	
	hidden_kernel_matrix = hidden_layers->kernel_matrix;//获取前一层的疏导本层数据的权重
	hidden_bias_matrix = hidden_layers->biases_matrix;//获取偏置参数矩阵
	//Hi=act_function(np.dot(Wi,i)+bh)	
	intput_to_hidden_feature = user_nn_matrix_mult_matrix(hidden_kernel_matrix, input_feature_matrix);//np.dot(Wi,i)
#if user_nn_use_bias
	user_nn_matrix_cum_matrix(hidden_feature_matrix, intput_to_hidden_feature, hidden_bias_matrix);//权重相加+bh加上偏置参数
	user_nn_activate_matrix(hidden_feature_matrix, user_nn_nn_softmax);//采用激活函数进行激活
#else
	user_nn_matrix_cpy_matrix(hidden_feature_matrix, intput_to_hidden_feature);//拷贝矩阵
	user_nn_activate_matrix(hidden_feature_matrix, user_nn_nn_softmax);//采用激活函数进行激活
#endif // user_nn_use_bias


	user_nn_matrix_delete(intput_to_hidden_feature);//删除矩阵

}
//intput->hidden->output
//output=tanh(hidden*w2+bias2)
//
void user_nn_ffp_output(user_nn_layers *prior_layer, user_nn_layers *output_layer) {
	user_nn_output_layers *output_layers = (user_nn_output_layers  *)output_layer->content;//获取本层的参数
	user_nn_matrix *output_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *output_bias_matrix = NULL;
	user_nn_matrix *output_feature_matrix = NULL;

	user_nn_matrix *intput_to_output_feature = NULL;
	int time_index = 0;

	//提取前一层的数据
	if (prior_layer->type == u_nn_layer_type_input) {
		input_feature_matrix = ((user_nn_input_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else if (prior_layer->type == u_nn_layer_type_hidden) {
		input_feature_matrix = ((user_nn_hidden_layers *)prior_layer->content)->feature_matrix;//转化输入层对象
	}
	else {
		return;
	}

	output_kernel_matrix = output_layers->kernel_matrix;//取得权重
	output_bias_matrix = output_layers->biases_matrix;//取得偏置参数
	output_feature_matrix = output_layers->feature_matrix;//获取输出特征
	//Oh=act_function(np.dot(Wo,Hi)+bo)
	intput_to_output_feature = user_nn_matrix_mult_matrix(output_kernel_matrix, input_feature_matrix);//np.dot(Wo,Hi)	
#if user_nn_use_bias	
	user_nn_matrix_cum_matrix(output_feature_matrix, intput_to_output_feature, output_bias_matrix);//+bo加上偏置参数
	user_nn_activate_matrix(output_feature_matrix, user_nn_nn_softmax);//数据进行激活,保存本层的输出数据里面
#else
	user_nn_matrix_cpy_matrix(output_feature_matrix, intput_to_output_feature);//拷贝矩阵
	user_nn_activate_matrix(output_feature_matrix, user_nn_nn_softmax);//采用激活函数进行激活
#endif // user_nn_use_bias

	user_nn_matrix_delete(intput_to_output_feature);//删除临时矩阵	

}