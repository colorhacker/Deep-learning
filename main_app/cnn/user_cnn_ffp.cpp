
#include "user_cnn_ffp.h"

//Feedforward Pass前向传播
//本层为卷积层
//参数
//pre_layer：前一层 pool 或者 input层
//dest_layer：本层
void user_cnn_ffp_convolution(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer){
	user_nn_list_matrix   *input_feature_matrices = NULL;//输入特征数据链表
	user_nn_matrix        *input_feature_matrix = NULL;//输入特征数据  矩阵
	user_cnn_conv_layers  *conv_layers = (user_cnn_conv_layers  *)conv_layer->content;//获取本层卷积数据
	user_nn_matrix        *conv_kernel_matrix = NULL;//卷积核矩阵
	float				  *conv_bias_value = ((user_cnn_conv_layers  *)conv_layer->content)->biases_matrix->data;//偏置参数
	user_nn_list_matrix   *output_feature_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->feature_matrices;//输出特征数据链表
	user_nn_matrix        *output_feature_matrix = NULL;//输出特征数据  矩阵
	user_nn_matrix        *_conv_matrix = NULL;//卷积结果矩阵
	user_nn_matrix        *_result_matrix = NULL;//缓存矩阵

	int input_feture_index, output_feture_index;//

	//提取前一层的数据
	if (prior_layer->type == u_cnn_layer_type_input){
		input_feature_matrices = ((user_cnn_input_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		input_feature_matrices = ((user_cnn_pool_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		input_feature_matrices = ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else{
		return ;
	}
	_result_matrix = user_nn_matrix_create(conv_layers->feature_width, conv_layers->feature_height);//创建一个与输出矩阵一样缓存矩阵

	for (output_feture_index = 0; output_feture_index < conv_layers->feature_number; output_feture_index++){
		user_nn_matrix_memset(_result_matrix, 0);//清空矩阵值
		for (input_feture_index = 0; input_feture_index < conv_layers->input_feature_number; input_feture_index++){
			input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, input_feture_index);//获取特征数据
			conv_kernel_matrix = user_nn_matrices_ext_matrix(conv_layers->kernel_matrices, input_feture_index, output_feture_index);//取出输出层对应的卷积核 
			_conv_matrix = user_nn_matrix_conv2(input_feature_matrix, conv_kernel_matrix, u_nn_conv2_type_valid);//数据进行卷积操作
			user_nn_matrix_cum_matrix(_result_matrix, _result_matrix, _conv_matrix);//对上一层的所有卷积结果进行累加
			user_nn_matrix_delete(_conv_matrix);//删除缓存矩阵
		}
		output_feature_matrix = user_nn_matrices_ext_matrix_index(output_feature_matrices, output_feture_index);
		user_nn_activate_matrix_sum_constant(output_feature_matrix, _result_matrix, *conv_bias_value++, user_nn_cnn_softmax);//加上偏置参数进行函数激活
	}
	user_nn_matrix_delete(_result_matrix);//删除矩阵
}
//池化特征数据
//
void user_cnn_ffp_pooling(user_cnn_layers *prior_layer, user_cnn_layers *pool_layer){
	user_cnn_pool_layers  *pool_layers				= (user_cnn_pool_layers  *)pool_layer->content;//获取本层池化层数据
	user_nn_matrix        *pool_kernel_matrix		= ((user_cnn_pool_layers  *)pool_layer->content)->kernel_matrix;//卷积核矩阵
	user_nn_list_matrix   *output_feature_matrices	= ((user_cnn_pool_layers  *)pool_layer->content)->feature_matrices;//输出特征数据链表
	user_nn_matrix        *output_feature_matrix		= NULL;//输出特征数据  矩阵
	user_nn_list_matrix   *input_feature_matrices	= NULL;//输出特征数据链表
	user_nn_matrix        *input_feature_matrix		= NULL;//输入特征数据  矩阵

	int input_feature_index = 0;

	//提取前一层的数据
	if (prior_layer->type == u_cnn_layer_type_input){
		input_feature_matrices = ((user_cnn_input_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		input_feature_matrices = ((user_cnn_pool_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		input_feature_matrices = ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices;//转化输入层对象
	}
	else{
		return;
	}

	for (input_feature_index = 0; input_feature_index < pool_layers->input_feature_number; input_feature_index++){
		input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, input_feature_index);//取出指定位置特征数据
		output_feature_matrix = user_nn_matrices_ext_matrix_index(output_feature_matrices, input_feature_index);//取出指定位置特征数据指针
		user_nn_matrix_pooling(output_feature_matrix, input_feature_matrix, pool_kernel_matrix);//对输入数据进行池化操作
	}
}
//计算全连接层
void user_cnn_ffp_fullconnect(user_cnn_layers *prior_layer, user_cnn_layers *full_layer){
	user_nn_matrix			*full_input_feture_matrix = ((user_cnn_full_layers *)full_layer->content)->input_feature_matrix;
	user_nn_matrix			*full_feature_matrix = ((user_cnn_full_layers *)full_layer->content)->feature_matrix;//输出层的特征数据
	user_nn_matrix			*full_kernel_matrix = ((user_cnn_full_layers *)full_layer->content)->kernel_matrix;//卷积核连续矩阵
	user_nn_matrix		    *full_bias_matrix = ((user_cnn_full_layers *)full_layer->content)->biases_matrix;//偏置参数 矩阵
	user_nn_list_matrix		*input_feature_matrices		= NULL;//输入特征数据链表
	user_nn_matrix			*_output_feature_matrix		= NULL;//输出特征数据  矩阵

	//提取前一层的数据
	if (prior_layer->type == u_cnn_layer_type_input){
		user_nn_matrices_to_matrix(full_input_feture_matrix, ((user_cnn_input_layers *)prior_layer->content)->feature_matrices);//把上一层的输入到本层的特征数据转化成一个矩阵
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		user_nn_matrices_to_matrix(full_input_feture_matrix, ((user_cnn_pool_layers *)prior_layer->content)->feature_matrices);//把上一层的输入到本层的特征数据转化成一个矩阵
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		user_nn_matrices_to_matrix(full_input_feture_matrix, ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices);//把上一层的输入到本层的特征数据转化成一个矩阵
	}
	else{
		return;
	}
	//y=simoid(w*x+b)
	_output_feature_matrix = user_nn_matrix_mult_matrix(full_kernel_matrix, full_input_feture_matrix);//矩阵乘法进行数据转化
	user_nn_activate_matrix_sum_matrix(full_feature_matrix, _output_feature_matrix, full_bias_matrix, user_nn_cnn_softmax);//输出特征数据进行与偏置参数求和在进行sigmoid处理

	user_nn_matrix_delete(_output_feature_matrix);//删除矩阵
}
//计算输出层数据
void user_cnn_ffp_output(user_cnn_layers *prior_layer, user_cnn_layers *output_layer){
	user_nn_matrix			*output_input_feture_matrix = ((user_cnn_output_layers *)output_layer->content)->input_feature_matrix;
	user_nn_matrix			*output_feature_matrix = ((user_cnn_output_layers *)output_layer->content)->feature_matrix;//输出层的特征数据
	user_nn_matrix			*output_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->kernel_matrix;//卷积核连续矩阵
	user_nn_matrix		    *output_bias_matrix = ((user_cnn_output_layers *)output_layer->content)->biases_matrix;//偏置参数 矩阵
	user_nn_list_matrix		*input_feature_matrices = NULL;//输入特征数据链表
	user_nn_matrix			*_output_feature_matrix = NULL;//输出特征数据  矩阵

	//提取前一层的数据
	if (prior_layer->type == u_cnn_layer_type_input){
		user_nn_matrices_to_matrix(output_input_feture_matrix, ((user_cnn_input_layers *)prior_layer->content)->feature_matrices);//把上一层的输入到本层的特征数据转化成一个矩阵
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		user_nn_matrices_to_matrix(output_input_feture_matrix, ((user_cnn_pool_layers *)prior_layer->content)->feature_matrices);//把上一层的输入到本层的特征数据转化成一个矩阵
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		user_nn_matrices_to_matrix(output_input_feture_matrix, ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices);//把上一层的输入到本层的特征数据转化成一个矩阵
	}
	else if (prior_layer->type == u_cnn_layer_type_full){
		user_nn_matrix_cpy_matrix(output_input_feture_matrix, ((user_cnn_full_layers *)prior_layer->content)->feature_matrix);//直接拷贝更新数据即可
	}
	else{
		return;
	}
	_output_feature_matrix = user_nn_matrix_mult_matrix(output_kernel_matrix, output_input_feture_matrix);//矩阵乘法进行数据转化
	user_nn_activate_matrix_sum_matrix(output_feature_matrix, _output_feature_matrix, output_bias_matrix, user_nn_cnn_softmax);//输出特征数据进行与偏置参数求和在进行sigmoid处理
	
	user_nn_matrix_delete(_output_feature_matrix);//删除矩阵
}


