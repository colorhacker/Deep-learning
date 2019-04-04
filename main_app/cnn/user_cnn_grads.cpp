
#include "user_cnn_grads.h"

//更新权重与偏置参数
//conv_layer 更新卷积层的对象
//alpha 更新系数
//返回 无
void user_cnn_grads_convolution(user_cnn_layers *conv_layer, float alpha){
	user_cnn_conv_layers  *conv_layers					= (user_cnn_conv_layers  *)conv_layer->content;//获取本层卷积数据
	user_nn_list_matrix	  *conv_kernel_matrices			= ((user_cnn_conv_layers  *)conv_layer->content)->kernel_matrices;
	user_nn_matrix		  *conv_kernel_matrix			= NULL;//卷积核模板
	float				  *conv_biases					= ((user_cnn_conv_layers  *)conv_layer->content)->biases_matrix->data;//本层偏置参数
	user_nn_list_matrix	  *conv_deltas_kernel_matrices	= ((user_cnn_conv_layers  *)conv_layer->content)->deltas_kernel_matrices;
	user_nn_matrix		  *conv_deltas_kernel_matrix	= NULL;//
	float				  *conv_deltas_biases			= ((user_cnn_conv_layers  *)conv_layer->content)->deltas_biases_matrix->data;//指向残差的更新值

	int output_feature_index, input_feature_index;//本层特征个数

	for (output_feature_index = 0; output_feature_index < conv_layers->feature_number; output_feature_index++){
		for (input_feature_index = 0; input_feature_index < conv_layers->input_feature_number; input_feature_index++){
			conv_kernel_matrix		  = user_nn_matrices_ext_matrix(conv_kernel_matrices, input_feature_index, output_feature_index);//取出指定位置特征数据
			conv_deltas_kernel_matrix = user_nn_matrices_ext_matrix(conv_deltas_kernel_matrices, input_feature_index, output_feature_index);//取出指定位置特征数据
			//conv_kernel_maps = conv_kernel_maps - alpha * conv_deltas_kernel_maps
			user_nn_matrix_sum_matrix_mult_alpha(conv_kernel_matrix, conv_deltas_kernel_matrix, -1.0f * alpha);
		}
		//conv_bias = conv_bias - alpha * conv_deltas_bias
		*conv_biases++ = *conv_biases - *conv_deltas_biases++ *alpha;
	}
}
//更新输出层的权值与偏置参数
//full_layer 更新全连接层的对象
//alpha 更新系数
//返回 无
void user_cnn_grads_full(user_cnn_layers *full_layer, float alpha){
	user_nn_matrix			*full_bias_matrix			= ((user_cnn_full_layers *)full_layer->content)->biases_matrix;//输出层的卷积模板
	user_nn_matrix          *full_kernel_matrix			= ((user_cnn_full_layers *)full_layer->content)->kernel_matrix;//输出层的卷积模板
	user_nn_matrix          *full_deltas_matrix			= ((user_cnn_full_layers *)full_layer->content)->deltas_matrix;//输出层的灵敏度或者残差
	user_nn_matrix          *full_deltas_kernel_matrix	= ((user_cnn_full_layers *)full_layer->content)->deltas_kernel_matrix;//输出层的灵敏度或者残差
	//output_kernel_maps = output_kernel_maps - alpha * output_grads_maps
	user_nn_matrix_sum_matrix_mult_alpha(full_kernel_matrix, full_deltas_kernel_matrix, -1.0f * alpha);
	//output_bias_maps = output_bias_maps - alpha * output_deltas_maps;
	user_nn_matrix_sum_matrix_mult_alpha(full_bias_matrix, full_deltas_matrix, -1.0f * alpha);
}
//更新输出层的权值与偏置参数
//output_layer 更新全输出层的对象
//alpha 更新系数
//返回 无
void user_cnn_grads_output(user_cnn_layers *output_layer, float alpha){
	user_nn_matrix			*output_bias_matrix = ((user_cnn_output_layers *)output_layer->content)->biases_matrix;//输出层的卷积模板
	user_nn_matrix          *output_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->kernel_matrix;//输出层的卷积模板
	user_nn_matrix          *output_deltas_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_matrix;//输出层的灵敏度或者残差
	user_nn_matrix          *output_deltas_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_kernel_matrix;//输出层的灵敏度或者残差
	//output_kernel_maps = output_kernel_maps - alpha * output_grads_maps
	user_nn_matrix_sum_matrix_mult_alpha(output_kernel_matrix, output_deltas_kernel_matrix, -1.0f * alpha);
	//output_bias_maps = output_bias_maps - alpha * output_deltas_maps;
	user_nn_matrix_sum_matrix_mult_alpha(output_bias_matrix, output_deltas_matrix, -1.0f * alpha);

}