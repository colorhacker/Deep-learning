#include "user_cnn_bp.h"

//输出层进行代价函数计算、错误值计算、数层残差计算
//1.利用output_feture_maps-1的方式得到期望值output_error_maps
//2.通过公式 1/2*(output_error_maps.*output_error_maps)/output_error_maps的列数 得到均方误差
//3.利用 output_error_maps.*(output_feture_maps.*(1-output_feture_maps))得到输出层的 output_deltas_maps
//4.利用output_deltas_maps.*output_kernel_maps 得到残差
//输出层反馈数据至上一层
//参数：
//prior_layer  上一层
//output_layer 本层
//index 设置某个位置期望值为1
//返回 无
void user_cnn_bp_output_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *output_layer){
	user_cnn_output_layers  *output_layers = (user_cnn_output_layers  *)output_layer->content;//获取本层池化层数据
	user_nn_matrix			*output_feature_matrix = ((user_cnn_output_layers *)output_layer->content)->feature_matrix;//输出特征数据  矩阵
	user_nn_matrix          *output_deltas_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_matrix;//输出层的灵敏度或者残差
	user_nn_matrix			*output_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->kernel_matrix;//卷积核连续矩阵
	user_nn_matrix          *output_error_matrix = ((user_cnn_output_layers *)output_layer->content)->error_matrix;//错误值保存
	user_nn_matrix			*output_target_matrix = ((user_cnn_output_layers *)output_layer->content)->target_matrix;//目标矩阵
	user_nn_matrix          *_feture_vector_deltas = NULL;//残差反向传播回前一层,net.fvd保存的是残差
	//计算输出层 错误值 E = 实际值 - 期望值
	user_nn_matrix_cum_matrix_mult_alpha(output_error_matrix, output_feature_matrix, output_target_matrix, -1.0f);//计算错误值
	user_nn_matrix_printf(NULL, output_target_matrix);
	user_nn_matrix_printf(NULL, output_feature_matrix);
	//计算输出层代价函数 Y = (1/2)*E^2
	
	output_layers->loss_function = user_nn_matrix_get_rmse(output_error_matrix);//代价函数，采用均方误差函数作为代价函数  
	//计算输出层灵残差 matlab公式：output_deltas_maps = output_error_maps.*output_feture_maps.*(1-output_feture_maps)  错误值*输出层数据sigmoid的导数
	user_nn_activate_matrix_d_mult_matrix(output_deltas_matrix, output_error_matrix, output_feature_matrix, user_nn_cnn_softmax);//对本层求导得到输出层的残差 
	//下面是把得到输出结果进行反向计算出输出层的残差均权重值
	user_nn_matrix_transpose(output_kernel_matrix);//交换output_kernel_maps 的 width与height 
	_feture_vector_deltas = user_nn_matrix_mult_matrix(output_kernel_matrix, output_deltas_matrix);//计算feature vector delta  ######如果不能相乘需要变化横纵大小######
	user_nn_matrix_transpose(output_kernel_matrix);//交换output_kernel_maps 的 width与height 交换回来
	//这里进行把得到的输出层残差反馈回前一层
	if (prior_layer->type == u_cnn_layer_type_input){
		//return;//输入层无残差
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		user_nn_list_matrix		*before_deltas_list = ((user_cnn_pool_layers *)prior_layer->content)->deltas_matrices;//转化输入层对象
		user_nn_matrix_to_matrices(before_deltas_list, _feture_vector_deltas);//保存残差到前一层
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		//feture_vector = feture_vector.*(output_feture_maps.*(1 - output_feture_maps))
		user_nn_list_matrix		*before_deltas_matrices = ((user_cnn_conv_layers *)prior_layer->content)->deltas_matrices;//转化输入层对象
		user_nn_list_matrix		*before_feature_matrices = ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices;//转化输入层对象	
		user_nn_matrix_to_matrices(before_deltas_matrices, _feture_vector_deltas);//保存残差到前一层
		user_nn_activate_matrices_d_mult_matrices(before_deltas_matrices, before_deltas_matrices, before_feature_matrices, user_nn_cnn_softmax);//求导
	} if (prior_layer->type == u_cnn_layer_type_full){
		user_nn_matrix		*before_deltas_matrix = ((user_cnn_full_layers *)prior_layer->content)->deltas_matrix;//转化输入层对象
		user_nn_matrix		*before_feature_matrix = ((user_cnn_full_layers *)prior_layer->content)->feature_matrix;//转化输入层对象	
		user_nn_matrix_cpy_matrix(before_deltas_matrix, _feture_vector_deltas);//直接复制 全连接层的数据是（1,N）的矩阵
		user_nn_activate_matrix_d_mult_matrix(before_deltas_matrix, before_deltas_matrix, before_feature_matrix, user_nn_cnn_softmax);//求导矩阵
	}
	else{
		//return;
	}
	user_nn_matrix_delete(_feture_vector_deltas);//删除矩阵
}
//全连接层残差计算
//prior_layer 全连接层的前一层
//full_layer 全连接层
//返回 无
void user_cnn_bp_fullconnect_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *full_layer){
	user_cnn_full_layers   *full_layers = (user_cnn_full_layers  *)full_layer->content;//获取本层池化层数据
	user_nn_matrix			*full_feature_matrix = ((user_cnn_full_layers *)full_layer->content)->feature_matrix;//输出特征数据  矩阵
	user_nn_matrix          *full_deltas_matrix = ((user_cnn_full_layers *)full_layer->content)->deltas_matrix;//输出层的灵敏度或者残差
	user_nn_matrix			*full_kernel_matrix = ((user_cnn_full_layers *)full_layer->content)->kernel_matrix;//卷积核连续矩阵
	user_nn_matrix          *_feture_vector_deltas = NULL;//残差反向传播回前一层,net.fvd保存的是残差
	//求取本层的残差权重
	user_nn_matrix_transpose(full_kernel_matrix);//交换output_kernel_maps 的 width与height 
	_feture_vector_deltas = user_nn_matrix_mult_matrix(full_kernel_matrix, full_deltas_matrix);//计算feature vector delta  ######如果不能相乘需要变化横纵大小######
	user_nn_matrix_transpose(full_kernel_matrix);//交换output_kernel_maps 的 width与height 交换回来
	//这里进行把得到的输出层残差反馈回前一层
	if (prior_layer->type == u_cnn_layer_type_input){
		//return;//输入层无残差
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		user_nn_list_matrix		*before_deltas_list = ((user_cnn_pool_layers *)prior_layer->content)->deltas_matrices;//转化输入层对象
		user_nn_matrix_to_matrices(before_deltas_list, _feture_vector_deltas);//保存残差到前一层
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		//feture_vector = feture_vector.*(output_feture_maps.*(1 - output_feture_maps))
		user_nn_list_matrix		*before_deltas_matrices = ((user_cnn_conv_layers *)prior_layer->content)->deltas_matrices;//转化输入层对象
		user_nn_list_matrix		*before_feature_matrices = ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices;//转化输入层对象	
		user_nn_matrix_to_matrices(before_deltas_matrices, _feture_vector_deltas);//保存残差到前一层
		user_nn_activate_matrices_d_mult_matrices(before_deltas_matrices, before_deltas_matrices, before_feature_matrices, user_nn_cnn_softmax);//求导
	} if (prior_layer->type == u_cnn_layer_type_full){
		user_nn_matrix		*before_deltas_matrix = ((user_cnn_full_layers *)prior_layer->content)->deltas_matrix;//转化输入层对象
		user_nn_matrix		*before_feature_matrix = ((user_cnn_full_layers *)prior_layer->content)->feature_matrix;//转化输入层对象	
		user_nn_matrix_cpy_matrix(before_deltas_matrix, _feture_vector_deltas);//直接复制 全连接层的数据是（1,N）的矩阵
		user_nn_activate_matrix_d_mult_matrix(before_deltas_matrix, before_deltas_matrix, before_feature_matrix, user_nn_cnn_softmax);//求导矩阵 得到前一层的残差
	}
	else{
		//return;
	}
	user_nn_matrix_delete(_feture_vector_deltas);//删除矩阵

}

//池化层到卷积层
//残差保存在卷积层
//1.将pool层的残差扩充至conv层大小，扩充方式采用均值扩充。
//2.使用公式：conv_feture_maps * (1 - conv_feture_maps) * pool_deltas_maps ，其中pool_deltas_maps是第一步扩充后的值 pool_deltas_maps*对前一层卷积进行求导
//本层应该是 pool层 前一层应该是卷积层
void user_cnn_bp_pooling_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *pool_layer){
	user_cnn_pool_layers	*pool_layers	  = (user_cnn_pool_layers  *)pool_layer->content;//输入特征数据链表
	user_nn_list_matrix		*pooling_deltas_matrices = ((user_cnn_pool_layers  *)pool_layer->content)->deltas_matrices;//输入特征数据链表
	user_nn_matrix          *pooling_deltas_matrix = NULL;//
	
	user_nn_list_matrix     *before_feature_matrices = NULL;
	user_nn_matrix          *before_feature_matrix = NULL;
	user_nn_list_matrix     *before_deltas_matrices = NULL;
	user_nn_matrix          *before_deltas_matrix = NULL;
	user_nn_matrix          *_deltas_matrix = NULL;
	int output_feature_index;

	if (prior_layer->type == u_cnn_layer_type_input){
		return;
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		before_feature_matrices = ((user_cnn_pool_layers  *)prior_layer->content)->feature_matrices;//
		before_deltas_matrices = ((user_cnn_pool_layers  *)prior_layer->content)->deltas_matrices;//
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		before_feature_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->feature_matrices;//
		before_deltas_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->deltas_matrices;//
	}
	else{
		return;
	}

	for (output_feature_index = 0; output_feature_index < pool_layers->feature_number; output_feature_index++){
		pooling_deltas_matrix = user_nn_matrices_ext_matrix_index(pooling_deltas_matrices, output_feature_index);//
		before_feature_matrix = user_nn_matrices_ext_matrix_index(before_feature_matrices, output_feature_index);//
		before_deltas_matrix = user_nn_matrices_ext_matrix_index(before_deltas_matrices, output_feature_index);//

		//扩充每个残差矩阵然后再进行求导
		_deltas_matrix = user_nn_matrix_expand_mult_constant(pooling_deltas_matrix, pool_layers->pool_width, pool_layers->pool_height, (float)1 / (pool_layers->pool_width * pool_layers->pool_height));//按照指定倍数扩大矩阵
		user_nn_activate_matrix_d_mult_matrix(before_deltas_matrix, _deltas_matrix, before_feature_matrix, user_nn_cnn_softmax);//对本层求导
		user_nn_matrix_delete(_deltas_matrix);//删除矩阵
	}
}
//卷积层到池化层
//把卷积层的残差 通过卷积核逆向计算到pool层
//
void user_cnn_bp_convolution_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer){
	user_cnn_conv_layers	*conv_layers		= (user_cnn_conv_layers  *)conv_layer->content;//输入特征数据链表
	user_nn_list_matrix     *conv_deltas_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_matrices;//
	user_nn_matrix          *conv_deltas_matrix = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_matrices->matrix;//
	user_nn_list_matrix     *conv_kernel_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->kernel_matrices;//
	user_nn_matrix			*conv_kernel_matrix = NULL;//

	user_nn_matrix          *before_feature_matrix = NULL;//
	user_nn_list_matrix     *before_deltas_matrices = NULL;
	user_nn_matrix          *before_deltas_matrix = NULL;//

	user_nn_matrix          *_total_matrix = NULL;
	user_nn_matrix          *_conv_matrix = NULL;
	user_nn_matrix          *_kernel_matrix = NULL;

	int input_feature_index, output_feature_index;

	if (prior_layer->type == u_cnn_layer_type_input){
		return;
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		before_feature_matrix = ((user_cnn_pool_layers  *)prior_layer->content)->feature_matrices->matrix;//
		before_deltas_matrices = ((user_cnn_pool_layers  *)prior_layer->content)->deltas_matrices;
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		before_feature_matrix = ((user_cnn_conv_layers  *)prior_layer->content)->feature_matrices->matrix;//
		before_deltas_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->deltas_matrices;
	}
	else{
		return;
	}

	_total_matrix = user_nn_matrix_create(before_feature_matrix->width, before_feature_matrix->height);//创建矩阵与前面一层相同大小矩阵

	for (input_feature_index = 0; input_feature_index < conv_layers->input_feature_number; input_feature_index++){
		user_nn_matrix_memset(_total_matrix, 0);//清空矩阵
		for (output_feature_index = 0; output_feature_index < conv_layers->feature_number; output_feature_index++){
			conv_deltas_matrix = user_nn_matrices_ext_matrix_index(conv_deltas_matrices, output_feature_index);//获取本层残差
			conv_kernel_matrix = user_nn_matrices_ext_matrix(conv_kernel_matrices, input_feature_index, output_feature_index);//获取指定位置的卷积核
			_kernel_matrix = user_nn_matrix_rotate180(conv_kernel_matrix);//卷积核旋转180°
			_conv_matrix = user_nn_matrix_conv2(conv_deltas_matrix, _kernel_matrix, u_nn_conv2_type_full);//进行卷积操作
			user_nn_matrix_cum_matrix(_total_matrix, _total_matrix, _conv_matrix);//数据累加
			user_nn_matrix_delete(_conv_matrix);//删除矩阵
			user_nn_matrix_delete(_kernel_matrix);//删除矩阵
		}
		before_deltas_matrix = user_nn_matrices_ext_matrix_index(before_deltas_matrices, input_feature_index);//提取残差指针
		user_nn_matrix_cpy_matrix(before_deltas_matrix, _total_matrix);//更新残差值
	}
	user_nn_matrix_delete(_total_matrix);//删除矩阵
}

//求解需要更新的权重值
void user_cnn_bp_convolution_deltas_kernel(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer){
	user_cnn_conv_layers  *conv_layers				= (user_cnn_conv_layers  *)conv_layer->content;//获取本层卷积数据
	user_nn_list_matrix   *conv_deltas_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_matrices;
	user_nn_matrix        *conv_deltas_matrix = NULL;
	user_nn_list_matrix	  *conv_deltas_kernel_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_kernel_matrices;//
	user_nn_matrix		  *conv_deltas_kernel_matrix = NULL;//
	float				  *conv_deltas_bias			= ((user_cnn_conv_layers  *)conv_layer->content)->deltas_biases_matrix->data;//指向残差的偏置参数
	user_nn_list_matrix   *before_feature_matrices = NULL;//输入特征数据  
	user_nn_matrix        *before_feature_matrix = NULL;//输入特征数据 
	user_nn_matrix        *_result_matrix = NULL;//缓存矩阵
	user_nn_matrix        *_deltas_matrix = NULL;//缓存矩阵
	
	int count_conv_maps, count_input_maps;//本层特征

	//提取前一层的特征数据
	if (prior_layer->type == u_cnn_layer_type_input){
		before_feature_matrices = ((user_cnn_input_layers *)prior_layer->content)->feature_matrices;
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		before_feature_matrices = ((user_cnn_pool_layers *)prior_layer->content)->feature_matrices;
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		before_feature_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->feature_matrices;
	}
	else{

	}
	for (count_conv_maps = 0; count_conv_maps < conv_layers->feature_number; count_conv_maps++){
		conv_deltas_matrix = user_nn_matrices_ext_matrix_index(conv_deltas_matrices, count_conv_maps);//获取一个残差
		for (count_input_maps = 0; count_input_maps < conv_layers->input_feature_number; count_input_maps++){
			before_feature_matrix = user_nn_matrices_ext_matrix_index(before_feature_matrices, count_input_maps);//提取前一层的特征数据
			conv_deltas_kernel_matrix = user_nn_matrices_ext_matrix(conv_deltas_kernel_matrices, count_input_maps, count_conv_maps);//提取残差权重对应的指针
			_deltas_matrix = user_nn_matrix_rotate180(before_feature_matrix);//将前一层的特征数据旋转180°
			_result_matrix = user_nn_matrix_conv2(_deltas_matrix, conv_deltas_matrix, u_nn_conv2_type_valid);//进行卷积操作
			user_nn_matrix_cpy_matrix(conv_deltas_kernel_matrix, _result_matrix);//这里保存的是残差与权重乘积
			user_nn_matrix_delete(_deltas_matrix);//删除矩阵
			user_nn_matrix_delete(_result_matrix);//删除矩阵
		}
		*conv_deltas_bias++ = user_nn_matrix_cum_element(conv_deltas_matrix);//求和残差作为残差偏置参数
	}
}
//求解全连接层的更新权重值
void user_cnn_bp_full_deltas_kernel(user_cnn_layers *full_layer){
	user_nn_matrix			*full_input_feture_matrix	= ((user_cnn_full_layers *)full_layer->content)->input_feature_matrix;
	user_nn_matrix          *full_deltas_matrix			= ((user_cnn_full_layers *)full_layer->content)->deltas_matrix;//输出层的灵敏度或者残差
	user_nn_matrix          *full_deltas_kernel_matrix	= ((user_cnn_full_layers *)full_layer->content)->deltas_kernel_matrix;//输出层的灵敏度或者残差
	user_nn_matrix          *_grads_matrix = NULL;

	// output_deltas_maps*input_feture_maps
	user_nn_matrix_transpose(full_input_feture_matrix);//交换output_kernel_maps 的 width与height 还有数据
	_grads_matrix = user_nn_matrix_mult_matrix(full_deltas_matrix, full_input_feture_matrix);//获取输出层的误差
	user_nn_matrix_transpose(full_input_feture_matrix);//交换output_kernel_maps 的 width与height
	user_nn_matrix_cpy_matrix(full_deltas_kernel_matrix, _grads_matrix);//
	user_nn_matrix_delete(_grads_matrix);//删除矩阵
}
//求解输出层的更新权重值
void user_cnn_bp_output_deltas_kernel(user_cnn_layers *output_layer){
	user_nn_matrix			*output_input_feture_matrix = ((user_cnn_output_layers *)output_layer->content)->input_feature_matrix;
	user_nn_matrix          *output_deltas_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_matrix;//输出层的灵敏度或者残差
	user_nn_matrix          *output_deltas_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_kernel_matrix;//输出层的灵敏度或者残差
	user_nn_matrix          *_grads_matrix = NULL;

	// output_deltas_maps*input_feture_maps
	user_nn_matrix_transpose(output_input_feture_matrix);//交换output_kernel_maps 的 width与height 还有数据
	_grads_matrix = user_nn_matrix_mult_matrix(output_deltas_matrix, output_input_feture_matrix);//获取输出层的误差
	user_nn_matrix_transpose(output_input_feture_matrix);//交换output_kernel_maps 的 width与height
	user_nn_matrix_cpy_matrix(output_deltas_kernel_matrix, _grads_matrix);//
	user_nn_matrix_delete(_grads_matrix);//删除矩阵
}