
#include "user_cnn_layers.h"

//返回指定层
user_cnn_layers *user_cnn_layers_get(user_cnn_layers *dest, int index){
	while (index--){
		if (dest->next != NULL){
			dest = dest->next;
		}else{
		}
	}
	return dest;
}

//创建一个层
//参数：
//type：层类型
//index：指数
//返回 创建后的层
user_cnn_layers *user_cnn_layers_create(user_cnn_layer_type type,int index){
	user_cnn_layers *cnn_layers = NULL;

	cnn_layers = (user_cnn_layers *)malloc(sizeof(user_cnn_layers));//分配内存
	cnn_layers->prior	= NULL;//指向上一层
	cnn_layers->type	= type;//层的类型
	cnn_layers->index	= index;//指数
	cnn_layers->content = NULL;//指向内容
	cnn_layers->next	= NULL;//指向下一层

	return cnn_layers;
}
//删除层
void user_cnn_layers_delete(user_cnn_layers *layers){
	if (layers != NULL){
		if (layers->content != NULL){
			free(layers->content);
		}
		free(layers);
	}
}
//创建输入层
//参数
//width：输入数据的宽度
//height：输入数据的高度
//maps：输入数据的数量
//返回：成功或失败
user_cnn_input_layers *user_cnn_layers_input_create(user_cnn_layers *cnn_layers, int feature_width, int feature_height, int feature_number){
	user_cnn_layers			*last_layers = cnn_layers;
	user_cnn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//轮询查找cnn_layers空对象
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_input, last_layers->index + 1);//创建输入层 输入层的指数为前一层+1
	last_layers->next->content = malloc(sizeof(user_cnn_input_layers));//分配内存输入层的对象空间
	input_layers = (user_cnn_input_layers *)last_layers->next->content;//转化当前层的值 用于设置参数

	input_layers->feature_width		= feature_width;//设置特征数据的宽度
	input_layers->feature_height	= feature_height;//设置特征数据的高度
	input_layers->feature_number	= feature_number;//设置输入数据的个数
	input_layers->feature_matrices	= user_nn_matrices_create(1, input_layers->feature_number, input_layers->feature_width, input_layers->feature_height);//创建本层的特征数据矩阵 

	return input_layers;
}

//创建卷积层
//参数
//outputmaps：输出图像数量
//kernelsize：卷积核大小
//返回 
user_cnn_conv_layers *user_cnn_layers_convolution_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height ,int feature_number){
	user_cnn_layers			*last_layers	= cnn_layers;
	user_cnn_conv_layers	*conv_layers	= NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//轮询查找cnn_layers空对象
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_conv, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_cnn_conv_layers));//分配卷积层内存输入对象空间
	conv_layers = (user_cnn_conv_layers *)last_layers->next->content;//本卷积层对象获取

	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers	= (user_cnn_input_layers *)last_layers->content;//获取上一层 输入层的值
		conv_layers->feature_width				= temp_layers->feature_width - kernel_width + 1;//计算当前层的数据的对象宽度 公式：卷积高度=输入高度-卷积核+1
		conv_layers->feature_height				= temp_layers->feature_height - kernel_height + 1;//计算当前层的数据对象高度   公式：卷积高度=输入高度-卷积核+1
		conv_layers->input_feature_number		= temp_layers->feature_number;//上一层的输出数据个数为本层的输入数据个数
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//如果前面一层是池化层
		user_cnn_pool_layers	*temp_layers	= (user_cnn_pool_layers *)last_layers->content;//转化对象
		conv_layers->feature_width				= temp_layers->feature_width - kernel_width + 1;//计算当前层的数据的对象宽度 公式：卷积高度=输入高度-卷积核+1
		conv_layers->feature_height				= temp_layers->feature_height - kernel_height + 1;//计算当前层的数据对象高度   公式：卷积高度=输入高度-卷积核+1
		conv_layers->input_feature_number		= temp_layers->feature_number;//上一层的输出数据个数为本层的输入数据个数
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers	= (user_cnn_conv_layers *)last_layers->content;
		conv_layers->feature_width				= temp_layers->feature_width - kernel_width + 1;//计算当前层的数据的对象宽度 公式：卷积高度=输入高度-卷积核+1
		conv_layers->feature_height				= temp_layers->feature_height - kernel_height + 1;//计算当前层的数据对象高度   公式：卷积高度=输入高度-卷积核+1
		conv_layers->input_feature_number		= temp_layers->feature_number;//上一层的输出数据个数为本层的输入数据个数
	}
	else{
		return NULL;
	}

	conv_layers->feature_number			= feature_number;	//设置输出数据个数
	conv_layers->kernel_width			= kernel_width;	//本层为卷积层  设置卷积核的宽度
	conv_layers->kernel_height			= kernel_height;//本层为卷积层  设置卷积核的高度
	conv_layers->biases_matrix			= user_nn_matrix_create(1, conv_layers->feature_number);//添加本层的偏置参数 参数个数与输出数据个数一致
	conv_layers->feature_matrices		= user_nn_matrices_create(1, conv_layers->feature_number, conv_layers->feature_width, conv_layers->feature_height);//创建保存本层的特征数据矩阵 个数就是输出特征数据的个数
	conv_layers->kernel_matrices		= user_nn_matrices_create(conv_layers->input_feature_number, conv_layers->feature_number, conv_layers->kernel_width, conv_layers->kernel_height);//创建卷积核，本层每个输出特征数据与上一层的特征数据都有一个对应卷积核，因此数量就是本层特征数*上层特征数。我们在这里把本层特征作为连续矩阵的行，上层作为列进行创建。
	conv_layers->deltas_matrices		= user_nn_matrices_create(1, conv_layers->feature_number, conv_layers->feature_width, conv_layers->feature_height);//创建本层的残差 大小与本层的特征一致
	conv_layers->deltas_kernel_matrices = user_nn_matrices_create(conv_layers->input_feature_number, conv_layers->feature_number, conv_layers->kernel_width, conv_layers->kernel_height);//残差对卷积核的导数，残差对前一层特征数据的卷积结果
	conv_layers->deltas_biases_matrix	= user_nn_matrix_create(1, conv_layers->feature_number);//添加残差误差 

	//本层的总输出参数个数 = (float)conv->outputmaps * conv->kernel_width * conv->kernel_height;//计算出输出总参数大小  --- 用于初始化卷积核
	//本层的总输入参数个数 = (float)conv->inputmaps  * conv->kernel_width * conv->kernel_height;//计算出输入总参数大小  --- 用于初始化卷积核
	//简化后：(conv->outputmaps + conv->inputmaps) * conv->kernel_width * conv->kernel_height
	user_nn_matrices_init_vaule(conv_layers->kernel_matrices, conv_layers->feature_number, conv_layers->input_feature_number*conv_layers->kernel_width*conv_layers->kernel_height);//对卷积核进行初始化

	return conv_layers;
}

//创建池化层
//参数
//scale：池化大小
//返回 
user_cnn_pool_layers *user_cnn_layers_pooling_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height, user_nn_pooling_type pool_type){
	user_cnn_layers		 *last_layers = cnn_layers;
	user_cnn_pool_layers *pool_layers = NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//轮询查找cnn_layers空对象
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_pool, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_cnn_pool_layers));//分配卷积层内存输入对象空间
	pool_layers = (user_cnn_pool_layers *)last_layers->next->content;//本卷积层对象获取

	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers	= (user_cnn_input_layers *)last_layers->content;//获取上一层 输入层的值
		pool_layers->feature_width				= temp_layers->feature_width / kernel_width;//
		pool_layers->feature_height				= temp_layers->feature_height / kernel_height;//
		pool_layers->input_feature_number		= temp_layers->feature_number;//上一层的输出数据个数为本层的输入数据个数
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//如果前面一层是池化层
		user_cnn_pool_layers	*temp_layers	= (user_cnn_pool_layers *)last_layers->content;//转化对象
		pool_layers->feature_width				= temp_layers->feature_width / kernel_width;//
		pool_layers->feature_height				= temp_layers->feature_height / kernel_height;//
		pool_layers->input_feature_number		= temp_layers->feature_number;//上一层的输出数据个数为本层的输入数据个数
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers	= (user_cnn_conv_layers *)last_layers->content;
		pool_layers->feature_width				= temp_layers->feature_width / kernel_width;//
		pool_layers->feature_height				= temp_layers->feature_height / kernel_height;//
		pool_layers->input_feature_number		= temp_layers->feature_number;//上一层的输出数据个数为本层的输入数据个数
	}
	else{
		return NULL;
	}

	pool_layers->feature_number		= pool_layers->input_feature_number;//输出和输入一致
	pool_layers->pool_width			= kernel_width;//卷积核大小
	pool_layers->pool_height		= kernel_height;//卷积核大小
	pool_layers->pool_type			= pool_type;//池化方式 平均值 最大值
	pool_layers->kernel_matrix		= user_nn_matrix_create(pool_layers->pool_width, pool_layers->pool_height);//创建池化层的矩阵
	pool_layers->feature_matrices	= user_nn_matrices_create(1, pool_layers->feature_number, pool_layers->feature_width, pool_layers->feature_height);//创建本层的特征数据矩阵 池化层特征数据数量为输入特征数据大小
	pool_layers->deltas_matrices	= user_nn_matrices_create(1, pool_layers->feature_number, pool_layers->feature_width, pool_layers->feature_height);;//残差需要反向传播进行赋值

	user_nn_matrix_memset(pool_layers->kernel_matrix, (float)1 / (pool_layers->pool_width * pool_layers->pool_height));//初始均值化 池化矩阵数据

	return pool_layers;
}
//创建全连接层
//参数
//count：分类个数
//返回 成功或失败
user_cnn_full_layers *user_cnn_layers_fullconnect_create(user_cnn_layers *cnn_layers){
	user_cnn_layers			*last_layers = cnn_layers;
	user_cnn_full_layers	*full_layers = NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//轮询查找cnn_layers空对象
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_full, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_cnn_full_layers));//分配卷积层内存输入对象空间
	full_layers = (user_cnn_full_layers *)last_layers->next->content;//本全连接层对象获取

	//全连接应该是上一层的数据输入总和
	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers = (user_cnn_input_layers *)last_layers->content;//获取上一层 输入层的值
		full_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		full_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//上层特征向量保存到本层
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//如果前面一层是池化层
		user_cnn_pool_layers	*temp_layers = (user_cnn_pool_layers *)last_layers->content;//转化对象
		full_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		full_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//上层特征向量保存到本层
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers = (user_cnn_conv_layers *)last_layers->content;
		full_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		full_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//上层特征向量保存到本层
	}
	else{
		return NULL;
	}
	//设置全连接层的偏置参数
	full_layers->biases_matrix			= user_nn_matrix_create(1, full_layers->feature_number);//添加N个偏置参数 可以使用softmat回归的偏置参数
	full_layers->feature_matrix			= user_nn_matrix_create(1, full_layers->feature_number);//创建输出值 分类个数
	full_layers->kernel_matrix			= user_nn_matrix_create(full_layers->feature_number, full_layers->feature_number);//全连接层的权重值
	full_layers->deltas_matrix			= user_nn_matrix_create(1, full_layers->feature_number);//保存残差
	full_layers->deltas_kernel_matrix	= user_nn_matrix_create(full_layers->feature_number, full_layers->feature_number);//本层残差对上层的卷积结果ΔW

	user_nn_matrix_init_vaule(full_layers->kernel_matrix, full_layers->feature_number, full_layers->feature_number);//初始化全连接的权重值

	return full_layers;
}

//创建输出层
//参数
//count：分类个数
//返回 成功或失败
user_cnn_output_layers *user_cnn_layers_output_create(user_cnn_layers *cnn_layers, int class_number){
	user_cnn_layers			*last_layers = cnn_layers;
	user_cnn_output_layers	*output_layers	= NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//轮询查找cnn_layers空对象
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_output, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_cnn_output_layers));//分配卷积层内存输入对象空间
	output_layers = (user_cnn_output_layers *)last_layers->next->content;//本全连接层对象获取

	//输出作为全连接层那么输出的特征数据总个数为前一层的特征矩阵元素总和
	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers = (user_cnn_input_layers *)last_layers->content;//获取上一层 输入层的值
		output_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//上层特征向量保存到本层
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//如果前面一层是池化层
		user_cnn_pool_layers	*temp_layers = (user_cnn_pool_layers *)last_layers->content;//转化对象
		output_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//上层特征向量保存到本层
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers = (user_cnn_conv_layers *)last_layers->content;
		output_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//上层特征向量保存到本层
	}
	else if (last_layers->type == u_cnn_layer_type_full){
		user_cnn_full_layers	*temp_layers = (user_cnn_full_layers *)last_layers->content;
		output_layers->feature_number = temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number);//上层特征向量保存到本层
	}
	else{
		return NULL;
	}

	output_layers->class_number			= class_number;//分类个数
	output_layers->loss_function		= 0.0f;//代价函数
	//设置全连接层的偏置参数
	output_layers->biases_matrix		= user_nn_matrix_create(1, output_layers->class_number);//添加N个偏置参数 可以使用softmat回归的偏置参数
	output_layers->feature_matrix		= user_nn_matrix_create(1, output_layers->class_number);//创建输出值 分类个数
	output_layers->kernel_matrix		= user_nn_matrix_create(output_layers->feature_number, output_layers->class_number);//创建输出层的kenerl模板
	output_layers->error_matrix			= user_nn_matrix_create(1, output_layers->class_number);//错误值保存矩阵
	output_layers->target_matrix		= user_nn_matrix_create(1, output_layers->class_number);//错误值保存矩阵
	output_layers->deltas_matrix		= user_nn_matrix_create(1, output_layers->class_number);//保存残差
	output_layers->deltas_kernel_matrix = user_nn_matrix_create(output_layers->feature_number, output_layers->class_number);//本层残差对上层的卷积结果ΔW

	user_nn_matrix_init_vaule(output_layers->kernel_matrix, output_layers->feature_number, output_layers->class_number);//初始化输出层kernel模板
	
	return output_layers;
}

