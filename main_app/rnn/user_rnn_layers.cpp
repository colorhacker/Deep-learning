
#include "user_rnn_layers.h"

//返回指定层
user_rnn_layers *user_rnn_layers_get(user_rnn_layers *dest, int index) {
	while (index--) {
		if (dest->next != NULL) {
			dest = dest->next;
		}
		else {
		}
	}
	return dest;
}

//创建一个层
//参数：
//type：层类型
//index：指数
//返回 创建后的层
user_rnn_layers *user_rnn_layers_create(user_rnn_layer_type type, int index) {
	user_rnn_layers *rnn_layers = NULL;

	rnn_layers = (user_rnn_layers *)malloc(sizeof(user_rnn_layers));//分配内存
	rnn_layers->prior = NULL;//指向上一层
	rnn_layers->type = type;//层的类型
	rnn_layers->index = index;//指数
	rnn_layers->content = NULL;//指向内容
	rnn_layers->next = NULL;//指向下一层

	return rnn_layers;
}
//删除层
void user_rnn_layers_delete(user_rnn_layers *layers) {
	if (layers != NULL) {
		if (layers->content != NULL) {
			free(layers->content);
		}
		free(layers);
	}
}
//创建输入层
//参数
//feature_width：输入数据的宽度
//feature_height：输入数据的高度
//feature_number：输入数据的数量
//返回：成功或失败
user_rnn_input_layers *user_rnn_layers_input_create(user_rnn_layers *rnn_layers, int feature_width, int feature_height, int time_number) {
	user_rnn_layers			*last_layers = rnn_layers;
	user_rnn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找rnn_layers空对象
	}
	last_layers->next = user_rnn_layers_create(u_rnn_layer_type_input, last_layers->index + 1);//创建输入层 输入层的指数为前一层+1
	last_layers->next->content = malloc(sizeof(user_rnn_input_layers));//分配内存输入层的对象空间
	input_layers = (user_rnn_input_layers *)last_layers->next->content;//转化当前层的值 用于设置参数

	input_layers->feature_width = feature_width;//设置特征数据的宽度
	input_layers->feature_height = feature_height;//设置特征数据的高度
	input_layers->time_number = time_number;//设置输入数据的个数
	input_layers->deltas_matrices = user_nn_matrices_create(1, input_layers->time_number, input_layers->feature_width, input_layers->feature_height);//下一层反馈回来的残差
	input_layers->feature_matrices = user_nn_matrices_create(1, input_layers->time_number, input_layers->feature_width, input_layers->feature_height);//创建本层的特征数据矩阵 

	return input_layers;
}
//创建隐藏层
//参数
//width：输入数据的宽度
//height：输入数据的高度
//返回 成功或失败
user_rnn_hidden_layers *user_rnn_layers_hidden_create(user_rnn_layers *rnn_layers,int feature_number, int time_number) {
	user_rnn_layers			*last_layers = rnn_layers;
	user_rnn_hidden_layers	*hidden_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找rnn_layers空对象
	}
	last_layers->next = user_rnn_layers_create(u_rnn_layer_type_hidden, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_rnn_hidden_layers));//分配空间
	hidden_layers = (user_rnn_hidden_layers *)last_layers->next->content;//本全连接层对象获取

	if (last_layers->type == u_rnn_layer_type_input) {
		user_rnn_input_layers	*temp_layers = (user_rnn_input_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_rnn_layer_type_hidden) {
		user_rnn_hidden_layers	*temp_layers = (user_rnn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->time_number = time_number;//设置时间片
	hidden_layers->feature_width  = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//动态计算权重矩阵大小
	hidden_layers->kernel_matrix_t = user_nn_matrix_create(hidden_layers->feature_height, hidden_layers->feature_height);//计算时间隐含层到隐含层的权重
	hidden_layers->biases_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//添加偏置参数

	hidden_layers->deltas_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);
	hidden_layers->deltas_kernel_matrix_t = user_nn_matrix_create(hidden_layers->feature_height, hidden_layers->feature_height);
	hidden_layers->deltas_biases_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);

	hidden_layers->deltas_matrix_t = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);
	hidden_layers->deltas_matrices = user_nn_matrices_create(1, hidden_layers->time_number, hidden_layers->feature_width, hidden_layers->feature_height);//下一层反馈回来的残差
	hidden_layers->feature_matrix_t = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//保存上个时间片的隐含层数据
	hidden_layers->feature_matrices = user_nn_matrices_create(1, hidden_layers->time_number, hidden_layers->feature_width, hidden_layers->feature_height);//输出层个数

	user_nn_matrix_init_vaule(hidden_layers->kernel_matrix, hidden_layers->time_number, hidden_layers->time_number);//初始化全连接的权重值
	user_nn_matrix_init_vaule(hidden_layers->kernel_matrix_t, hidden_layers->time_number, hidden_layers->time_number);//初始化全连接的权重值
	user_nn_matrix_init_vaule(hidden_layers->biases_matrix, hidden_layers->time_number, hidden_layers->time_number);//初始化全连接的权重值

	return hidden_layers;
}
//创建输出层
//参数
//count：分类个数
//返回 成功或失败
user_rnn_output_layers *user_rnn_layers_output_create(user_rnn_layers *rnn_layers, int feature_number, int time_number) {
	user_rnn_layers			*last_layers = rnn_layers;
	user_rnn_output_layers	*output_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找rnn_layers空对象
	}
	last_layers->next = user_rnn_layers_create(u_rnn_layer_type_output, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_rnn_output_layers));//分配卷积层内存输入对象空间
	output_layers = (user_rnn_output_layers *)last_layers->next->content;//本全连接层对象获取
	 //以下是当前层的参数
	if (last_layers->type == u_rnn_layer_type_input) {
		user_rnn_input_layers	*temp_layers = (user_rnn_input_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_rnn_layer_type_hidden) {
		user_rnn_hidden_layers	*temp_layers = (user_rnn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->time_number = time_number;//设置当前层的特征数据个数
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;

	output_layers->loss_function = 0.0f;
	output_layers->kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//全连接层的权重值
	output_layers->biases_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//添加N个偏置参数 可以使用softmat回归的偏置参数
	
	output_layers->feature_matrices		= user_nn_matrices_create(1, output_layers->time_number, output_layers->feature_width, output_layers->feature_height);//
	output_layers->target_matrices		= user_nn_matrices_create(1, output_layers->time_number, output_layers->feature_width, output_layers->feature_height);//
	output_layers->error_matrix	= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//添加错误值

	output_layers->deltas_matrices		= user_nn_matrices_create(1, output_layers->time_number, output_layers->feature_width, output_layers->feature_height);//保存残差
	output_layers->deltas_kernel_matrix	= user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//本层残差对上层的结果ΔW
	output_layers->deltas_biases_matrix	= user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//

	user_nn_matrix_init_vaule(output_layers->kernel_matrix, output_layers->time_number, output_layers->time_number);//初始化全连接的权重值
	user_nn_matrix_init_vaule(output_layers->biases_matrix, output_layers->time_number, output_layers->time_number);//初始化全连接的权重值

	return output_layers;
}

