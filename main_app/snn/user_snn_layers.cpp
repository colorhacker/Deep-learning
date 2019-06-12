
#include "user_snn_layers.h"

//返回指定层
user_snn_layers *user_snn_layers_get(user_snn_layers *dest, int index) {
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
user_snn_layers *user_snn_layers_create(user_snn_layer_type type, int index) {
	user_snn_layers *snn_layers = NULL;

	snn_layers = (user_snn_layers *)malloc(sizeof(user_snn_layers));//分配内存
	snn_layers->prior = NULL;//指向上一层
	snn_layers->type = type;//层的类型
	snn_layers->index = index;//指数
	snn_layers->content = NULL;//指向内容
	snn_layers->next = NULL;//指向下一层

	return snn_layers;
}
//删除层
void user_snn_layers_delete(user_snn_layers *layers) {
	if (layers != NULL) {
		if (layers->content != NULL) {
			if (layers->type == u_snn_layer_type_input) {
				user_nn_matrix_delete(((user_snn_input_layers *)layers->content)->thred_matrix);
				user_nn_matrix_delete(((user_snn_input_layers *)layers->content)->feature_matrix);
			}
			else if (layers->type == u_snn_layer_type_flat) {
				user_nn_matrix_delete(((user_snn_flat_layers *)layers->content)->thred_kernel_matrix);
				user_nn_matrix_delete(((user_snn_flat_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_flat_layers *)layers->content)->thred_matrix);
			}
			else if (layers->type == u_snn_layer_type_hidden) {
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_kernel_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_matrix);
			}
			else if (layers->type == u_snn_layer_type_output) {
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_kernel_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->feature_matrix);
				user_nn_matrix_delete(((user_snn_hidden_layers *)layers->content)->thred_matrix);
			}
			free(layers->content);
		}
		free(layers);
	}
}
//删除层所有
void user_snn_layers_all_delete(user_snn_layers *layers) {
	user_snn_layers *layer = layers;
	user_snn_layers *layer_next = NULL;
	while (layer != NULL) {
		layer_next = layer->next;
		user_snn_layers_delete(layer);//删除当前矩阵
		layer = layer_next;//更新矩阵
	}
}
//创建输入层
//参数
//feature_width：输入数据的宽度
//feature_height：输入数据的高度
//feature_number：输入数据的数量
//返回：成功或失败
user_snn_input_layers *user_snn_layers_input_create(user_snn_layers *nn_layers, int feature_width, int feature_height) {
	user_snn_layers			*last_layers = nn_layers;
	user_snn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找nn_layers空对象
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_input, last_layers->index + 1);//创建输入层 输入层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_snn_input_layers));//分配内存输入层的对象空间
	input_layers = (user_snn_input_layers *)last_layers->next->content;//转化当前层的值 用于设置参数

	input_layers->feature_width = feature_width;//设置特征数据的宽度
	input_layers->feature_height = feature_height;//设置特征数据的高度
	input_layers->feature_matrix = user_nn_matrix_create(input_layers->feature_width, input_layers->feature_height);//创建本层的特征数据矩阵
	input_layers->thred_matrix = user_nn_matrix_create(input_layers->feature_width, input_layers->feature_height);//创建本层的特征数据矩阵
	//input_layers->feature_matrix = NULL;//预留指向数据地址

	return input_layers;
}
//创建平层
//参数
//width：输入数据的宽度
//height：输入数据的高度
//返回 成功或失败
user_snn_flat_layers *user_snn_layers_flat_create(user_snn_layers *snn_layers) {
	user_snn_layers			*last_layers = snn_layers;
	user_snn_flat_layers	*flat_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找nn_layers空对象
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_flat, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_snn_flat_layers));//分配空间
	flat_layers = (user_snn_flat_layers *)last_layers->next->content;//本全连接层对象获取

	if (last_layers->type == u_snn_layer_type_input) {
		user_snn_input_layers	*temp_layers = (user_snn_input_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_flat) {
		user_snn_flat_layers	*temp_layers = (user_snn_flat_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	flat_layers->feature_width = intput_featrue_width;
	flat_layers->feature_height = intput_feature_height;

	flat_layers->thred_kernel_matrix = user_nn_matrix_create(flat_layers->feature_width, flat_layers->feature_height);//神经元矩阵

	flat_layers->feature_matrix = user_nn_matrix_create(flat_layers->feature_width, flat_layers->feature_height);//创建输出矩阵
	flat_layers->thred_matrix = user_nn_matrix_create(flat_layers->feature_width, flat_layers->feature_height);//创建变化矩阵

	user_snn_init_matrix(flat_layers->thred_kernel_matrix);//初始化矩阵

	return flat_layers;
}
//创建隐藏层
//参数
//width：输入数据的宽度
//height：输入数据的高度
//返回 成功或失败
user_snn_hidden_layers *user_snn_layers_hidden_create(user_snn_layers *snn_layers, int feature_number) {
	user_snn_layers			*last_layers = snn_layers;
	user_snn_hidden_layers	*hidden_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找nn_layers空对象
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_hidden, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_snn_hidden_layers));//分配空间
	hidden_layers = (user_snn_hidden_layers *)last_layers->next->content;//本全连接层对象获取

	if (last_layers->type == u_snn_layer_type_input) {
		user_snn_input_layers	*temp_layers = (user_snn_input_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_flat) {
		user_snn_flat_layers	*temp_layers = (user_snn_flat_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->feature_width = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->thred_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//神经元矩阵

	hidden_layers->feature_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//创建输出矩阵
	hidden_layers->thred_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//创建变化矩阵

	user_snn_init_matrix(hidden_layers->thred_kernel_matrix);//初始化矩阵

	return hidden_layers;
}
//创建输出层
//参数
//count：分类个数
//返回 成功或失败
user_snn_output_layers *user_snn_layers_output_create(user_snn_layers *nn_layers, int feature_number) {
	user_snn_layers			*last_layers = nn_layers;
	user_snn_output_layers	*output_layers = NULL;
	int intput_featrue_width = 0;
	int intput_feature_height = 0;

	while (last_layers->next != NULL) {
		last_layers = last_layers->next;//轮询查找nn_layers空对象
	}
	last_layers->next = user_snn_layers_create(u_snn_layer_type_output, last_layers->index + 1);//创建卷积层 卷积层的指数为前一层+1
	last_layers->next->prior = last_layers;//指向前一层
	last_layers->next->content = malloc(sizeof(user_snn_output_layers));//分配卷积层内存输入对象空间
	output_layers = (user_snn_output_layers *)last_layers->next->content;//本全连接层对象获取
																		//以下是当前层的参数
	if (last_layers->type == u_snn_layer_type_input) {
		user_snn_input_layers	*temp_layers = (user_snn_input_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_flat) {
		user_snn_flat_layers	*temp_layers = (user_snn_flat_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;

	output_layers->loss_function = 0.0f;
	output_layers->thred_kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//神经元矩阵

	output_layers->feature_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//创建输出矩阵
	output_layers->thred_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//创建变化矩阵
	output_layers->target_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//创建目标矩阵

	user_snn_init_matrix(output_layers->thred_kernel_matrix);//初始化矩阵

	return output_layers;
}


//数据中值处理
void user_snn_data_softmax(user_nn_matrix *src_matrix) {
	float ave_value = user_nn_matrix_cum_element(src_matrix) / (float)(src_matrix->height * src_matrix->width);
	user_nn_matrix_sub_constant(src_matrix, ave_value);//平均值设置为0.0f
}
//初始化阈值矩阵
//输入矩阵 最小最大矩阵
//输出 无
void user_snn_init_matrix(user_nn_matrix *thred_matrix) {
	int count = thred_matrix->height * thred_matrix->width;
	float *thred_data = thred_matrix->data;
	while (count--) {
		*thred_data++ = user_nn_init_uniform();
	}
}

//升降值计算
//src_matrix 输出矩阵
//target_matrix 目标矩阵
//返回 结果矩阵
void user_nn_matrix_thred_process(user_nn_matrix *thred_matrix,user_nn_matrix *src_matrix, user_nn_matrix *target_matrix) {
	int count = src_matrix->width * src_matrix->height;
	float *src_data = src_matrix->data;
	float *target_data = target_matrix->data;
	float *thred_data = thred_matrix->data;
	while (count--) {
		if (*target_data > *src_data) {
			*thred_data = user_nn_snn_thred_add;
		}else if (*target_data < *src_data) {
			*thred_data = user_nn_snn_thred_acc;
		}else {
			*thred_data = user_nn_snn_thred_none;
		}
		src_data++;
		target_data++;
		thred_data++;
	}
}

//矩阵通过阈值进行输出
//src_matrix 输入矩阵
//min_matrix 低阈值
//max_matrix 高阈值
//输出 返回结果矩阵
void user_nn_matrix_thred_flat(user_nn_matrix *src_matrix, user_nn_matrix *thred_matrix, user_nn_matrix *output_matrix) {
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *output_data = output_matrix->data;

	for (int count = 0; count < src_matrix->height * src_matrix->width; count++) {
		if ((0.0f < *src_data) && (*src_data <= *thred_data)) {
			*output_data += 1.0f;
		}
		if ((*thred_data <= *src_data) && (*src_data < 0.0f)) {
			*output_data -= 1.0f;
		}
		src_data++;
		thred_data++;
		output_data++;
	}
}
//矩阵按照设置的阈值进行累加计算 和矩阵乘法类似 只是值是进行判断
//src_matrix 输入矩阵
//min_matrix 低阈值
//max_matrix 高阈值
//输出 返回结果矩阵
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *thred_matrix,user_nn_matrix *output_matrix) {
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *output_data = output_matrix->data;
	if (thred_matrix->width != src_matrix->height) {//矩阵乘积只有当第一个矩阵的列数=第二个矩阵的行数才有意义
		return;
	}
	if ((output_matrix->width != src_matrix->width) || (output_matrix->height != thred_matrix->height)) {
		return;
	}

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for 
	for (int height = 0; height < output_matrix->height; height++) {
		for (int width = 0; width < output_matrix->width; width++) {
			for (int point = 0; point < src_matrix->height; point++) {
				if ((0.0f < src_data[width + point*src_matrix->width]) && (src_data[width + point*src_matrix->width] <= thred_data[height * thred_matrix->width + point])) {
					output_data[height*output_matrix->width + width] += 1.0f;
				}
				if ((thred_data[height * thred_matrix->width + point] <= src_data[width + point*src_matrix->width]) && (src_data[width + point*src_matrix->width] < 0.0f)) {
					output_data[height*output_matrix->width + width] -= 1.0f;
				}

			}
		}
	}
#else
	for (int height = 0; height < output_matrix->height; height++) {
		for (int width = 0; width < output_matrix->width; width++) {
			min_data = min_matrix->data + height * min_matrix->width;//指向行开头
			max_data = max_matrix->data + height * max_matrix->width;//指向行开头
			src_data = src_matrix->data + width;//指向列开头
			for (int point = 0; point < src_matrix->height; point++) {
				if ((0.0f < *src_data) && (*src_data <= *thred_data)) {
					*output_data += 1.0f;
				}
				if ((*thred_data <= *src_data) && (*src_data < 0.0f)) {
					*output_data -= 1.0f;
				}
				src_data += src_matrix->width;
				thred_data++;
				output_data++;
				}
			output_data++;
			}
		}
#endif

}
//矩阵阈值进行更新
//src_matrix 输入矩阵
//min_matrix 低阈值
//max_matrix 高阈值
//输出 返回结果矩阵
void user_nn_matrix_update_flat(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *thred_matrix,user_nn_matrix *target_matrix ,float step_value){
	float *src_exp_data = src_exp_matrix->data;
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *target_data = target_matrix->data;//
	
	for (int count = 0; count < src_matrix->height * src_matrix->width; count++) {
		if (*target_data == user_nn_snn_thred_add) {//增加输出值
			if ((0.0f < *thred_data) && (*thred_data < *src_data)) {
				*thred_data += step_value;
				*src_exp_data = *src_data <= *thred_data ? *src_exp_data : (*src_exp_data - user_nn_snn_add_value);
			}
			if ((*thred_data < *src_data) && (*src_data < 0.0f)) {
				*thred_data += step_value;
				*src_exp_data = *src_data >= *thred_data ? (*src_exp_data - user_nn_snn_add_value) : *src_exp_data;
			}
		}
		else if (*target_data == user_nn_snn_thred_acc) {//减少输出值
			if ((0.0f < *src_data) && (*src_data < *thred_data)) {
				*thred_data -= step_value;
				*src_exp_data = *src_data <= *thred_data ? (*src_exp_data + user_nn_snn_add_value) : *src_exp_data;
			}
			if ((*src_data < *thred_data) && (*thred_data < 0.0f)) {
				*thred_data -= step_value;
				*src_exp_data = *src_data >= *thred_data ? *src_exp_data : (*src_exp_data + user_nn_snn_add_value);
			}
		}
		else {

		}
		src_data++;
		src_exp_data++;
		target_data++;
		thred_data++;
	}
}
//按高低进行阈值更新
//src_matrix 前一层输入数据
//src_exp_matrix 前一层的目标值
//src_target_matrix 前一层需要改变的目标值
//min_matrix 神经元低阈值
//max_matrix 神经元高阈值
//thred_matrix 本层输出值需要改变的目标值量化
//输出 无
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *thred_matrix, user_nn_matrix *target_matrix, float step_value) {
	float *src_exp_data = src_exp_matrix->data;
	float *src_data = src_matrix->data;//
	float *target_data = target_matrix->data;//
	float *thred_data = thred_matrix->data;//
	
	//float avg_value = 1.0f;
	//float step_value = 0.001f;
	if (thred_matrix->width != src_matrix->height) {//矩阵乘积只有当第一个矩阵的列数=第二个矩阵的行数才有意义
		return;
	}
	if ((target_matrix->width != src_matrix->width) || (target_matrix->height != thred_matrix->height)) {
		return;
	}
	for (int height = 0; height < target_matrix->height; height++) {
		for (int width = 0; width < target_matrix->width; width++) {
			thred_data = thred_matrix->data + height * thred_matrix->width;//指向行开头
			src_data = src_matrix->data + width;//指向列开头
			src_exp_data = src_exp_matrix->data + width;
			for (int point = 0; point < src_matrix->height; point++) {
				if (*target_data == user_nn_snn_thred_add) {//增加输出值
					if ((0.0f < *thred_data) && (*thred_data < *src_data)) {
						*thred_data += step_value;
						*src_exp_data = *src_data <= *thred_data ? *src_exp_data : (*src_exp_data - user_nn_snn_add_value);
					}
					if ((*thred_data < *src_data) && (*src_data < 0.0f)) {
						*thred_data += step_value;
						*src_exp_data = *src_data >= *thred_data ? (*src_exp_data - user_nn_snn_add_value) : *src_exp_data;
					}
				}
				else if (*target_data == user_nn_snn_thred_acc) {//减少输出值
					if ((0.0f < *src_data) && (*src_data < *thred_data)) {
						*thred_data -= step_value;
						*src_exp_data = *src_data <= *thred_data ? (*src_exp_data + user_nn_snn_add_value) : *src_exp_data;
					}
					if ((*src_data < *thred_data) && (*thred_data < 0.0f)) {
						*thred_data -= step_value;
						*src_exp_data = *src_data >= *thred_data ? *src_exp_data : (*src_exp_data + user_nn_snn_add_value);
					}
				}
				else {

				}
				src_data += src_matrix->width;
				src_exp_data += src_exp_matrix->width;
				thred_data++;
			}
			thred_data++;
		}
	}
}
