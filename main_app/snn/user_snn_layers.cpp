
#include "user_snn_layers.h"


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
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	hidden_layers->feature_width = intput_featrue_width;
	hidden_layers->feature_height = feature_number;

	hidden_layers->min_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//神经元矩阵
	hidden_layers->max_kernel_matrix = user_nn_matrix_create(intput_feature_height, hidden_layers->feature_height);//神经元矩阵

	hidden_layers->feature_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//创建输出矩阵
	hidden_layers->thred_matrix = user_nn_matrix_create(hidden_layers->feature_width, hidden_layers->feature_height);//创建变化矩阵

	user_snn_init_matrix(hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix);//初始化矩阵

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
	else if (last_layers->type == u_snn_layer_type_hidden) {
		user_snn_hidden_layers	*temp_layers = (user_snn_hidden_layers *)last_layers->content;//获取上一层 输入层的值
		intput_featrue_width = temp_layers->feature_width;//
		intput_feature_height = temp_layers->feature_height;
	}
	output_layers->feature_width = intput_featrue_width;
	output_layers->feature_height = feature_number;


	output_layers->min_kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//神经元矩阵
	output_layers->max_kernel_matrix = user_nn_matrix_create(intput_feature_height, output_layers->feature_height);//神经元矩阵

	output_layers->feature_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//创建输出矩阵
	output_layers->thred_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//创建变化矩阵
	output_layers->target_matrix = user_nn_matrix_create(output_layers->feature_width, output_layers->feature_height);//创建目标矩阵

	user_snn_init_matrix(output_layers->min_kernel_matrix, output_layers->max_kernel_matrix);//初始化矩阵

	return output_layers;
}


//数据中值处理
void user_snn_data_softmax(user_nn_matrix *src_matrix) {
	float count = (float)(src_matrix->height * src_matrix->width);
	float *src_data = src_matrix->data;
	float *max_value = user_nn_matrix_return_max_addr(src_matrix);
	if (*max_value > 1.0f) {
		user_nn_matrix_divi_constant(src_matrix, *max_value);//归一
	}
	user_nn_matrix_sum_constant(src_matrix, (count - user_nn_matrix_cum_element(src_matrix)) / count);//平均值设置为1.0f
	user_nn_matrix_divi_constant(src_matrix, 0.0001f);//除法
	user_nn_matrxi_floor(src_matrix);//取整
	user_nn_matrix_mult_constant(src_matrix, 0.0001f);//乘法
	//*max_value += count - user_nn_matrix_cum_element(src_matrix);
	//printf("%-10.6f\n", user_nn_matrix_cum_element(src_matrix));
}
//初始化阈值矩阵
//输入矩阵 最小最大矩阵
//输出 无
void user_snn_init_matrix(user_nn_matrix *min_matrix, user_nn_matrix *max_matrix) {
	int count = min_matrix->height * min_matrix->width;
	float *min_data = min_matrix->data;
	float *max_data = max_matrix->data;
	while (count--) {
		*min_data = user_nn_init_normal();
		*max_data = *min_data + user_nn_init_normal() + 1.0f;
		min_data++;
		max_data++;
	}
	//user_snn_data_softmax(min_matrix);
	//user_snn_data_softmax(max_matrix);
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
			*thred_data = snn_thred_add;
		}else if (*target_data < *src_data) {
			*thred_data = snn_thred_acc;
		}else {
			*thred_data = snn_thred_none;
		}
		src_data++;
		target_data++;
		thred_data++;
	}
}
//矩阵按照设置的阈值进行累加计算 和矩阵乘法类似 只是值是进行判断
//src_matrix 输入矩阵
//min_matrix 低阈值
//max_matrix 高阈值
//输出 返回结果矩阵
void user_nn_matrix_thred_acc(user_nn_matrix *src_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix,user_nn_matrix *output_matrix) {
	user_nn_matrix *result = NULL;//结果矩阵
	float *min_data = min_matrix->data;//
	float *src_data = src_matrix->data;//
	float *max_data = max_matrix->data;//
	float *output_data = output_matrix->data;
	if (min_matrix->width != src_matrix->height) {//矩阵乘积只有当第一个矩阵的列数=第二个矩阵的行数才有意义
		return;
	}
	if ((output_matrix->width != src_matrix->width) || (output_matrix->height != min_matrix->height)) {
		return;
	}

#if defined _OPENMP && _USER_API_OPENMP
#pragma omp parallel for 
	for (int height = 0; height < output_matrix->height; height++) {
		for (int width = 0; width < output_matrix->width; width++) {
			for (int point = 0; point < src_matrix->height; point++) {
				if ((min_data[ height * min_matrix->width + point] <= src_data[width + point*src_matrix->width]) && (src_data[width + point*src_matrix->width] <= max_data[height * max_matrix->width + point])) {
					//if ((min_data[width + point*src_matrix->width] <= src_data[height * min_matrix->width + point]) && (src_data[height * min_matrix->width + point] <= max_data[width + point*src_matrix->width])) {
					output_data[height*output_matrix->width + width] += 1.0f;//进行满足阈值结果累加
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
				if ((*min_data <= *src_data) && (*src_data <= *max_data)) {
					*output_data += 1.0f;//进行满足阈值结果累加
				}
				src_data += src_matrix->width;
				min_data++;
				max_data++;
				}
			output_data++;
			}
		}
#endif

}

//按高低进行阈值更新
//src_matrix 前一层输入数据
//src_target_matrix 前一层需要改变的目标值
//min_matrix 神经元低阈值
//max_matrix 神经元高阈值
//thred_matrix 本层输出值需要改变的目标值量化
//输出 无
void user_nn_matrix_update_thred(user_nn_matrix *src_matrix, user_nn_matrix *src_exp_matrix, user_nn_matrix *min_matrix, user_nn_matrix *max_matrix, user_nn_matrix *thred_matrix, float avg_value, float step_value) {
	float *src_exp_data = src_exp_matrix->data;
	float *src_data = src_matrix->data;//
	float *thred_data = thred_matrix->data;//
	float *min_data = min_matrix->data;//
	float *max_data = max_matrix->data;//
	
	//float avg_value = 1.0f;
	//float step_value = 0.001f;
	if (min_matrix->width != src_matrix->height) {//矩阵乘积只有当第一个矩阵的列数=第二个矩阵的行数才有意义
		return;
	}
	if ((thred_matrix->width != src_matrix->width) || (thred_matrix->height != min_matrix->height)) {
		return;
	}
#if defined _OPENMP && _USER_API_OPENMP && false
#pragma omp parallel for 
	for (int height = 0; height < src_matrix->height; height++) {
		for (int width = 0; width < min_matrix->width; width++) {
			for (int point = 0; point < min_matrix->height; point++) {
				min_data = min_matrix->data + height * min_matrix->width + point;//指向行开头
				max_data = max_matrix->data + height * max_matrix->width + point;//指向行开头
				src_data = src_matrix->data + width + point*src_matrix->width;;//指向列开头
				thred_data = thred_matrix->data + height*thred_matrix->width + width;

				if (*thred_data == snn_thred_heighten) {
					if (*src_data >= avg_value) {
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? *max_data : (*max_data + step_value);
					}
					else {
						*min_data = *min_data > *src_data ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else if (*thred_data == snn_thred_lower) {
					if (*src_data >= avg_value) {
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? (*max_data - step_value) : *max_data;
					}
					else {
						*min_data = *min_data > *src_data ? *min_data : (*min_data + step_value);
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else {

				}
			}
		}
	}
#else
	for (int height = 0; height < thred_matrix->height; height++) {
		for (int width = 0; width < thred_matrix->width; width++) {
			min_data = min_matrix->data + height * min_matrix->width;//指向行开头
			max_data = max_matrix->data + height * max_matrix->width;//指向行开头
			src_data = src_matrix->data + width;//指向列开头
			src_exp_data = src_exp_matrix->data + width;
			for (int point = 0; point < src_matrix->height; point++) {
				if (*thred_data == snn_thred_add) {
					if (*src_data >= avg_value) {
						//avg_value = *src_data;
						//在保持前一层数据输入情况不变下移动阈值
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? *max_data : (*max_data + step_value);
						//在保持本层阈值不变情况下移动输入值
						*src_exp_data = *src_data < *min_data ? (*src_exp_data + 0.1f) : *src_exp_data;
						*src_exp_data = *src_data > *max_data ? (*src_exp_data - 0.1f) : *src_exp_data;
					}
					else {
						//avg_value = *src_data;
						*min_data = *min_data > *src_data ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);

						*src_exp_data = *src_data < *min_data ? (*src_exp_data + 0.1f) : *src_exp_data;
						*src_exp_data = *src_data > *max_data ? (*src_exp_data - 0.1f) : *src_exp_data;
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else if (*thred_data == snn_thred_acc) {
					if (*src_data >= avg_value) {
						//avg_value = *src_data;
						*min_data = *min_data > avg_value ? (*min_data - step_value) : *min_data;
						*max_data = *max_data > *src_data ? (*max_data - step_value) : *max_data;

						if (*min_data < *src_data && *src_data < *max_data) {
							*src_exp_data += 0.1f;
						}
					}
					else {
						//avg_value = *src_data;
						*min_data = *min_data > *src_data ? *min_data : (*min_data + step_value);
						*max_data = *max_data > avg_value ? *max_data : (*max_data + step_value);

						if (*min_data < *src_data && *src_data < *max_data) {
							*src_exp_data -= 0.1f;
						}
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else {

				}
				src_data += src_matrix->width;
				src_exp_data += src_exp_matrix->width;
				min_data++;
				max_data++;
			}
			thred_data++;
		}
	}
#endif
}

/*
float min[] = { 0.5f,0.1f };
float max[] = { 1.6f,1.0f };

user_nn_matrix *src_matrix = user_nn_matrix_create_memset(1, 1, src);

user_nn_matrix *min_matrix = user_nn_matrix_create_memset(1, 2, min);
user_nn_matrix *max_matrix = user_nn_matrix_create_memset(1, 2, max);

user_nn_matrix *res_matrix = user_nn_matrix_create(1, 2);

user_nn_matrix_thred_acc(src_matrix, min_matrix, max_matrix, res_matrix);//

if (res_matrix != NULL) {
	user_nn_matrix_printf(NULL, res_matrix);//打印矩阵
}
else {
	printf("null\n");
}
printf("\nend");
*/