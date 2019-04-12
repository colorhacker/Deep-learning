
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../cnn/user_cnn_layers.h"
#include "../cnn/user_cnn_ffp.h"
#include "../cnn/user_cnn_bp.h"
#include "../cnn/user_cnn_grads.h"
#include "../cnn/user_cnn_model.h"

//把一幅图像转化为一个矩阵
//path 图像路径
//返回 矩阵对象
user_nn_matrix *user_cnn_model_obtain_image(char *path){
	user_nn_matrix *result = NULL;
	//IplImage *load_image = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);//加载图像文件 采用单通道加载方式
	//result = user_nn_matrix_create(load_image->width, load_image->height);//创建矩阵
	//user_nn_matrix_memcpy_uchar_mult_constant(result, (unsigned char *)load_image->imageData, (float)1 / 255);//拷贝矩阵数据并且归1化数据
	//cvReleaseImage(&load_image);//释放内存
	return result;
}
//垂直拼接连续矩阵
//src_matrices 连续矩阵
//返回 拼接后的矩阵
user_nn_matrix *user_cnn_model_matrices_splice(user_nn_list_matrix *src_matrices){
	//把连续的矩阵拼接成一个矩阵
	user_nn_matrix *result = NULL;
	user_nn_matrix *src_matrix = src_matrices->matrix;
	int count_matrix;//
	float *result_data = NULL;//指向对象数据
	result = user_nn_matrix_create(src_matrix->width, (src_matrix->height * src_matrices->width * src_matrices->height));//创建矩阵
	result_data = result->data;//获取数据指针
	for (count_matrix = 0; count_matrix < (src_matrices->width * src_matrices->height); count_matrix++){
		memcpy((char *)result_data, (char *)src_matrix->data, src_matrix->height * src_matrix->width * sizeof(float));//按照字节拷贝数据
		result_data += src_matrix->height * src_matrix->width;//按照float移动指针
		src_matrix = src_matrix->next;//指向下一个指针
	}
	return result;
}
//把连续矩阵转化为一个矩阵 并且放大gain倍 返回新的矩阵
//src_matries 需要转化的连续矩阵对象
//gain 放大倍数
//返回 无
user_nn_matrix  *user_cnn_model_matrices_gain_matrix(user_nn_list_matrix *src_matries, int gain){
	user_nn_matrix  *splice_matrix = NULL;
	user_nn_matrix  *gain_matrix = NULL;

	splice_matrix = user_cnn_model_matrices_splice(src_matries);//拼接矩阵
	gain_matrix = user_nn_matrix_expand_mult_constant(splice_matrix, gain, gain, (float)255);//进行放大处理

	user_nn_matrix_delete(splice_matrix);
	return gain_matrix;
}

//显示一连续矩阵
//window_name 窗口名称
//src_matrices 连续矩阵的对象
//gain 放大倍数
//返回 无
void user_cnn_model_display_matrices(char *window_name, user_nn_list_matrix  *src_matrices, int x, int y) {
	user_nn_matrix *dest_matrix = user_cnn_model_matrices_splice(src_matrices);//转化矩阵
	int width = (int)sqrt(dest_matrix->height*dest_matrix->width);
	int height = (int)sqrt(dest_matrix->height*dest_matrix->width);
	cv::Mat img(width, height, CV_32FC1, dest_matrix->data);
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	//cv::resizeWindow(window_name, width, height);
	//cv::updateWindow(win);//opengl
	//cv::startWindowThread();
	cv::moveWindow(window_name,x,y);
	cv::imshow(window_name, img);
	cv::waitKey(1);
}
//显示特征数据
//layers 显示的模型对象
//返回 无
void user_cnn_model_display_feature(user_cnn_layers *layers){
	static int create_flags = 0;
	int window_count = -1;
	char windows_name[128];

	if (create_flags == 0){
		create_flags = 1;
	}
	while (1){
		window_count++;
		memset(windows_name, 0, sizeof(windows_name));
		switch (layers->type){
		case u_cnn_layer_type_null:
			break;
		case u_cnn_layer_type_input:
			sprintf(windows_name, "input%d", layers->index);
			user_cnn_model_display_matrices(windows_name, ((user_cnn_input_layers  *)layers->content)->feature_matrices, 50 + window_count * 150, 20);//显示到指定窗口
			break;
		case u_cnn_layer_type_conv:
			sprintf(windows_name, "conv%d", layers->index);
			user_cnn_model_display_matrices(windows_name, ((user_cnn_conv_layers  *)layers->content)->feature_matrices, 50 + window_count * 150, 20);//显示到指定窗口
			break;
		case u_cnn_layer_type_pool:
			//sprintf(windows_name, "pool%d", layers->index);
			//user_cnn_model_display_matrices(windows_name, ((user_cnn_pool_layers  *)layers->content)->feature_matrices, 50 + window_count * 150,20);//显示到指定窗口
			break;
		case u_cnn_layer_type_full:
			break;
		case u_cnn_layer_type_output:
			//sprintf(windows_name, "output%d", layers->index);
			//user_cnn_model_display_matrices(windows_name, ((user_cnn_output_layers  *)layers->content)->feature_matrices, 2);//显示到指定窗口
			break;
		default:
			break;
		}

		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
}
//通过输入层的信息创建模型
//layer_infor 网络模型参数
//返回 所创建的网络模型对象
user_cnn_layers *user_cnn_model_create(int *layer_infor){
	user_cnn_layers			*cnn_layers = NULL;
	cnn_layers = user_cnn_layers_create(u_cnn_layer_type_null, 0);//创建一个空层

	while (1){
		switch (*layer_infor){
		case 'i':
			user_cnn_layers_input_create(cnn_layers, *(layer_infor + 1), *(layer_infor + 2), *(layer_infor + 3));	//输入层 输入特征数据 28x28单位矩阵 输入特征数据1个
			layer_infor += 4;
			break;
		case 'c':
			user_cnn_layers_convolution_create(cnn_layers, *(layer_infor + 1), *(layer_infor + 2), *(layer_infor + 3));//卷积层 卷积核为5x5 输出特征数据6个 
			layer_infor += 4;
			break;
		case 's':
			user_cnn_layers_pooling_create(cnn_layers, *(layer_infor + 1), *(layer_infor + 2), u_nn_pooling_type_mean);	//池化层 池化矩阵2x2
			layer_infor += 3;
			break;
		case 'f':
			user_cnn_layers_fullconnect_create(cnn_layers);//全连接
			layer_infor += 1;
			break;
		case 'o':
			user_cnn_layers_output_create(cnn_layers, *(layer_infor + 1));//创建全连接层 设置分类个数为10
			layer_infor += 2;
			return cnn_layers;
		default:
			printf("set error\n"); while (1);
			break;
		}
	}
	return NULL;
}
//加载特征数据到指定到输入特征数据中
//layers 加载对象层
//src_matrix 目标数据
//index 加载到层的第几个矩阵中
//返回 无
void user_cnn_model_load_input_feature(user_cnn_layers *layers, user_nn_matrix *src_matrix, int index){
	user_cnn_layers *cnn_input_layer = user_cnn_layers_get(layers, 1);//获取输入层
	user_nn_matrix *save_matrix = user_nn_matrices_ext_matrix_index(((user_cnn_input_layers *)cnn_input_layer->content)->feature_matrices, index - 1);//获取矩阵位置
	user_nn_matrix_cpy_matrix(save_matrix, src_matrix);//更新矩阵值
}
//加载图像数据到输入特征数据中
//layers 加载对象层
//path 加载图像路径
//index 加载到层的第几个矩阵中
//返回 无
void user_cnn_model_load_input_image(user_cnn_layers *layers, char *path, int index){
	//IplImage *load_image = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);//加载图像文件
	//user_cnn_layers *cnn_input_layer = user_cnn_layers_get(layers, 1);//获取输入层
	//user_nn_matrix *save_matrix = user_nn_matrices_ext_matrix_index(((user_cnn_input_layers *)cnn_input_layer->content)->feature_matrices, index - 1);//获取矩阵位置
	//user_nn_matrix_memcpy_uchar_mult_constant(save_matrix, (unsigned char *)load_image->imageData, (float)1 / 255);//拷贝矩阵数据并且归1化数据
	//cvReleaseImage(&load_image);//释放内存
}

//加载特征数据到指定到期望特征数据中
//layers 加载对象层
//src_matrix 目标数据
//返回 无
void user_cnn_model_load_target_feature(user_cnn_layers *layers, user_nn_matrix *src_matrix) {
	user_cnn_layers *nn_output_layer = user_cnn_model_return_layer(layers, u_cnn_layer_type_output);//获取输出层
	user_nn_matrix_cpy_matrix(((user_cnn_output_layers *)nn_output_layer->content)->target_matrix, src_matrix);
}
//正向执行一次迭代
//layers 所创建的层
//返回值 无
void user_cnn_model_ffp(user_cnn_layers *layers){
	while (1){
		switch (layers->type){
		case u_cnn_layer_type_null:
			break;
		case u_cnn_layer_type_input:
			break;
		case u_cnn_layer_type_conv:
			user_cnn_ffp_convolution(layers->prior, layers);//进行卷积计算
			break;
		case u_cnn_layer_type_pool:
			user_cnn_ffp_pooling(layers->prior, layers);//子采样处理
			break;
		case u_cnn_layer_type_output:
			user_cnn_ffp_output(layers->prior, layers);//输出层计算
			break;
		case u_cnn_layer_type_full:
			user_cnn_ffp_fullconnect(layers->prior, layers);//全连接层
			break;
		default:
			break;
		}
		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
}

//反向传播一次
//layers：层起始位置
//index：当前标签位置
//alpha：更新系数
//返回值：无
void user_cnn_model_bp(user_cnn_layers *layers,float alpha){
	//取得指向最后一层数据指针
	while (layers->next != NULL){
		layers = layers->next;
	}
	//反向计算残差
	while (1){
		switch (layers->type){
		case u_cnn_layer_type_null:
			break;
		case u_cnn_layer_type_input:
			break;
		case u_cnn_layer_type_conv:
			user_cnn_bp_convolution_back_prior(layers->prior, layers);
			break;
		case u_cnn_layer_type_pool:
			user_cnn_bp_pooling_back_prior(layers->prior, layers);
			break;
		case u_cnn_layer_type_full:
			user_cnn_bp_fullconnect_back_prior(layers->prior, layers);
			break;
		case u_cnn_layer_type_output:
			user_cnn_bp_output_back_prior(layers->prior, layers);
			break;
		default:
			break;
		}
		if (layers->prior == NULL){
			break;
		}
		else{
			layers = layers->prior;
		}
	}
	//求解权重残差值
	while (1){
		switch (layers->type){
		case u_cnn_layer_type_null:
			break;
		case u_cnn_layer_type_input:
			break;
		case u_cnn_layer_type_conv:
			user_cnn_bp_convolution_deltas_kernel(layers->prior, layers);//求解权重残差值
			user_cnn_grads_convolution(layers, alpha);//更新权重
			break;
		case u_cnn_layer_type_pool:
			break;
		case u_cnn_layer_type_full:
			user_cnn_bp_full_deltas_kernel(layers);//求解全连接层权重
			user_cnn_grads_full(layers, alpha);//更新权重
			break;
		case u_cnn_layer_type_output:
			user_cnn_bp_output_deltas_kernel(layers);//求解权重残差值
			user_cnn_grads_output(layers, alpha);//更新权重
			break;
		default:
			break;
		}
		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
}
//获取识别类别指数
//layers 获取对象层
//返回 最大值的index位置
int user_cnn_model_return_class(user_cnn_layers *layers){
	while (1){
		if (layers->type == u_cnn_layer_type_output){
			return user_nn_matrix_return_max_index(((user_cnn_output_layers *)layers->content)->feature_matrix);//返回最大值
		}
		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
	return 0xFFFF;
}
//获取loss损失值
//layers 获取对象层
//返回 损失值的大小
float user_cnn_model_return_loss(user_cnn_layers *layers){
	static float loss_function = 0;//全局变量的loss值
	while (1){
		if (layers->type == u_cnn_layer_type_output){		
			if (loss_function == 0){
				loss_function = ((user_cnn_output_layers *)layers->content)->loss_function;
			}
			else{
				loss_function = 0.99f * loss_function + 0.01f * ((user_cnn_output_layers *)layers->content)->loss_function;
			}
		}
		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
	return loss_function;
}
//从整个网络中获取一个指定层 按顺序查找
//layers 查找的对象层
//type 目标层类型
//返回 结果对象层
user_cnn_layers *user_cnn_model_return_layer(user_cnn_layers *layers, user_cnn_layer_type type){
	while (1){
		if (layers->type == type){
			return layers;//返回
		}
		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
	return NULL;
}

//显示layers所有属性配置
//layers 查找的对象层
//type 直接打印出来
//返回 结果对象层
void user_cnn_model_info_layer(user_cnn_layers *layers) {
	user_cnn_input_layers	*input_infor = NULL;
	user_cnn_conv_layers	*conv_infor = NULL;
	user_cnn_pool_layers	*pool_infor = NULL;
	user_cnn_output_layers  *output_infor = NULL;
	user_cnn_full_layers	*full_infor = NULL;
	printf("\n\n-----CNN神经网络层信息-----\n");
	while (1) {
		switch (layers->type) {
		case u_cnn_layer_type_null:
			break;
		case u_cnn_layer_type_input:
			input_infor = (user_cnn_input_layers *)layers->content;
			printf("\n第%d层,输入%d个(%d,%d)数据", layers->index,input_infor->feature_number, input_infor->feature_width, input_infor->feature_height);
			break;
		case u_cnn_layer_type_conv:
			conv_infor = (user_cnn_conv_layers *)layers->content;
			printf("\n第%d层,卷积%d个(%d,%d)神经集", layers->index, conv_infor->feature_number, conv_infor->kernel_width, conv_infor->kernel_height);
			break;
		case u_cnn_layer_type_pool:
			pool_infor = (user_cnn_pool_layers *)layers->content;
			printf("\n第%d层,池化类型%d,大小(%d,%d)", layers->index, pool_infor->pool_type, pool_infor->pool_width, pool_infor->pool_height);
			break;
		case u_cnn_layer_type_full:
			full_infor = (user_cnn_full_layers *)layers->content;
			printf("\n第%d层,全连接大小(%d,%d)", layers->index, full_infor->kernel_matrix->width, full_infor->kernel_matrix->height);
			break;
		case u_cnn_layer_type_output:
			output_infor = (user_cnn_output_layers *)layers->content;
			printf("\n第%d层,输出大小(%d,%d)\n", layers->index, output_infor->feature_matrix->width, output_infor->feature_matrix->height);
			break;
		default:
			break;
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
}

