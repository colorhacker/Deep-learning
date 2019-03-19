
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_activate.h"
#include "../cnn/user_cnn_layers.h"
#include "../cnn/user_cnn_ffp.h"
#include "../cnn/user_cnn_bp.h"
#include "../cnn/user_cnn_grads.h"
#include "../cnn/user_cnn_model.h"

//获取exe文件路径
//返回 可执行文件完整路径
char *user_cnn_model_get_exe_path(void){
	static char exeFullPath[MAX_PATH]=""; 
	if (exeFullPath[0] == 0){
		GetModuleFileName(NULL, exeFullPath, MAX_PATH);
		char *p = strrchr(exeFullPath, '\\');
		*p = 0x00;
	}
	return exeFullPath;
}
//对folder文件夹进行扫描，然后获取index位置的文件名称
//folder 文件夹名称
//index 返回文件名的 指数
//返回 null或文件名
char *user_cnn_model_search_file_name(char *folder, int index){
	static WIN32_FIND_DATA FindFileData;
	static HANDLE hFind = INVALID_HANDLE_VALUE;

	FindClose(hFind);//删除已存在的扫描句柄
	hFind = FindFirstFile(folder, &FindFileData);
	if (hFind != INVALID_HANDLE_VALUE){
		if (index == 1){
			return FindFileData.cFileName;
		}
		else{
			while (index--){
				if (FindNextFile(hFind, &FindFileData)){
					if (index == 0){
						return FindFileData.cFileName;
					}
				}
				else{
					break;
				}
			}
		}
	}
	return NULL;
}
//给定一个训练的文件夹 获取其完整路径
//files 文件夹
//index 获取第几个文件的名称
//返回 文件名或者null
char *user_cnn_model_full_path(char *files,int index){
	static char full_path[MAX_PATH];
	char *file_name = NULL;

	memset(full_path, 0, sizeof(full_path));
	sprintf(full_path, "%s/%s/*%s", user_nn_cnn_training_folder, files, user_nn_cnn_training_type);
	
	file_name = user_cnn_model_search_file_name(full_path, index);
	memset(full_path, 0, sizeof(full_path));
	if (file_name != NULL){
		sprintf(full_path, "%s/%s/%s", user_nn_cnn_training_folder, files, file_name);
		return full_path;
	}
	else{
		return NULL;
	}	
}
//把一幅图像转化为一个矩阵
//path 图像路径
//返回 矩阵对象
user_nn_matrix *user_cnn_model_obtain_image(char *path){
	user_nn_matrix *result = NULL;
	IplImage *load_image = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);//加载图像文件 采用单通道加载方式

	result = user_nn_matrix_create(load_image->width, load_image->height);//创建矩阵
	user_nn_matrix_memcpy_uchar_mult_constant(result, (unsigned char *)load_image->imageData, (float)1 / 255);//拷贝矩阵数据并且归1化数据
	cvReleaseImage(&load_image);//释放内存
	
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
//显示一个矩阵
//window_name 窗口名称
//src_matrix 矩阵对象
//返回 无
void user_cnn_model_display_matrix(char *window_name, user_nn_matrix  *src_matrix){
	CvSize cvsize = { src_matrix->width, src_matrix->height }; 
	IplImage *dest_image = cvCreateImage(cvsize, IPL_DEPTH_8U, 1);
	user_nn_matrix_uchar_memcpy((unsigned char *)dest_image->imageData, src_matrix);//更新图像数据
	cvShowImage(window_name, dest_image);//显示图像
	cvWaitKey(1);
	cvReleaseImage(&dest_image);//释放内存
}
//显示一连续矩阵
//window_name 窗口名称
//src_matrices 连续矩阵的对象
//gain 放大倍数
//返回 无
void user_cnn_model_display_matrices(char *window_name, user_nn_list_matrix  *src_matrices,int gain){ 
	user_nn_matrix *dest_matrix = user_cnn_model_matrices_gain_matrix(src_matrices, gain);//转化矩阵 并且放大
	CvSize cvsize = { dest_matrix->width, dest_matrix->height };
	IplImage *dest_image = cvCreateImage(cvsize, IPL_DEPTH_8U, 1);
	user_nn_matrix_uchar_memcpy((unsigned char *)dest_image->imageData, dest_matrix);//更新图像数据
	cvShowImage(window_name, dest_image);//显示图像
	cvWaitKey(1);
	cvReleaseImage(&dest_image);//释放内存
	user_nn_matrix_delete(dest_matrix);//删除矩阵
}
//显示特征数据
//layers 显示的模型对象
//返回 无
void user_cnn_model_display_feature(user_cnn_layers *layers){
	static int create_flags = 0;
	char windows_name[128];

	if (create_flags == 0){
		create_flags = 1;
		/*cvNamedWindow("input1", CV_WINDOW_AUTOSIZE); cvMoveWindow("input1", 100, 100);
		cvNamedWindow("conv2", CV_WINDOW_AUTOSIZE); cvMoveWindow("conv2", 300, 100);
		//cvNamedWindow("pool3", CV_WINDOW_AUTOSIZE);cvMoveWindow("pool3", 100, 100);
		cvNamedWindow("conv4", CV_WINDOW_AUTOSIZE); cvMoveWindow("conv4", 500, 100);
		//cvNamedWindow("pool5", CV_WINDOW_AUTOSIZE);cvMoveWindow("pool5", 100, 100);
		//cvNamedWindow("output6", CV_WINDOW_AUTOSIZE);cvMoveWindow("output6", 100, 100);*/
	}
	while (1){
		memset(windows_name, 0, sizeof(windows_name));
		switch (layers->type){
		case u_cnn_layer_type_null:
			break;
		case u_cnn_layer_type_input:
			sprintf(windows_name, "input%d", layers->index);
			user_cnn_model_display_matrices(windows_name, ((user_cnn_input_layers  *)layers->content)->feature_matrices, 2);//显示到指定窗口
			break;
		case u_cnn_layer_type_conv:
			sprintf(windows_name, "conv%d", layers->index);
			user_cnn_model_display_matrices(windows_name, ((user_cnn_conv_layers  *)layers->content)->feature_matrices, 2);//显示到指定窗口
			break;
		case u_cnn_layer_type_pool:
			//sprintf(windows_name, "pool%d", layers->index);
			//user_cnn_model_display_matrices(windows_name, ((user_cnn_pool_layers  *)layers->content)->feature_matrices, 2);//显示到指定窗口
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
	IplImage *load_image = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);//加载图像文件
	user_cnn_layers *cnn_input_layer = user_cnn_layers_get(layers, 1);//获取输入层
	user_nn_matrix *save_matrix = user_nn_matrices_ext_matrix_index(((user_cnn_input_layers *)cnn_input_layer->content)->feature_matrices, index - 1);//获取矩阵位置
	user_nn_matrix_memcpy_uchar_mult_constant(save_matrix, (unsigned char *)load_image->imageData, (float)1 / 255);//拷贝矩阵数据并且归1化数据
	cvReleaseImage(&load_image);//释放内存
}
//加载mnist数据至输入层返回类别数字
//mnist 连续矩阵
//mnist_index 图像矩阵位置
//layers 加载对象层
//layers_index 加载到层的第几个矩阵中
//返回 当前图像数字
void user_cnn_model_load_input_mnist(user_nn_list_matrix *mnist, int mnist_index, user_cnn_layers *layers, int layers_index) {
	user_cnn_layers *cnn_input_layer = user_cnn_layers_get(layers, 1);//获取输入层
	user_nn_matrix *save_matrix = user_nn_matrices_ext_matrix_index(((user_cnn_input_layers *)cnn_input_layer->content)->feature_matrices, layers_index - 1);//获取矩阵位置
	user_nn_matrix *mnist_matrix = user_nn_matrices_ext_matrix_index(mnist, mnist_index);
	//user_nn_matrix_cpy_matrix(save_matrix, mnist_matrix);//拷贝矩阵
	user_nn_matrix_cpy_matrix_p(save_matrix, mnist_matrix);//指向矩阵
	//user_nn_matrix_divi_constant(save_matrix,255.0);//归一化
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
void user_cnn_model_bp(user_cnn_layers *layers, user_nn_matrix *target, float alpha){
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
			user_cnn_bp_output_back_prior(layers->prior, layers, target);
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
				loss_function = (float)0.99f * loss_function + 0.01f * ((user_cnn_output_layers *)layers->content)->loss_function;
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
			return layers;//返回最大值
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

//指定文件追加字符串
void user_model_save_string(char *str){
	FILE *output_file = NULL;
	output_file = fopen("debug.txt", "a+");
	fprintf(output_file, "%s", str);
	fclose(output_file);
}
//指定文件追加int整形值
void user_model_save_int(int vaule){
	FILE *output_file = NULL;
	output_file = fopen("debug.txt", "a+");
	fprintf(output_file, "%d", vaule);
	fclose(output_file);
}
//指定文件追加浮点数值
void user_model_save_float(float vaule){
	FILE *output_file = NULL;
	output_file = fopen("debug.txt", "a+");
	fprintf(output_file, "%f", vaule);
	fclose(output_file);
}
/*
FILE *debug_file = NULL;
debug_file = fopen("model.bin", "w+");
user_cnn_input_layers  *input_layer = (user_cnn_input_layers  *)user_cnn_layers_get(layers, 1)->content;//
user_cnn_conv_layers   *conv_layer_1 = (user_cnn_conv_layers   *)user_cnn_layers_get(layers, 2)->content;//
user_cnn_pool_layers   *pool_layer_1 = (user_cnn_pool_layers   *)user_cnn_layers_get(layers, 3)->content;//
user_cnn_conv_layers   *conv_layer_2 = (user_cnn_conv_layers   *)user_cnn_layers_get(layers, 4)->content;//
user_cnn_pool_layers   *pool_layer_2 = (user_cnn_pool_layers   *)user_cnn_layers_get(layers, 5)->content;//
user_cnn_output_layers *output_layer = (user_cnn_output_layers *)user_cnn_layers_get(layers, 6)->content;//

user_nn_matrices_printf(debug_file, "卷积层1", conv_layer_1->kernel_matrices);//打印卷积核数据
user_nn_matrix_printf(debug_file,					conv_layer_1->biases_matrix);//卷积层1 偏置参数
user_nn_matrix_printf(debug_file,					pool_layer_1->kernel_matrix);
user_nn_matrices_printf(debug_file, "卷积层2", conv_layer_2->kernel_matrices);//打印卷积核数据
user_nn_matrix_printf(debug_file,					conv_layer_2->biases_matrix);//卷积层2 偏置参数
user_nn_matrix_printf(debug_file,					pool_layer_2->kernel_matrix);
user_nn_matrix_printf(debug_file,					output_layer->kernel_matrix);//打印输出层核数据
user_nn_matrix_printf(debug_file,					output_layer->biases_matrix);//打印输出层的偏置参数

fclose(debug_file);
*/