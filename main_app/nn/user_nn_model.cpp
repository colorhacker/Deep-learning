
#include "user_nn_model.h"
#include "user_nn_ffp.h"
#include "user_nn_bp.h"
#include "user_nn_grads.h"

//通过输入层的信息创建模型
//layer_infor 网络模型参数
//返回 所创建的网络模型对象
user_nn_layers *user_nn_model_create(int *layer_infor) {
	user_nn_layers			*nn_layers = NULL;
	nn_layers = user_nn_layers_create(u_nn_layer_type_null, 0);//创建一个空层

	while (1) {
		switch (*layer_infor) {
		case 'i':
			user_nn_layers_input_create(nn_layers, *(layer_infor + 1), *(layer_infor + 2));	//输入层
			layer_infor += 3;
			break;
		case 'h':
			user_nn_layers_hidden_create(nn_layers, *(layer_infor + 1));//隐含层
			layer_infor += 2;
			break;
		case 'o':
			user_nn_layers_output_create(nn_layers, *(layer_infor + 1));//输出层
			layer_infor += 2;
			return nn_layers;
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
//返回 无
void user_nn_model_load_input_feature(user_nn_layers *layers, user_nn_matrix *src_matrix) {
	user_nn_layers *nn_input_layer = user_nn_layers_get(layers, 1);//获取输入层
	//user_nn_matrix_cpy_matrix(((user_nn_input_layers *)nn_input_layer->content)->feature_matrix, src_matrix);
	user_nn_matrix_memcpy(((user_nn_input_layers *)nn_input_layer->content)->feature_matrix, src_matrix->data);
	//((user_nn_input_layers *)nn_input_layer->content)->feature_matrix = src_matrix;
}
//加载特征数据到指定到期望特征数据中
//layers 加载对象层
//src_matrix 目标数据
//返回 无
void user_nn_model_load_target_feature(user_nn_layers *layers, user_nn_matrix *src_matrix) {
	user_nn_layers *nn_output_layer = user_nn_model_return_layer(layers, u_nn_layer_type_output);//获取输入层
	//user_nn_matrix_cpy_matrix(((user_nn_output_layers *)nn_output_layer->content)->target_matrix, src_matrix);
	user_nn_matrix_memcpy(((user_nn_output_layers *)nn_output_layer->content)->target_matrix, src_matrix->data);
	//((user_nn_output_layers *)nn_output_layer->content)->target_matrix = src_matrix;
}
//正向执行一次迭代
//layers 所创建的层
//返回值 无
void user_nn_model_ffp(user_nn_layers *layers) {
	while (1) {
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			break;
		case u_nn_layer_type_hidden:
			user_nn_ffp_hidden(layers->prior, layers);//子采样处理
			break;
		case u_nn_layer_type_output:
			user_nn_ffp_output(layers->prior, layers);//输出层计算
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

//反向传播一次
//layers：层起始位置
//index：当前标签位置
//alpha：更新系数
//返回值：无
void user_nn_model_bp(user_nn_layers *layers, float alpha) {
	//取得指向最后一层数据指针
	while (layers->next != NULL) {
		layers = layers->next;
	}
	//反向计算残差
	while (1) {
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			break;
		case u_nn_layer_type_hidden:
			user_nn_bp_hidden_back_prior(layers->prior, layers);
			break;
		case u_nn_layer_type_output:
			user_nn_bp_output_back_prior(layers->prior, layers);
			break;
		default:
			break;
		}
		if (layers->prior == NULL) {
			break;
		}
		else {
			layers = layers->prior;
		}
	}
	//求解权重残差值
	while (1) {
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			break;
		case u_nn_layer_type_hidden:
			user_nn_grads_hidden(layers, alpha);//更新权重
			break;
		case u_nn_layer_type_output:
			user_nn_grads_output(layers, alpha);//更新权重
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
//获取loss损失值
//layers 获取对象层
//返回 损失值的大小
float user_nn_model_return_loss(user_nn_layers *layers) {
	static float loss_function = 0;//全局变量的loss值
	while (1) {
		if (layers->type == u_nn_layer_type_output) {
			if (loss_function == 0) {
				loss_function = ((user_nn_output_layers *)layers->content)->loss_function;
			}
			else {
				loss_function = (float)0.99f * loss_function + 0.01f * ((user_nn_output_layers *)layers->content)->loss_function;
			}
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return loss_function;
}

//从整个网络中获取一个指定层 按顺序查找
//layers 查找的对象层
//type 目标层类型
//返回 结果对象层
user_nn_layers *user_nn_model_return_layer(user_nn_layers *layers, user_nn_layer_type type) {
	while (1) {
		if (layers->type == type) {
			return layers;//返回最大值
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return NULL;
}
//显示layers所有属性配置
//layers 查找的对象层
//type 直接打印出来
//返回 结果对象层
void user_nn_model_info_layer(user_nn_layers *layers) {
	user_nn_input_layers	*input_infor = NULL;
	user_nn_hidden_layers	*hidden_infor = NULL;
	user_nn_output_layers   *output_infor = NULL;
	printf("\n\n-----NN神经网络层信息-----\n");
	while (1) {
		switch (layers->type) {
			case u_nn_layer_type_null:
				break;
			case u_nn_layer_type_input:
				input_infor = (user_nn_input_layers *)layers->content;
				printf("\n第%d层,输入数据(%d,%d).", layers->index, input_infor->feature_width, input_infor->feature_height);
				break;
			case u_nn_layer_type_hidden:
				hidden_infor = (user_nn_hidden_layers *)layers->content;
				printf("\n第%d层,神经元大小(%d,%d).", layers->index, hidden_infor->feature_width, hidden_infor->feature_height);
				break;
			case u_nn_layer_type_output:
				output_infor = (user_nn_output_layers *)layers->content;
				printf("\n第%d层,输出数据(%d,%d).\n", layers->index, output_infor->feature_width, output_infor->feature_height);
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

//获取输出矩阵
//layers 获取对象层
//返回 矩阵值
user_nn_matrix *user_nn_model_return_result(user_nn_layers *layers) {
	return ((user_nn_output_layers *)user_nn_model_return_layer(layers, u_nn_layer_type_output)->content)->feature_matrix;
}

//显示一连续矩阵
//window_name 窗口名称
//src_matrices 连续矩阵的对象
//gain 放大倍数

//返回 无
void user_nn_model_display_matrix(char *window_name, user_nn_matrix  *src_matrix,int x,int y) {
	int width = (int)sqrt(src_matrix->height*src_matrix->width);
	int height = (int)sqrt(src_matrix->height*src_matrix->width);
	cv::Mat img(width, height, CV_32FC1, src_matrix->data);
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	//cv::resizeWindow(window_name, width, height);
	//cv::updateWindow(win);//opengl
	//cv::startWindowThread();
	cv::moveWindow(window_name,x,y);
	cv::imshow(window_name, img);
	cv::waitKey(1);
}
void user_nn_model_display_feature(user_nn_layers *layers) {
	static int create_flags = 0;
	int window_count = -1;
	char windows_name[128];

	if (create_flags == 0) {
		create_flags = 1;
	}
	while (1) {
		window_count++;
		memset(windows_name, 0, sizeof(windows_name));
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			sprintf(windows_name, "input%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_input_layers  *)layers->content)->feature_matrix, 50 + window_count * 150,20);//显示到指定窗口
			break;
		case u_nn_layer_type_hidden:
			sprintf(windows_name, "hidden%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_hidden_layers  *)layers->content)->kernel_matrix, 50 + window_count * 150, 20);//显示到指定窗口
			break;
		case u_nn_layer_type_output:
			sprintf(windows_name, "output%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_output_layers  *)layers->content)->feature_matrix, 50 + window_count * 150, 20);//显示到指定窗口
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