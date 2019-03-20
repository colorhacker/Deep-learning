
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
	user_nn_matrix_cpy_matrix(((user_nn_input_layers *)nn_input_layer->content)->feature_matrix, src_matrix);
}
//加载特征数据到指定到期望特征数据中
//layers 加载对象层
//src_matrix 目标数据
//返回 无
void user_nn_model_load_target_feature(user_nn_layers *layers, user_nn_matrix *src_matrix) {
	user_nn_layers *nn_output_layer = user_nn_model_return_layer(layers, u_nn_layer_type_output);//获取输入层
	user_nn_matrix_cpy_matrix(((user_nn_output_layers *)nn_output_layer->content)->target_matrix, src_matrix);
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
void user_nn_model_display_matrix(char *window_name, user_nn_matrix  *src_matrix, int gain) {
	user_nn_matrix *dest_matrix = user_nn_matrix_expand_mult_constant(src_matrix, gain, gain, (float)255);//进行放大处理
	CvSize cvsize = { dest_matrix->width, dest_matrix->height };
	IplImage *dest_image = cvCreateImage(cvsize, IPL_DEPTH_8U, 1);
	user_nn_matrix_uchar_memcpy((unsigned char *)dest_image->imageData, dest_matrix);//更新图像数据
	cvShowImage(window_name, dest_image);//显示图像
	cvWaitKey(1);
	cvReleaseImage(&dest_image);//释放内存
	user_nn_matrix_delete(dest_matrix);//删除矩阵
}
void user_nn_model_display_feature(user_nn_layers *layers) {
	static int create_flags = 0;
	char windows_name[128];

	if (create_flags == 0) {
		create_flags = 1;
	}
	while (1) {
		memset(windows_name, 0, sizeof(windows_name));
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			sprintf(windows_name, "input%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_input_layers  *)layers->content)->feature_matrix, 2);//显示到指定窗口
			break;
		case u_nn_layer_type_hidden:
			sprintf(windows_name, "hidden%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_hidden_layers  *)layers->content)->feature_matrix, 2);//显示到指定窗口
			break;
		case u_nn_layer_type_output:
			sprintf(windows_name, "output%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_output_layers  *)layers->content)->feature_matrix, 2);//显示到指定窗口
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