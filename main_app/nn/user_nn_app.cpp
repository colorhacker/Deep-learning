
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 784, //输入层 特征（宽度、高度）
		'h', 784, //隐含层 特征 （高度）
		'h', 784, //隐含层 特征 （高度）
		'o', 784 //输出层 特征 （高度）
	};

	float loss_function = 1.0f,loss_target= 0.0003f;
	int save_model_count = 0;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//创建模型
	}
	user_nn_model_info_layer(nn_layers);
	while (1) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载输入数据
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载目标数据	
			user_nn_model_ffp(nn_layers);//正向计算一次
			user_nn_model_bp(nn_layers, 0.01f);//反向计算一次
			loss_function = user_nn_model_return_loss(nn_layers);
			//user_nn_model_display_feature(nn_layers);
			if (loss_function <= loss_target) {
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//保存模型
				break;
			}
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				printf("\ntarget:%f loss:%f", loss_target, loss_function);
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//保存一次模型
			}
		}
		if (loss_function < loss_target) {
			break;//跳出训练
		}
	}
	system("pause");
}
void user_nn_app_test(int argc, const char** argv) {
	user_nn_app_train(argc,argv);
}