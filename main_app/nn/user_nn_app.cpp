
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 28, 28, //输入层 特征（宽度、高度）
		'h', 28, //隐含层 特征 （高度）
		'o', 28 //输出层 特征 （高度）
	};

	float loss_function = 1.0f;
	bool model_is_exist = false;
	//加载mnist数据
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//载入模型
	user_nn_matrix *input_mnist_data = user_nn_matrix_create(28,28);
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		nn_layers = user_nn_model_create(user_layers);//创建模型
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	if (model_is_exist == false) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_matrix_cpy_matrix(input_mnist_data, user_nn_matrices_ext_matrix_index(train_images, index));
			//user_nn_matrix_divi_constant(input_mnist_data, 255.0);
			user_nn_model_load_input_feature(nn_layers, input_mnist_data);//加载输入数据
			user_nn_model_load_target_feature(nn_layers, input_mnist_data);//加载目标数据									   
			user_nn_model_ffp(nn_layers);//正向计算一次
			user_nn_model_bp(nn_layers, 0.5f);//反向计算一次
			loss_function = user_nn_model_return_loss(nn_layers);
			user_nn_model_display_feature(nn_layers);
			printf("\n%d loss:%f",  index, loss_function);
			if (loss_function <= 0.000001f) {
				//user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//保存模型
				break;
			}
		}
	}
	system("pause");
}
