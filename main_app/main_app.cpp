#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"
#include "matrix/user_nn_matrix_cuda.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"

int main(int argc, const char** argv){
	user_cnn_app_test(NULL,NULL);
	/*
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 2, //输入层 特征（宽度、高度、时间长度）
		'h', 2, //隐含层 特征 （高度、时间长度）
		'o', 2 //输出层 特征 （高度、时间长度）
	};
	user_nn_input_layers	*nn_input_layers = NULL;
	user_nn_hidden_layers	*nn_hidden_layers = NULL;
	user_nn_output_layers	*nn_output_layers = NULL;

	float loss_function = 0.0f;
	bool model_is_exist = false;
	user_nn_matrix *input_data = user_nn_matrix_create(1, 2);//创建输入数据
	user_nn_matrix *target_data = user_nn_matrix_create(1, 2);//创建输入数据

	user_nn_layers *rnn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//载入模型
	if (rnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		rnn_layers = user_nn_model_create(user_layers);//创建模型
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	input_data->data[0] = 1;
	input_data->data[1] = 1;
	target_data->data[0] = 0.01f;
	target_data->data[1] = 0.01f;
	while (!model_is_exist) {
		user_nn_model_load_input_feature(rnn_layers, input_data);//加载输入数据
		user_nn_model_load_target_feature(rnn_layers, target_data);//记载目标数据
																   //正向计算一次 按时间片迭代N此
		user_nn_model_ffp(rnn_layers);
		//反向计算一次 按时间片迭代N此
		user_nn_model_bp(rnn_layers, 0.01f);
		loss_function = user_nn_model_return_loss(rnn_layers);
		if (loss_function <= 0.001f) {
			user_nn_model_save_model(user_nn_model_nn_file_name, rnn_layers);//保存模型
			break;
		}
		printf("\nloss:%f", loss_function);
	}
	user_nn_model_load_input_feature(rnn_layers, input_data);//加载输入数据
	user_nn_model_ffp(rnn_layers);//进行计算
	user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));
	*/
	getchar();
	return 0;
}

/*
	float test_content[] = {
		1,1,0,0,
		0,1,-1,1,
		0,0,1,1,
		1,0,1,0
	};//矩阵数据
	user_nn_matrix *test_matrix = user_nn_matrix_create(4,4);
	user_nn_matrix_memcpy(test_matrix, test_content);

	user_nn_matrix *path_matrix = get_loop_path_list(test_matrix, user_nn_matrix_return_min_index(test_matrix));
	if (path_matrix != NULL)
		user_nn_matrix_printf(NULL, path_matrix);//
	else
		printf("\nmatrix error!\n");
*/