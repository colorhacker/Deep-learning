
#include "user_rnn_app.h"


void user_rnn_app_test(int argc, const char** argv) {
	int user_layers[] = {
		'i', 1, 2, 2,//输入层 特征（宽度、高度、时间长度）
		'h', 2, 2,//隐含层 特征 （高度、时间长度）
		'o', 2, 2//输出层 特征 （高度、时间长度）
	};

	user_rnn_input_layers	*rnn_input_layers = NULL;
	user_rnn_hidden_layers	*rnn_hidden_layers = NULL;
	user_rnn_output_layers	*rnn_output_layers = NULL;

	float loss_function = 0.0f;
	bool model_is_exist = false;
	user_nn_list_matrix *input_data = user_nn_matrices_create(1, 2, 1, 2);//创建输入数据
	user_nn_list_matrix *target_data = user_nn_matrices_create(1, 2, 1, 2);//创建输入数据

	user_rnn_layers *rnn_layers = user_rnn_model_load_model(user_nn_model_rnn_file_name);//载入模型
	if (rnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		rnn_layers = user_rnn_model_create(user_layers);//创建模型
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	//加载数据
	user_nn_list_matrix *input_feature_matrices = ((user_rnn_input_layers *)user_rnn_layers_get(rnn_layers, 1)->content)->feature_matrices;
	int index = 0;
	for (index = 0; index < (input_data->height * input_data->width); index++) {
		*user_nn_matrix_ext_value_index(user_nn_matrices_ext_matrix_index(input_data, index), 0) = (float)index + 1;
		*user_nn_matrix_ext_value_index(user_nn_matrices_ext_matrix_index(input_data, index), 1) = (float)index + 1;
		*user_nn_matrix_ext_value_index(user_nn_matrices_ext_matrix_index(target_data, index), 0) = (index + 1) * 0.01f;
		*user_nn_matrix_ext_value_index(user_nn_matrices_ext_matrix_index(target_data, index), 1) = 1 - (index + 1) * 0.01f;
	}

	//user_nn_matrices_printf(NULL, "input", input_data);
	//user_nn_matrices_printf(NULL, "target", target_data);
	while (!model_is_exist) {
		user_rnn_model_load_input_feature(rnn_layers, input_data);//加载输入数据
		user_rnn_model_load_target_feature(rnn_layers, target_data);//加载目标数据
																	//正向计算一次 按时间片迭代N此
		user_rnn_model_ffp(rnn_layers);
		//反向计算一次 按时间片迭代N此
		user_rnn_model_bp(rnn_layers, 0.01f);
		loss_function = user_rnn_model_return_loss(rnn_layers);
		if (loss_function <= 0.001f) {
			user_rnn_model_save_model(user_nn_model_rnn_file_name, rnn_layers);//保存模型
			break;
		}
		printf("\nloss:%f", loss_function);
	}
	user_rnn_model_ffp(rnn_layers);
	user_nn_matrices_printf(NULL, "result", user_rnn_model_return_result(rnn_layers));

	getchar();
}