
#include "user_nn_app.h"
#include "..\other\user_nn_opencv.h"

user_nn_matrix *user_nn_app_set_input(void) {
	user_nn_matrix *src_matrix = user_nn_matrix_create(1, 1);//卷积矩阵
	src_matrix->data[0] = 1;
	return src_matrix;
}

void train_mnist_gen_network() {
	int gen_layers[] = { 'i',1,1,'o',784 };
	user_nn_matrix *input_matirx = user_nn_matrix_create(1, 1);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_layers *nn_gen_layers = user_nn_model_load_model(1);
	if (nn_gen_layers == NULL) {
		nn_gen_layers = user_nn_model_create(gen_layers);
		for (int count = 1; count < 10000; count++) {
			user_nn_model_load_input_feature(nn_gen_layers, input_matirx);
			user_nn_model_load_target_feature(nn_gen_layers, user_nn_matrices_ext_matrix_index(train_images, count));
			user_nn_model_ffp(nn_gen_layers);
			user_nn_model_bp(nn_gen_layers, 0.01f);
			if (count % 1000 == 0) {
				printf("\ncount:%d,loss:%f", count, user_nn_model_return_loss(nn_gen_layers));
			}
			user_nn_model_display_feature(nn_gen_layers);//显示图像
		}
		printf("\nloss:%f", user_nn_model_return_loss(nn_gen_layers));
		user_nn_model_save_model(nn_gen_layers, 1);
	}
	float gen_loss = user_nn_model_return_loss(nn_gen_layers);

	user_nn_matrix *start_kernel = user_nn_matrix_create(1,784);
	user_nn_list_matrix *train_kernel_matrces = user_nn_model_file_read_matrices("./model/kernel_data.bin", 0);
	if(train_kernel_matrces == NULL){ 
		train_kernel_matrces = user_nn_matrices_create(1, 1000, 1, 784);
		for (int index = 0; index < 1000; index++) {
			user_nn_layers_all_delete(nn_gen_layers);//删除层
			nn_gen_layers = user_nn_model_load_model(1);//加载模型
			for (int count = 0; count < 40; count++) {
				user_nn_model_load_target_feature(nn_gen_layers, user_nn_matrices_ext_matrix_index(train_images, index));
				user_nn_model_ffp(nn_gen_layers);
				user_nn_model_bp(nn_gen_layers, 0.01f);
			}
			user_nn_matrix_cum_matrix(user_nn_matrices_ext_matrix_index(train_kernel_matrces, index), ((user_nn_output_layers *)user_nn_layers_get(nn_gen_layers, 2)->content)->kernel_matrix, ((user_nn_output_layers *)user_nn_layers_get(nn_gen_layers, 2)->content)->biases_matrix);
			//user_nn_matrix_cpy_matrix(user_nn_matrices_ext_matrix_index(train_kernel_matrces, index), ((user_nn_hidden_layers *)user_nn_layers_get(nn_gen_layers, 2)->content)->kernel_matrix);
			printf("\ncount:%d,%d", index, user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(train_lables, index)));
			user_nn_model_display_feature(nn_gen_layers);//显示图像
		}
		user_nn_model_file_save_matrices("./model/kernel_data.bin", 0, train_kernel_matrces);
	}
	int dist_layers[] = { 'i',1,784,'o',10 };
	user_nn_layers *nn_dist_layers = user_nn_model_load_model(2);
	if (nn_dist_layers == NULL) {
		nn_dist_layers = user_nn_model_create(dist_layers);
		for(;;){
			for (int index = 0; index < 1000; index++) {
				user_nn_model_load_input_feature(nn_dist_layers, user_nn_matrices_ext_matrix_index(train_kernel_matrces, index));
				user_nn_model_load_target_feature(nn_dist_layers, user_nn_matrices_ext_matrix_index(train_lables, index));
				user_nn_model_ffp(nn_dist_layers);
				user_nn_model_bp(nn_dist_layers, 0.01f);
				printf("\nloss:%f", user_nn_model_return_loss(nn_dist_layers));
				if (user_nn_model_return_loss(nn_dist_layers) < 0.01f) {
					break;
				}
				if (user_nn_matrix_return_max_index(user_nn_model_return_result(nn_dist_layers)) != user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(train_lables, index))) {
					printf("--error %d,%d", user_nn_matrix_return_max_index(user_nn_model_return_result(nn_dist_layers)), user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(train_lables, index)));
				}
			}
			if (user_nn_model_return_loss(nn_dist_layers) < 0.01f) {
				user_nn_model_save_model(nn_gen_layers, 2);
				break;
			}
		}
	}

	_getch();
}
void train_mnist_distinguish_network(int id) {

}

void user_nn_app_train(int argc, const char** argv) {
	train_mnist_gen_network();
	return;
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 1, //输入层 特征（宽度、高度）
		'h', 392, //隐含层 特征 （高度）
		'o', 784 //输出层 特征 （高度）
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0, exit_train_count = 1000;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----训练可视化-----\n");
	printf("\n1.开启");
	printf("\n2.关闭（或者其他按键）");
	printf("\n请输入数字：");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//创建模型
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	user_nn_matrix *input_matirx = user_nn_app_set_input();
	while (1)
	{
		for (int index = 0; index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, input_matirx);//加载输入数据
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载目标数据
			//user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载输入数据
			//user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_lables, index));//加载目标数据	
			user_nn_model_ffp(nn_layers);//正向计算一次
			user_nn_model_bp(nn_layers, 0.01f);//反向计算一次
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target) {
				user_nn_model_save_model(nn_layers, 0);//保存模型
				break;
			}
			printf("\ntarget:%f loss:%f", loss_target, loss_function);
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(nn_layers, 0);//保存一次模型
				start_time = clock();
			}
			if (exit_train_count-- <= 0) {
				user_nn_model_save_model(nn_layers, 0);//保存一次模型
				break;
			}
		}
		if (loss_function < loss_target) {
			break;
		}
		if (exit_train_count-- <= 0) {
			break;
		}
	}

	/*FILE *debug_file = NULL;
	debug_file = fopen("kernel.txt", "w+");
	user_nn_layers *nn_output_layer = user_nn_model_return_layer(nn_layers, u_nn_layer_type_output);
	((user_nn_output_layers *)nn_output_layer->content)->target_matrix->height = 28;
	((user_nn_output_layers *)nn_output_layer->content)->target_matrix->width = 28;
	user_nn_matrix_printf(debug_file, ((user_nn_output_layers *)nn_output_layer->content)->kernel_matrix);
	user_nn_matrix_printf(debug_file, user_nn_matrices_ext_matrix_index(train_images, 0));
	*/
	system("pause");
}
void user_nn_app_ident(int argc, const char** argv) {
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		system("pause");
		return;
	}
	user_nn_model_info_layer(nn_layers);
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	for (int index = 0; index < train_images->height * train_images->width; index++) {
		user_nn_matrix *src_matrix = user_nn_matrices_ext_matrix_index(train_images, index);
		src_matrix->width = 1;
		src_matrix->height = 784;
		user_nn_model_load_input_feature(nn_layers, src_matrix);
		user_nn_model_ffp(nn_layers);
		user_nn_model_display_feature(nn_layers);
		_getch();
	}
	system("pause");
}
void user_nn_app_test(int argc, const char** argv) {
	printf("\n-----功能选择-----\n");
	printf("\n1.训练数据");
	printf("\n2.识别数据");
	printf("\n请输入数字：");
	switch (_getch()) {
	case '1':user_nn_app_train(argc, argv); break;
	case '2':user_nn_app_ident(argc, argv); break;
	default: break;
	}
}