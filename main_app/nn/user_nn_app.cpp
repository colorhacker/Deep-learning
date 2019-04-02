
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
#if defined _OPENMP && _USER_API_OPENMP
	int user_layers[] = {
		'i', 1, 784, //输入层 特征（宽度、高度）
		'h', 784, //隐含层 特征 （高度）
		'h', 784, //隐含层 特征 （高度）
		'o', 784 //输出层 特征 （高度）
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----训练可视化-----\n");
	printf("\n1.开启");
	printf("\n2.关闭（或者其他按键）");
	printf("\n请输入数字：");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	const int parallel_count = 10;
	user_nn_layers *nn_layers[parallel_count];

	for (int index = 0; index < parallel_count; index++) {
		nn_layers[index] = user_nn_model_load_model(user_nn_model_nn_file_name);
		if (nn_layers[index] == NULL) {
			nn_layers[index] = user_nn_model_create(user_layers);//创建模型
		}
	}
	user_nn_model_layer_average(nn_layers, parallel_count);//求取一次平均值
	user_nn_model_info_layer(nn_layers[0]);
	start_time = clock();
	while (1) {
		int index = 0;
		#pragma omp parallel for //reduction(+: save_model_count)
		for(int index_p = 0; index_p < parallel_count; index_p++){
			for (int index = 0; index < train_images->height * train_images->width; index++) {
				user_nn_model_load_input_feature(nn_layers[index_p], user_nn_matrices_ext_matrix_index(train_images, index));//加载输入数据
				user_nn_model_load_target_feature(nn_layers[index_p], user_nn_matrices_ext_matrix_index(train_images, index));//加载目标数据	
				user_nn_model_ffp(nn_layers[index_p]);//正向计算一次
				user_nn_model_bp(nn_layers[index_p], 0.01f);//反向计算一次

				if (save_model_count++ > 1000) {
					break;
				}
			}
			if (save_model_count > 1000) {
				break;
			}
		}
		user_nn_model_layer_average(nn_layers, parallel_count);//统一所有神经网络值
		loss_function = user_nn_model_return_loss(nn_layers[0]);//
		save_model_count = 0;
		end_time = clock();
		printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
		user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers[0]);//保存一次模型
		start_time = clock();
		if (loss_function < loss_target) {
			break;//跳出训练
		}
		if (sw_display) {
			user_nn_model_display_feature(nn_layers[0]);
		}
	}
#else
	int user_layers[] = {
		'i', 1, 784, //输入层 特征（宽度、高度）
		'h', 784, //隐含层 特征 （高度）
		'h', 784, //隐含层 特征 （高度）
		'o', 784 //输出层 特征 （高度）
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----训练可视化-----\n");
	printf("\n1.开启");
	printf("\n2.关闭（或者其他按键）");
	printf("\n请输入数字：");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//创建模型
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	while (1) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载输入数据
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载目标数据	
			user_nn_model_ffp(nn_layers);//正向计算一次
			user_nn_model_bp(nn_layers, 0.01f);//反向计算一次
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target) {
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//保存模型
				break;
			}
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//保存一次模型
				start_time = clock();
			}
		}
		if (loss_function < loss_target) {
			break;//跳出训练
		}
	}
#endif
	system("pause");
}
void user_nn_app_test(int argc, const char** argv) {
	user_nn_app_train(argc,argv);
}