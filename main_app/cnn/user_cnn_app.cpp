
#include "user_cnn_app.h"

void user_cnn_mnist_train() {
	int user_layers[] = {
		'i', 28, 28, 1,//输入层 特征（宽度、高度、个数）
		'c', 5, 5, 6,//卷积层 特征（宽度、高度、个数）
		's', 2, 2,//子采样层 特征（宽度、高度、个数）
		'c', 5, 5, 12,
		's', 2, 2,
		'f',//全连接层---增减全链接层之后 训练速度明显降低许多
		'o', 10//输出层 特征（分类个数）
	};
	bool sw_display = false;
	int save_model_count = 0;
	float loss_function = 0,target_loss= 0.001f;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----训练可视化-----\n");
	printf("\n1.开启");
	printf("\n2.关闭（或者其他按键）");
	printf("\n请输入数字：");
	sw_display = (_getch() == '1') ? true : false;
	//加载mnist数据
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	if (train_images == NULL) {
		printf("not found mnist files!");
		system("pause");
		return ;
	}
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(0);//载入模型
	if (cnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		cnn_layers = user_cnn_model_create(user_layers);//创建模型
	}
	user_cnn_model_info_layer(cnn_layers);
	start_time = clock();
	while (1) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_cnn_model_load_input_feature(cnn_layers, user_nn_matrices_ext_matrix_index(train_images, train_index), 1);
			user_cnn_model_load_target_feature(cnn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index));//加载目标矩阵
			user_cnn_model_ffp(cnn_layers);//正向计算一次
			user_cnn_model_bp(cnn_layers, 0.01f);//反向训练一次
			loss_function = user_cnn_model_return_loss(cnn_layers);//获取损失函数
			//printf("\n%f", loss_function);
			if (sw_display) {
				user_cnn_model_display_feature(cnn_layers);//显示所有特征数据
			}
			//如果损失函数小于期望值直接退出
			if (loss_function < target_loss) {
				break;//跳出迭代
			}
			if (save_model_count++ > 100) {
				save_model_count = 0;
				printf("train count:%d,loss:%f\n", train_index,loss_function);
				//user_cnn_model_save_model(cnn_layers,0);//保存一次模型
			}
		}
		printf("target:%f loss:%f\n", target_loss, loss_function);
		//如果损失函数小于期望值直接退出
		if (loss_function < target_loss) {
			break;//跳出训练
		}
	}
	end_time = (clock() - start_time) / 1000 / 60;//获取结束时间

	user_nn_debug_printf("%s","\n模型训练结束，损失值:");
	user_nn_debug_printf("%f",(void *)&loss_function);
	user_nn_debug_printf("%s","总时间:");
	user_nn_debug_printf("%d", (void *)&end_time);
	user_nn_debug_printf("%s","分钟");
	user_cnn_model_save_model(cnn_layers,0);//保存模型
	printf("\n\n");
	system("pause");
}

void user_cnn_mnist_test() {
	char model_path[256] = "";
	sprintf_s(model_path, "%s\\%s", user_cnn_model_get_exe_path(), user_nn_model_cnn_file_name);
	printf("%s\n", model_path);
	user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(0);//载入模型
	if (cnn_layers == NULL) {
		printf("\n载入模型失败!\n\n");
		system("pause");
		return;
	}
	//进行测试
	float error_count = 0;
	for (int test_index = 0; test_index < test_images->height * test_images->width; test_index++) {
		user_cnn_model_load_input_feature(cnn_layers, user_nn_matrices_ext_matrix_index(test_images, test_index), 1);
		user_cnn_model_ffp(cnn_layers);//正向计算一次
		if (user_cnn_model_return_class(cnn_layers) != user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index))) {
			error_count++;
			user_nn_debug_printf("%s","\n识别错误！图像数字:");
			user_nn_debug_printf("%d", (void *)user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index)));
			user_nn_debug_printf("%s","识别为:");
			user_nn_debug_printf("%s", (void *)user_cnn_model_return_class(cnn_layers));
		}
	}
	user_nn_debug_printf("%s","\n\n识别成功率:");
	error_count = ((float)1 - (float)error_count / (test_images->height * test_images->width)) * 100;
	user_nn_debug_printf("%d", (void *)&error_count);
	user_nn_debug_printf("%s","%");
	system("pause");
}


bool user_cnn_load_ident(int argc, const char** argv) {
	char model_path[256] = "";
	sprintf_s(model_path, "%s\\%s", user_cnn_model_get_exe_path(), user_nn_model_cnn_file_name);
	printf("%s\n", model_path);
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(0);//载入模型
	if (cnn_layers != NULL) {
		printf("loading model success\n");
		if (argv[1] == NULL) {
			printf("path error\n");
		}
		else {
			printf("\n%s\n", argv[1]);
			const char *full_path = argv[1];
			user_cnn_model_load_input_image(cnn_layers, (char *)full_path, 1);//加载图像至输入层的第一个特征中
			user_cnn_model_ffp(cnn_layers);//识别
			printf("识别结果为：%d\n", user_cnn_model_return_class(cnn_layers));
			user_nn_matrix_printf(NULL, ((user_cnn_output_layers *)user_cnn_model_return_layer(cnn_layers, u_cnn_layer_type_output)->content)->feature_matrix);//打印矩阵
		}
		getchar();
		return true;
	}
	else {
		printf("loading model faile\n");
		return false;
	}
}

void user_cnn_app_test(int argc, const char** argv) {
	printf("\n-----功能选择-----\n");
	printf("\n1.训练mnist数据");
	printf("\n2.测试mnist数据");
	printf("\n3.识别28*28图像\n");
	printf("\n请输入数字：");
	switch (_getch()) {
		case '1':user_cnn_mnist_train(); break;
		case '2':user_cnn_mnist_test(); break;
		case '3':user_cnn_load_ident(argc, argv); break;
		default: break;
	}
}