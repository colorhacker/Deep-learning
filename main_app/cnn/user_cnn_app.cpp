
#include "user_cnn_app.h"

const char *files_list[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
//const char *files_list[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
//"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
//const char *files_list[] = { "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };

void cnn_detect(int argc, const char** argv) {
	bool sw_display = false;
	//char scanf_char;
	printf("\n是否开启训练特征显示（开始特征显示训练时间较长）：\n 键入 y 回车进行开启，直接回车默认关闭\n\n");
	sw_display = (getchar() == 'y') ? true : false;

	int user_layers[] = {
		'i', 28, 28, 1,//输入层 特征（宽度、高度、个数）
		'c', 5, 5, 6,//卷积层 特征（宽度、高度、个数）
		's', 2, 2,//子采样层 特征（宽度、高度、个数）
		'c', 5, 5, 12,
		's', 2, 2,
		'f',//全连接层---增减全链接层之后 训练速度明显降低许多
		'o', (sizeof(files_list) / sizeof(files_list[0]))//输出层 特征（分类个数）
	};


	char *full_path = NULL;
	bool model_is_exist = false;
	int null_file = 0, train_count = 50, train_index = 0;
	float loss_function = 0;
	user_nn_list_matrix *target_matrices = NULL;

	srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(user_nn_model_cnn_file_name);//载入模型
	if (cnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		cnn_layers = user_cnn_model_create(user_layers);//创建模型
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	//user_nn_matrix_printf(NULL, ((user_cnn_conv_layers *)user_cnn_layers_get(cnn_layers, 2)->content)->kernel_matrices->matrix);
	//user_nn_matrix_printf(NULL, ((user_cnn_conv_layers *)user_cnn_layers_get(load_layers, 2)->content)->kernel_matrices->matrix);
	target_matrices = user_nn_matrices_create(1, (sizeof(files_list) / sizeof(files_list[0])), 1, (sizeof(files_list) / sizeof(files_list[0])));
	for (int index = 0; index < (target_matrices->width *target_matrices->height); index++) {
		*user_nn_matrix_ext_value_index(user_nn_matrices_ext_matrix_index(target_matrices, index), index) = 1;
	}

	clock_t start_time = clock();
	//进行训练
	while (!model_is_exist) {
		train_count++;
		null_file = 0;
		for (train_index = 0; train_index < (sizeof(files_list) / sizeof(files_list[0])); train_index++) {
			full_path = user_cnn_model_full_path((char *)files_list[train_index], train_count);//获取训练的数据的完整路径
			if (full_path != NULL) {
				user_cnn_model_load_input_image(cnn_layers, full_path, 1);//加载图像至输入层的第一个特征中
				user_cnn_model_ffp(cnn_layers);//正向训练一次
				user_cnn_model_bp(cnn_layers, user_nn_matrices_ext_matrix_index(target_matrices, train_index), 0.5f);//反向训练一次
				loss_function = user_cnn_model_return_loss(cnn_layers);//获取损失函数
			}
			else {
				null_file++;//查找文件失败进行空文件计数
			}
			if (sw_display) {
				user_cnn_model_display_feature(cnn_layers);//显示所有特征数据
			}
			//如果损失函数小于期望值直接退出
			if (loss_function < 0.001f) {
				break;//跳出迭代
			}
		}
		printf("target loss:0.001,loss:%f,null:%d,count:%d\n", loss_function, null_file, train_count);
		//如果损失函数小于期望值直接退出
		if (loss_function < 0.001f) {
			break;//跳出训练
		}
		//如果空文件计数超过当前训练的个数，可以进行选择继续迭代或者退出迭代
		if (null_file >= 4) {
			train_count = 50;//继续迭代
							 //break;//退出迭代
		}
	}
	clock_t end_time = (clock() - start_time) / 1000 / 60;//获取结束时间
	user_model_save_string("\n模型训练结束，损失值:");
	user_model_save_float(loss_function);
	user_model_save_string("\n总时间:");
	user_model_save_int(end_time);
	user_model_save_string("分钟");
	//进行测试
	int test_count = 0, error_count = 0;
	while (1) {
		test_count++;
		null_file = 0;
		for (train_index = 0; train_index < (sizeof(files_list) / sizeof(files_list[0])); train_index++) {
			full_path = user_cnn_model_full_path((char *)files_list[train_index], test_count);//获取训练的数据的完整路径
			if (full_path != NULL) {
				user_cnn_model_load_input_image(cnn_layers, full_path, 1);//加载图像至输入层的第一个特征中
				user_cnn_model_ffp(cnn_layers);//正向训练一次
				if (train_index != user_cnn_model_return_class(cnn_layers)) {
					error_count++;
					user_model_save_string("\n\n识别错误\n 文件路径:");
					user_model_save_string(full_path);
					user_model_save_string("\n测试数值为:");
					user_model_save_int(train_index);
					user_model_save_string("\n识别为:");
					user_model_save_int(user_cnn_model_return_class(cnn_layers));
					//((user_cnn_output_layers *)cnn_layers->content)->feature_matrix;
				}
				else {
					//user_model_save_string("\n测试成功 文件路径:");
					//user_model_save_string(full_path);
				}
			}
			else {
				null_file++;//查找文件失败进行空文件计数
			}
		}
		//如果空文件计数超过当前训练的个数，可以进行选择继续迭代或者退出
		if (null_file >= train_index) {
			break;//退出迭代
		}
		if (test_count >= 50) {
			break;
		}
	}
	user_model_save_string("\n\n识别成功率:");
	user_model_save_float(((float)1 - (float)error_count / 500) * 100);
	user_model_save_string("%");

	user_cnn_model_save_model(user_nn_model_cnn_file_name, cnn_layers);//保存模型
																	   //system("pause");

}

bool cnn_test(int argc, const char** argv) {
	/*
	Mat a = Mat::ones(2048, 2048, CV_32FC1);
	Mat b = Mat::ones(2048, 2048, CV_32FC1);
	Mat c;
	printf("start:\n");
	InitMat(a, 2.5f);
	InitMat(b, 1.0f);
	clock_t test_start_time = clock();
	c = a*b;
	clock_t test_end_time = (clock() - test_start_time) / 1000;//获取结束时间
	printf("total time : %d\n", test_end_time);
	*/
	char model_path[256] = "";
	sprintf_s(model_path, "%s\\%s", user_cnn_model_get_exe_path(), user_nn_model_cnn_file_name);
	printf("%s\n", model_path);
	//printf("%s\n", argv[0]);//这个是exe文件路径
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(model_path);//载入模型
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
			printf("识别结果为：%d,解析为：%s\n", user_cnn_model_return_class(cnn_layers), files_list[user_cnn_model_return_class(cnn_layers)]);
			//
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
	if (cnn_test(argc, argv) == false) {
		cnn_detect(argc, argv);
	}
}