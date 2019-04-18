
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 784, //输入层 特征（宽度、高度）
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
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//创建模型
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	train_lables = user_nn_matrices_create(20000, 1, 1, 784);
	train_images = user_nn_matrices_create(20000, 1, 1, 784);
	//user_nn_matrices_init_vaule(rand_matrix_list,3,3);
	user_nn_matrix *images_matrix = train_images->matrix;
	user_nn_matrix *lables_matrix = train_lables->matrix;
	user_nn_matrix *kernel_matrix = user_nn_matrix_create(4, 4);//卷积矩阵
	user_nn_matrix *same_matrix = NULL;//卷积矩阵
	user_nn_matrix_memset(kernel_matrix, 0.9f);
	for (int count = 0; count < train_images->height*train_images->width; count++) {
		images_matrix->width = 28;
		images_matrix->height = 28;
		user_nn_matrix_paint_rectangle(images_matrix,
			(int)(user_nn_init_normal() * (images_matrix->width - 2)),
			(int)(user_nn_init_normal() * (images_matrix->height - 2)),
			(int)(user_nn_init_normal() * (images_matrix->width - 2)),
			(int)(user_nn_init_normal() * (images_matrix->height - 2)), 1.0f);//画矩形
		same_matrix = user_nn_matrix_conv2(images_matrix, kernel_matrix, u_nn_conv2_type_same);
		images_matrix->width = 1;
		images_matrix->height = 784;
		same_matrix->width = 1;
		same_matrix->height = 784;
		user_nn_matrix_cpy_matrix(lables_matrix, same_matrix);
		user_nn_matrix_delete(same_matrix);

		images_matrix = images_matrix->next;
		lables_matrix = lables_matrix->next;
	}
	while (1) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_lables, index));//加载输入数据
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载目标数据	
			user_nn_model_ffp(nn_layers);//正向计算一次
			user_nn_model_bp(nn_layers, 0.01f);//反向计算一次
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target) {
				user_nn_model_save_model(nn_layers,0);//保存模型
				break;
			}
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(nn_layers,0);//保存一次模型
				start_time = clock();
			}
			
		}
		if (loss_function < loss_target) {
			break;//跳出训练
		}
	}
	system("pause");
}
void user_nn_app_ident(int argc, const char** argv) {
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		system("pause");
		return ;
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