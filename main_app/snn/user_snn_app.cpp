
#include "user_snn_app.h"

user_nn_matrix *user_nn_matrix_create_memset(int width, int height,float *data) {
	user_nn_matrix *dest;

	dest = (user_nn_matrix *)malloc(sizeof(user_nn_matrix));//分配保存矩阵空间的大小
	dest->width = width;
	dest->height = height;
	dest->data = (float *)malloc(dest->width * dest->height * sizeof(float));//分配矩阵数据空间
	dest->next = NULL;
	memcpy(dest->data, data,sizeof(data)*sizeof(float));
	return dest;
}

void user_snn_app_train(int argc, const char** argv) {
	int user_layers[] = {
		'i', 1, 784, //输入层 特征（宽度、高度）
		'h', 392, //隐含层 特征 （高度）
		'o', 10 //输出层 特征 （高度）
	};
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);

	user_snn_layers *snn_layers = user_snn_model_load_model(0);//载入模型
	if (snn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		snn_layers = user_snn_model_create(user_layers);//创建模型
	}
	//user_nn_matrix *src_matrix = user_nn_matrix_create(1, 2);//卷积矩阵
	//src_matrix->data[0] = 1.5f;
	//src_matrix->data[1] = 0.5f;
	//for (;;) {
	//	for (int train_index = 0; train_index < 1000; train_index++) {
	//		user_snn_model_load_input_feature(snn_layers, src_matrix);//加载输入数据
	//		user_snn_model_load_target_feature(snn_layers, src_matrix);//加载目标数据	
	//		user_snn_model_ffp(snn_layers);
	//		user_snn_model_bp(snn_layers);
	//	}
	//	break;
	//}
	//user_snn_model_ffp(snn_layers);
	//user_snn_output_layers *snn_output_layer = (user_snn_output_layers *)user_snn_model_return_layer(snn_layers, u_snn_layer_type_output)->content;
	//user_nn_matrix_printf(NULL, snn_output_layer->feature_matrix);
	int info = 0;
	for (;;) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_images, train_index));//加载输入数据
			user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index));//加载目标数据	
			user_snn_model_ffp(snn_layers);
			user_snn_model_bp(snn_layers);
			if (info++ > 1000) {
				info = 0;
				printf("\n%d", train_index);
			}
		}
		break;
	}
	float success = 0;
	for (;;) {
		for (int test_index = 0; test_index < test_images->height * test_images->width; test_index++) {
			user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(test_images, test_index));//加载输入数据
			user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(test_lables, test_index));//加载目标数据	
			user_snn_model_ffp(snn_layers);

			if (*user_nn_matrix_return_max_addr(user_snn_model_return_result(snn_layers)) == *user_nn_matrix_return_max_addr(user_nn_matrices_ext_matrix_index(test_lables, test_index))) {
				success++;
			}
			if (info++ > 1000) {
				info = 0;
				printf("\n%d", test_index);
			}
		}
		break;
	}
	printf("\n%.4f\n", 100 * success /(float)(test_images->height * test_images->width));
	system("pause");
}
void user_snn_app_ident(int argc, const char** argv) {

}
void user_snn_app_test(int argc, const char** argv) {
	printf("\n-----功能选择-----\n");
	printf("\n1.训练数据");
	printf("\n2.识别数据");
	printf("\n请输入数字：");
	switch (_getch()) {
	case '1':user_snn_app_train(argc, argv); break;
	case '2':user_snn_app_ident(argc, argv); break;
	default: break;
	}
}

