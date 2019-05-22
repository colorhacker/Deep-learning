
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
/*
	float src[] = { 1.0f };
	float min[] = { 0.5f,0.1f };
	user_nn_matrix *src_matrix = user_nn_matrix_create_memset(1, 1,src);
	user_nn_matrix *min_matrix = user_nn_matrix_create_memset(2, 1,min);
	user_nn_matrix *res_matrix = user_nn_matrix_mult_matrix(src_matrix, min_matrix);//

	if (res_matrix != NULL) {
		user_nn_matrix_printf(NULL, res_matrix);//打印矩阵
	}
	else {
		printf("null\n");
	}
	printf("\nend");
	_getch();
	return;*/
	/*
	for (;;) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_snn_data_softmax(user_nn_matrices_ext_matrix_index(train_images, train_index));
			user_opencv_show_matrix("a", user_nn_matrices_ext_matrix_index(train_images, train_index), 100, 100, 1);
			_getch();
		}
	}
	*/
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	user_snn_layers	*snn_layers = user_snn_layers_create(u_snn_layer_type_null, 0);//创建一个空层

	user_snn_input_layers *snn_input = user_snn_layers_input_create(snn_layers, 1, 2);
	user_snn_hidden_layers *snn_hidden = user_snn_layers_hidden_create(snn_layers, 2);
	user_snn_output_layers *snn_output = user_snn_layers_output_create(snn_layers,2);
	for (int count=0;count<1000;count++) {
		for (int i = 0; i < 2;i++) {
			if (i == 0) {
				snn_input->feature_matrix->data[0] = 1.5f;
				snn_input->feature_matrix->data[1] = 0.5f;
				snn_output->target_matrix->data[0] = 2.0f;
				snn_output->target_matrix->data[1] = 0.0f;
			}
			if (i == 1) {
				snn_input->feature_matrix->data[0] = 0.5f;
				snn_input->feature_matrix->data[1] = 1.5f;
				snn_output->target_matrix->data[0] = 0.0f;
				snn_output->target_matrix->data[1] = 2.0f;
			}
			//user_nn_matrix_printf(NULL, snn_hidden->min_kernel_matrix);
			//user_nn_matrix_printf(NULL, snn_hidden->max_kernel_matrix);
			user_snn_ffp_hidden(snn_layers->next, snn_layers->next->next);//输入层到中间层
			user_snn_ffp_output(snn_layers->next->next, snn_layers->next->next->next);//中间层到输入层
			user_snn_bp_output_back_prior(snn_layers->next->next, snn_layers->next->next->next);
			user_snn_bp_hidden_back_prior(snn_layers->next, snn_layers->next->next);

			//user_snn_ffp_output(snn_layers->next, snn_layers->next->next);//中间层到输入层
			//user_snn_bp_output_back_prior(snn_layers->next, snn_layers->next->next);
		}
	}
	//user_nn_matrix_printf(NULL, snn_hidden->min_kernel_matrix);
	//user_nn_matrix_printf(NULL, snn_hidden->max_kernel_matrix);

	user_nn_matrix_printf(NULL, snn_output->min_kernel_matrix);
	user_nn_matrix_printf(NULL, snn_output->max_kernel_matrix);
	for (int i = 0; i < 2; i++) {
		if (i == 0) {
			snn_input->feature_matrix->data[0] = 1.5f;
			snn_input->feature_matrix->data[1] = 0.5f;
		}
		if (i == 1) {
			snn_input->feature_matrix->data[0] = 0.5f;
			snn_input->feature_matrix->data[1] = 1.5f;
		}
		user_snn_ffp_output(snn_layers->next, snn_layers->next->next);
		user_nn_matrix_printf(NULL, snn_output->feature_matrix);
	}
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

