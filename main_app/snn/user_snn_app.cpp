
#include "user_snn_app.h"


void user_snn_app_train(int argc, const char** argv) {
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
	user_snn_output_layers *snn_output = user_snn_layers_output_create(snn_layers,1);
	for (int count=0;count<1000;count++) {
		for (int i = 0; i < 4;i++) {
			if (i == 0) {
				snn_input->feature_matrix->data[0] = 1.5f;
				snn_input->feature_matrix->data[1] = 0.5f;
				snn_output->target_matrix->data[0] = 1.0f;
			}
			if (i == 2) {
				snn_input->feature_matrix->data[0] = 0.5f;
				snn_input->feature_matrix->data[1] = 1.5f;
				snn_output->target_matrix->data[0] = 0.0f;
			}
			if (i == 3) {
				snn_input->feature_matrix->data[0] = 1.0f;
				snn_input->feature_matrix->data[1] = 1.0f;
				snn_output->target_matrix->data[0] = 0.0f;
			}
			if (i == 4) {
				snn_input->feature_matrix->data[0] = 1.0f;
				snn_input->feature_matrix->data[1] = 1.0f;
				snn_output->target_matrix->data[0] = 0.0f;
			}
			snn_output->min_kernel_matrix->data[0] = 1.0f;
			snn_output->max_kernel_matrix->data[0] = 1.6f;

			snn_output->min_kernel_matrix->data[1] = 0.1f;
			snn_output->max_kernel_matrix->data[1] = 1.5f;

			user_snn_ffp_output(snn_layers->next, snn_layers->next->next);

			//user_nn_matrix_printf(NULL, snn_input->feature_matrix);
			//user_nn_matrix_printf(NULL, snn_output->min_kernel_matrix);
			//user_nn_matrix_printf(NULL, snn_output->max_kernel_matrix);
			user_nn_matrix_printf(NULL, snn_output->feature_matrix);

			user_snn_bp_output_back_prior(snn_layers->next, snn_layers->next->next);

			//user_nn_matrix_printf(NULL, snn_output->min_kernel_matrix);
			//user_nn_matrix_printf(NULL, snn_output->max_kernel_matrix);
			//user_nn_matrix_printf(NULL, snn_output->thred_matrix);
		}
	}
	user_nn_matrix_printf(NULL, snn_output->min_kernel_matrix);
	user_nn_matrix_printf(NULL, snn_output->max_kernel_matrix);
	for (int i = 0; i < 4; i++) {
		if (i == 0) {
			snn_input->feature_matrix->data[0] = 1.5f;
			snn_input->feature_matrix->data[1] = 0.5f;
			snn_output->target_matrix->data[0] = 1.0f;
		}
		if (i == 2) {
			snn_input->feature_matrix->data[0] = 0.5f;
			snn_input->feature_matrix->data[1] = 1.5f;
			snn_output->target_matrix->data[0] = 0.0f;
		}
		if (i == 3) {
			snn_input->feature_matrix->data[0] = 1.0f;
			snn_input->feature_matrix->data[1] = 1.0f;
			snn_output->target_matrix->data[0] = 0.0f;
		}
		if (i == 4) {
			snn_input->feature_matrix->data[0] = 1.0f;
			snn_input->feature_matrix->data[1] = 1.0f;
			snn_output->target_matrix->data[0] = 0.0f;
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

