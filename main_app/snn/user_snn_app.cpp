
#include "user_snn_app.h"


void user_snn_app_train(int argc, const char** argv) {
	/*
	for (;;) {
	for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
	user_snn_data_softmax(user_nn_matrices_ext_matrix_index(train_images, train_index));
	user_opencv_show_matrix("a", user_nn_matrices_ext_matrix_index(train_images, train_index), 100, 100, 1);
	_getch();
	}
	}*/
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_matrix *input_matrix = NULL;
	user_nn_matrix *threshold_min = NULL;
	user_nn_matrix *threshold_max = NULL;
	user_nn_matrix *output_matrix = NULL;
	user_nn_matrix *target_matrix = NULL;
	user_nn_matrix *thred_matrix = NULL;

	user_nn_matrix *max = user_nn_matrix_create(10, 784);
	user_nn_matrix *min = user_nn_matrix_create(10, 784);
	//user_nn_matrix *max = user_nn_matrix_create(10, 10);
	//user_nn_matrix *min = user_nn_matrix_create(10, 10);
	srand((unsigned)time(NULL));
	user_snn_init_matrix(min,max);
	//user_nn_matrix_printf(NULL, min);
	//user_nn_matrix_printf(NULL, max);
	for (int index = 0; index < train_images->height * train_images->width; index++) {
		user_nn_matrices_ext_matrix_index(train_images, index)->width = 784;
		user_nn_matrices_ext_matrix_index(train_images, index)->height = 1;
		user_snn_data_softmax(user_nn_matrices_ext_matrix_index(train_images, index));//处理矩阵
	}
	float total = 0.0f, success = 0.0f, ctotal = 0.0f, csuccess = 0.0f, loss = 0.0f, old_loss = 0.0f;
	clock_t start_time = clock();
	for (;;) {
		for (int index = 0; index < train_images->height * train_images->width; index++) {
			input_matrix = user_nn_matrices_ext_matrix_index(train_images, index);
			target_matrix = user_nn_matrices_ext_matrix_index(train_lables, index);

			output_matrix = user_nn_matrix_thred_acc(input_matrix,min,max);	
			thred_matrix = user_nn_matrix_thred_process(output_matrix, target_matrix);

			ctotal++;total++;
			if (user_nn_matrix_return_max_index(output_matrix) == user_nn_matrix_return_max_index(target_matrix)) {
				csuccess++; success++;
			}
			user_nn_matrix_update_thred(input_matrix, thred_matrix,min,max,1.0f,0.001f);
			user_nn_matrix_delete(output_matrix);
			user_nn_matrix_delete(thred_matrix);
		}
		loss = (csuccess / ctotal)*100.0f;
		if (loss > old_loss) {
			old_loss = loss;
			printf("\nsingle:%.2f,total:%.4f,time:%ds", loss, (success / total)*100.0f, (clock() - start_time) / 1000);
			start_time = clock();
		}
		if (loss >= 80 || total >= 10* train_images->height * train_images->width) {
			break;
		}
		ctotal = 0;
		csuccess = 0;
	}

	user_nn_matrix_printf(NULL, min);
	user_nn_matrix_printf(NULL, max);

	total = 0; success = 0;
	for (int index = 0; index < 100; index++) {
		input_matrix = user_nn_matrices_ext_matrix_index(train_images, index+500);
		target_matrix = user_nn_matrices_ext_matrix_index(train_lables, index+500);
		output_matrix = user_nn_matrix_thred_acc(input_matrix, min, max);
		thred_matrix = user_nn_matrix_thred_process(output_matrix, target_matrix);

		total++;
		if (user_nn_matrix_return_max_index(output_matrix) == user_nn_matrix_return_max_index(target_matrix)) {
			success++;
		}
	}

	printf("\nsuccess:%.4f\n", (success / total)*100.0f);

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

//user_nn_matrix *input_matrix = NULL;
//user_nn_matrix *threshold_min = NULL;
//user_nn_matrix *threshold_max = NULL;
//user_nn_matrix *output_matrix = NULL;
//user_nn_matrix *target_matrix = NULL;
//user_nn_matrix *thred_matrix = NULL;
//
//user_nn_matrix *max = user_nn_matrix_create(1, 2);
//user_nn_matrix *min = user_nn_matrix_create(1, 2);
//user_nn_matrix_memset(min, 0.2f);
//user_nn_matrix_memset(max, 0.3f);
//
//user_nn_list_matrix *train_images = user_nn_matrices_create(1, 4, 2, 1);
//user_nn_list_matrix *train_lables = user_nn_matrices_create(1, 4, 1, 1);
//train_images->matrix->data[0] = 1.5f;
//train_images->matrix->data[1] = 0.5f;
//train_images->matrix->next->data[0] = 0.8f;
//train_images->matrix->next->data[1] = 0.4f;
//train_images->matrix->next->next->data[0] = 1.0f;
//train_images->matrix->next->next->data[1] = 1.0f;
//train_images->matrix->next->next->next->data[0] = 1.0f;
//train_images->matrix->next->next->next->data[1] = 1.0f;
//
//train_lables->matrix->data[0] = 1.0f;
//train_lables->matrix->next->data[0] = 1.0f;
//train_lables->matrix->next->next->data[0] = 0.0f;
//train_lables->matrix->next->next->next->data[0] = 0.0f;
//
//for (int count = 0; count < 1000; count++) {
//	for (int index = 0; index < 4; index++) {
//		input_matrix = user_nn_matrices_ext_matrix_index(train_images, index);
//		target_matrix = user_nn_matrices_ext_matrix_index(train_lables, index);
//		user_snn_data_softmax(input_matrix);//处理矩阵
//
//		output_matrix = user_nn_matrix_thred_acc(input_matrix, min, max);
//		thred_matrix = user_nn_matrix_thred_process(output_matrix, target_matrix);
//		user_nn_matrix_update_thred(input_matrix, thred_matrix, min, max, 1.0f, 0.001f);
//		user_nn_matrix_delete(output_matrix);
//		user_nn_matrix_delete(thred_matrix);
//	}
//}
//
//user_nn_matrix_printf(NULL, min);
//user_nn_matrix_printf(NULL, max);
//
//output_matrix = user_nn_matrix_thred_acc(user_nn_matrices_ext_matrix_index(train_images, 0), min, max);
//user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
//output_matrix = user_nn_matrix_thred_acc(user_nn_matrices_ext_matrix_index(train_images, 1), min, max);
//user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
//output_matrix = user_nn_matrix_thred_acc(user_nn_matrices_ext_matrix_index(train_images, 2), min, max);
//user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
//output_matrix = user_nn_matrix_thred_acc(user_nn_matrices_ext_matrix_index(train_images, 3), min, max);
//user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
//
//system("pause");