
#include "user_snn_app.h"

void user_snn_data_softmax(user_nn_matrix *src_matrix) {
	float count = (float)(src_matrix->height * src_matrix->height);
	float *src_data = src_matrix->data;
	float *max_value = user_nn_matrix_return_max_addr(src_matrix);
	if (*max_value > 1.0f) {
		user_nn_matrix_divi_constant(src_matrix, *max_value);//归一
	}
	user_nn_matrix_sum_constant(src_matrix, (count - user_nn_matrix_cum_element(src_matrix)) / count);//平均值设置为1.0f
	user_nn_matrix_divi_constant(src_matrix, 0.0001f);//除法
	user_nn_matrxi_floor(src_matrix);//取整
	user_nn_matrix_mult_constant(src_matrix, 0.0001f);//乘法
	//*max_value += count - user_nn_matrix_cum_element(src_matrix);
	//printf("%-10.6f\n", user_nn_matrix_cum_element(src_matrix));
}

user_nn_matrix *user_snn_ffp(user_nn_matrix *input_matrix, user_nn_matrix *matrix_min, user_nn_matrix *matrix_max) {
	user_nn_matrix *result = NULL;//结果矩阵
	int count = input_matrix->width * input_matrix->height;
	float *input_data = input_matrix->data;
	float *min_data = matrix_min->data;
	float *max_data = matrix_min->data;
	float *result_data = NULL;

	result = user_nn_matrix_create(input_matrix->width, input_matrix->height);//创建新的矩阵
	result_data = result->data;//获取数据指针

	for (int height = 0; height < result->height; height++) {
		for (int width = 0; width < result->width; width++) {
			input_data = input_matrix->data + height * input_matrix->width;//指向行开头
			min_data = matrix_min->data + width;//指向列开头
			max_data = matrix_max->data + width;//指向列开头
			for (int point = 0; point < matrix_min->height; point++) {
				printf("\n%f", *input_data);
				if ((*min_data <= *input_data) && (*input_data <= *max_data)) {
					*result_data += 1.0f;
				}
				min_data += matrix_min->width;
				max_data += matrix_max->width;
				input_data++;
			}
			result_data++;
		}
	}
	return result;
}
user_nn_matrix *user_snn_dist(user_nn_matrix *output_matrix, user_nn_matrix *target_matrix) {
	user_nn_matrix *result = NULL;//结果矩阵
	int count = output_matrix->width * output_matrix->height;
	float *output_data = output_matrix->data;
	float *target_data = target_matrix->data;
	float *result_data = NULL;

	result = user_nn_matrix_create(output_matrix->width, output_matrix->height);//创建新的矩阵
	result_data = result->data;//获取数据指针
	
	while (count--) {
		if (*output_data > *target_data) {
			*result_data = 1.0f;
		}
		else if (*output_data < *target_data) {
			*result_data = 0.5f;
		}
		else {

		}
		result_data++;
		output_data++;
		target_data++;
	}
	return result;
}
void user_snn_bp(user_nn_matrix *matrix_input, user_nn_matrix *matrix_dist,user_nn_matrix *matrix_min, user_nn_matrix *matrix_max) {
	int count = matrix_dist->width * matrix_dist->height;
	float *input_data = matrix_input->data;
	float *dist_data = matrix_dist->data;
	float *min_data = matrix_min->data;
	float *max_data = matrix_min->data;
	float *result_data = NULL;
	float avg_value = 1.0f;
	float step_value = 0.000001f;

	for (int height = 0; height < matrix_dist->height; height++) {
		for (int width = 0; width < matrix_dist->width; width++) {
			input_data = matrix_input->data + height * matrix_input->width;//指向行开头
			dist_data = matrix_dist->data + height * matrix_dist->width;//指向行开头
			min_data = matrix_min->data + width;//指向列开头
			max_data = matrix_max->data + width;//指向列开头
			for (int point = 0; point < matrix_min->height; point++) {
				if (*dist_data == 1.0f) {
					if (*input_data >= avg_value) {
						*min_data = *min_data > avg_value ? (*min_data - step_value) : (*min_data + step_value);
						*max_data = *max_data > *input_data ? (*max_data - step_value) : *max_data;
					}else {
						*max_data = *max_data > avg_value ? (*max_data - step_value) : (*max_data + step_value);
						*min_data = *min_data > *input_data ? *min_data: (*max_data + step_value);
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}else if (*dist_data == 0.5f) {
					if (*input_data >= avg_value) {
						*min_data = *min_data > avg_value ? (*min_data - step_value) : (*min_data + step_value);
						*max_data = *max_data > *input_data ? *max_data : (*max_data + step_value);
					}
					else {
						*max_data = *max_data > avg_value ? (*max_data - step_value) : (*max_data + step_value);
						*min_data = *min_data > *input_data ? (*max_data - step_value) : *min_data;
					}
					*max_data = *min_data > *max_data ? *min_data : *max_data;
				}
				else {

				}
				min_data += matrix_min->width;
				max_data += matrix_max->width;
				dist_data++;
				input_data++;
			}
			result_data++;
		}
	}
}
void user_snn_app_train(int argc, const char** argv) {
	/*user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	for (;;) {
	for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
	user_snn_data_softmax(user_nn_matrices_ext_matrix_index(train_images, train_index));
	user_opencv_show_matrix("a", user_nn_matrices_ext_matrix_index(train_images, train_index), 100, 100, 1);
	_getch();
	}
	}*/
	user_nn_matrix *input_matrix = NULL;
	user_nn_matrix *threshold_min = NULL;
	user_nn_matrix *threshold_max = NULL;
	user_nn_matrix *output_matrix = NULL;
	user_nn_matrix *target_matrix = NULL;
	user_nn_matrix *discriminate_matrix = NULL;

	user_nn_matrix *max = user_nn_matrix_create(1, 2);
	user_nn_matrix *min = user_nn_matrix_create(1, 2);
	user_nn_matrix_memset(min, 0.2f);
	user_nn_matrix_memset(max, 0.3f);

	user_nn_list_matrix *train_images = user_nn_matrices_create(1, 4, 1, 2);
	user_nn_list_matrix *train_lables = user_nn_matrices_create(1, 4, 1, 1);
	train_images->matrix->data[0] = 1.5f;
	train_images->matrix->data[1] = 0.5f;
	train_images->matrix->next->data[0] = 0.5f;
	train_images->matrix->next->data[1] = 1.5f;
	train_images->matrix->next->next->data[0] = 1.0f;
	train_images->matrix->next->next->data[1] = 1.0f;
	train_images->matrix->next->next->next->data[0] = 1.0f;
	train_images->matrix->next->next->next->data[1] = 1.0f;

	train_lables->matrix->data[0] = 1.0f;
	train_lables->matrix->next->data[0] = 0.0f;
	train_lables->matrix->next->next->data[0] = 0.0f;
	train_lables->matrix->next->next->next->data[0] = 0.0f;
	for (int count=0;count < 1000;count++) {
		for (int index = 0; index < 4;index++) {
			input_matrix = user_nn_matrices_ext_matrix_index(train_images, index);
			target_matrix = user_nn_matrices_ext_matrix_index(train_lables, index);

			output_matrix = user_snn_ffp(input_matrix,min,max);
			discriminate_matrix = user_snn_dist(output_matrix, target_matrix);
			user_snn_bp(input_matrix,discriminate_matrix,min,max);
			user_nn_matrix_delete(output_matrix);
			user_nn_matrix_delete(discriminate_matrix);
		}
	}

	user_nn_matrix_printf(NULL, min);
	user_nn_matrix_printf(NULL, max);

	output_matrix = user_snn_ffp(user_nn_matrices_ext_matrix_index(train_images, 0), min, max);
	user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
	output_matrix = user_snn_ffp(user_nn_matrices_ext_matrix_index(train_images, 1), min, max);
	user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
	output_matrix = user_snn_ffp(user_nn_matrices_ext_matrix_index(train_images, 2), min, max);
	user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);
	output_matrix = user_snn_ffp(user_nn_matrices_ext_matrix_index(train_images, 3), min, max);
	user_nn_matrix_printf(NULL, output_matrix); user_nn_matrix_delete(output_matrix);

	user_nn_matrices_printf(NULL, "i", train_images);
	user_nn_matrices_printf(NULL, "o", train_lables);

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