
#include "user_snn_app.h"

user_nn_matrix *user_nn_matrix_create_memset(int width, int height,float *data) {
	user_nn_matrix *dest;

	dest = (user_nn_matrix *)malloc(sizeof(user_nn_matrix));//���䱣�����ռ�Ĵ�С
	dest->width = width;
	dest->height = height;
	dest->data = (float *)malloc(dest->width * dest->height * sizeof(float));//����������ݿռ�
	dest->next = NULL;
	memcpy(dest->data, data,sizeof(data)*sizeof(float));
	return dest;
}

void user_snn_app_train(int argc, const char** argv) {
	srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
		'f',
		'h', 784, //������ ���� ���߶ȣ�
		'f',
		'o', 10 //����� ���� ���߶ȣ�
	};
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);

	user_snn_layers *snn_layers = user_snn_model_load_model(0);//����ģ��
	if (snn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		snn_layers = user_snn_model_create(user_layers);//����ģ��
	}
	int info = 0;
	clock_t start_time = clock();
	for (int count = 0; count < 1; count++) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_images, train_index));//������������
			user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index));//����Ŀ������	
			user_snn_model_ffp(snn_layers);
			user_snn_model_bp(snn_layers);
			if (info++ >= 1000) {
				info = 0;
				printf("\n--->:%d", train_index);
			}
			//user_snn_model_display_feature(snn_layers);
		}
	}
	//user_snn_model_save_model(snn_layers,0);//
	printf("\ntime:%ds", (clock() - start_time) / 1000);
	start_time = clock();
	float success = 0;
	for (;;) {
		for (int test_index = 0; test_index < test_images->height * test_images->width; test_index++) {
			user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(test_images, test_index));//������������
			user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(test_lables, test_index));//����Ŀ������	
			user_snn_model_ffp(snn_layers);
			if (user_nn_matrix_return_max_index(user_snn_model_return_result(snn_layers)) == user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index))) {
				success++;
			}
		}
		break;
	}
	printf("\nsuccess:%.4f%%,time:%ds\n", 100 * success /(float)(test_images->height * test_images->width), (clock() - start_time) / 1000);
	system("pause");
}
void user_snn_app_ident(int argc, const char** argv) {
	float success = 0;
	clock_t start_time = clock();
	user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);

	user_snn_layers *snn_layers = user_snn_model_load_model(0);//����ģ��
	if (snn_layers == NULL) {
		return;
	}
	for (;;) {
		for (int test_index = 0; test_index < test_images->height * test_images->width; test_index++) {
			user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(test_images, test_index));//������������
			user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(test_lables, test_index));//����Ŀ������	
			user_snn_model_ffp(snn_layers);
			if (user_nn_matrix_return_max_index(user_snn_model_return_result(snn_layers)) == user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index))) {
				success++;
			}
		}
		break;
	}
	printf("\nsuccess:%.4f%%,time:%ds\n", 100 * success / (float)(test_images->height * test_images->width), (clock() - start_time) / 1000);
	system("pause");
}
void user_snn_app_test(int argc, const char** argv) {
	printf("\n-----����ѡ��-----\n");
	printf("\n1.ѵ������");
	printf("\n2.ʶ������");
	printf("\n���������֣�");
	switch (_getch()) {
	case '1':user_snn_app_train(argc, argv); break;
	case '2':user_snn_app_ident(argc, argv); break;
	default: break;
	}
}

