
#include "user_snn_app.h"

void user_snn_app_train(int argc, const char** argv) {
	//float data[] = { 1,0,0,0,0,0,0,0,0,0};
	//user_nn_matrix *matrix = user_nn_matrix_create_memset(1, sizeof(data)/ sizeof(float), data);
	//user_snn_data_softmax(matrix);
	//user_nn_matrix_printf(NULL, matrix);
	//system("pause");
	//return;
	srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int layers[] = {
		'i', 1, 2, 
		//'f',
		'h', 2,
		//'f',
		'o', 2
	};
	int io = 0;
	user_snn_layers *layer = user_snn_model_create(layers);//����ģ��

	float input[][2] = { { -0.1f,0.1f },{ -0.15f,0.15f } ,{ -0.2f,0.2f },{ -0.3f,0.3f } };
	float output[][2] = { { -1.0f,1.0f },{ -1.5f,1.5f } ,{ -2.0f,2.0f },{ -3.0f,3.0f } };
	user_nn_matrix *input1 = user_nn_matrix_create_memset(1, 2, input[0]);
	user_nn_matrix *input2 = user_nn_matrix_create_memset(1, 2, input[1]);
	user_nn_matrix *input3 = user_nn_matrix_create_memset(1, 2, input[2]);
	user_nn_matrix *input4 = user_nn_matrix_create_memset(1, 2, input[3]);
	user_nn_matrix *output1 = user_nn_matrix_create_memset(1, 2, output[0]);
	user_nn_matrix *output2 = user_nn_matrix_create_memset(1, 2, output[1]);
	user_nn_matrix *output3 = user_nn_matrix_create_memset(1, 2, output[2]);
	user_nn_matrix *output4 = user_nn_matrix_create_memset(1, 2, output[3]);

	//user_snn_data_softmax(output4);//��������
	//user_nn_matrix_printf(NULL, output4);
	for (int count = 0; count < 5000; count++) {
		user_snn_model_load_input_feature(layer, input1);//������������
		user_snn_model_load_target_feature(layer, output1);//����Ŀ������
		user_snn_model_ffp(layer);
		user_snn_model_bp(layer);
		user_snn_model_load_input_feature(layer, input2);//������������
		user_snn_model_load_target_feature(layer, output2);//����Ŀ������
		user_snn_model_ffp(layer);
		user_snn_model_bp(layer);
		user_snn_model_load_input_feature(layer, input3);//������������
		user_snn_model_load_target_feature(layer, output3);//����Ŀ������
		user_snn_model_ffp(layer);
		user_snn_model_bp(layer);
		user_snn_model_load_input_feature(layer, input4);//������������
		user_snn_model_load_target_feature(layer, output4);//����Ŀ������
		user_snn_model_ffp(layer);
		user_snn_model_bp(layer);

		if (io++ >= 1000) {
			io = 0;
			printf("\n--->:%f",  user_snn_model_return_loss(layer));
		}
		//user_snn_model_display_feature(snn_layers);
		
	}
	user_snn_model_load_input_feature(layer, input1);//������������
	user_snn_model_load_target_feature(layer, output1);//����Ŀ������
	user_snn_model_ffp(layer);
	user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));
	user_snn_model_load_input_feature(layer, input2);//������������
	user_snn_model_load_target_feature(layer, output2);//����Ŀ������
	user_snn_model_ffp(layer);
	user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));
	user_snn_model_load_input_feature(layer, input3);//������������
	user_snn_model_load_target_feature(layer, output3);//����Ŀ������
	user_snn_model_ffp(layer);
	user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));
	user_snn_model_load_input_feature(layer, input4);//������������
	user_snn_model_load_target_feature(layer, output4);//����Ŀ������
	user_snn_model_ffp(layer);
	user_nn_matrix_printf(NULL, user_snn_model_return_result(layer));

	user_nn_matrix_delete(input1);
	user_nn_matrix_delete(input2);
	user_nn_matrix_delete(input3);
	user_nn_matrix_delete(input4);
	user_nn_matrix_delete(output1);
	user_nn_matrix_delete(output2);
	user_nn_matrix_delete(output3);
	user_nn_matrix_delete(output4);
	system("pause");
	return;
	srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
		//'f',
		'h', 128, //������ ���� ���߶ȣ�
		//'f',
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
	for (int count = 0; count < 5; count++) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_images, train_index));//������������
			user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index));//����Ŀ������	
			user_snn_model_ffp(snn_layers);
			user_snn_model_bp(snn_layers);
			if (info++ >= 1000) {
				info = 0;
				printf("\n--->:%d,%f", train_index, user_snn_model_return_loss(snn_layers));
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

