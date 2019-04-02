
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
#if defined _OPENMP && _USER_API_OPENMP
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
		'h', 784, //������ ���� ���߶ȣ�
		'h', 784, //������ ���� ���߶ȣ�
		'o', 784 //����� ���� ���߶ȣ�
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----ѵ�����ӻ�-----\n");
	printf("\n1.����");
	printf("\n2.�رգ���������������");
	printf("\n���������֣�");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	const int parallel_count = 10;
	user_nn_layers *nn_layers[parallel_count];

	for (int index = 0; index < parallel_count; index++) {
		nn_layers[index] = user_nn_model_load_model(user_nn_model_nn_file_name);
		if (nn_layers[index] == NULL) {
			nn_layers[index] = user_nn_model_create(user_layers);//����ģ��
		}
	}
	user_nn_model_layer_average(nn_layers, parallel_count);//��ȡһ��ƽ��ֵ
	user_nn_model_info_layer(nn_layers[0]);
	start_time = clock();
	while (1) {
		int index = 0;
		#pragma omp parallel for //reduction(+: save_model_count)
		for(int index_p = 0; index_p < parallel_count; index_p++){
			for (int index = 0; index < train_images->height * train_images->width; index++) {
				user_nn_model_load_input_feature(nn_layers[index_p], user_nn_matrices_ext_matrix_index(train_images, index));//������������
				user_nn_model_load_target_feature(nn_layers[index_p], user_nn_matrices_ext_matrix_index(train_images, index));//����Ŀ������	
				user_nn_model_ffp(nn_layers[index_p]);//�������һ��
				user_nn_model_bp(nn_layers[index_p], 0.01f);//�������һ��

				if (save_model_count++ > 1000) {
					break;
				}
			}
			if (save_model_count > 1000) {
				break;
			}
		}
		user_nn_model_layer_average(nn_layers, parallel_count);//ͳһ����������ֵ
		loss_function = user_nn_model_return_loss(nn_layers[0]);//
		save_model_count = 0;
		end_time = clock();
		printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
		user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers[0]);//����һ��ģ��
		start_time = clock();
		if (loss_function < loss_target) {
			break;//����ѵ��
		}
		if (sw_display) {
			user_nn_model_display_feature(nn_layers[0]);
		}
	}
#else
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
		'h', 784, //������ ���� ���߶ȣ�
		'h', 784, //������ ���� ���߶ȣ�
		'o', 784 //����� ���� ���߶ȣ�
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----ѵ�����ӻ�-----\n");
	printf("\n1.����");
	printf("\n2.�رգ���������������");
	printf("\n���������֣�");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//����ģ��
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//����ģ��
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	while (1) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//������������
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//����Ŀ������	
			user_nn_model_ffp(nn_layers);//�������һ��
			user_nn_model_bp(nn_layers, 0.01f);//�������һ��
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target) {
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//����ģ��
				break;
			}
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//����һ��ģ��
				start_time = clock();
			}
		}
		if (loss_function < loss_target) {
			break;//����ѵ��
		}
	}
#endif
	system("pause");
}
void user_nn_app_test(int argc, const char** argv) {
	user_nn_app_train(argc,argv);
}