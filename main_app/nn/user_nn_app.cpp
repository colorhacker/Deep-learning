
#include "user_nn_app.h"

#include "..\other\user_nn_opencv.h"

user_nn_matrix *user_nn_app_set_input(void) {
	user_nn_matrix *src_matrix = user_nn_matrix_create(1, 1);//�������
	src_matrix->data[0] = 1;
	return src_matrix;
}

void train_mnist_gen_network(int argc, const char** argv) {
	int gen_layers[] = { 'i',1,10,'h',392,'o',784 };
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_layers *nn_gen_layers = user_nn_model_load_model(1);
	if (nn_gen_layers == NULL) {
		nn_gen_layers = user_nn_model_create(gen_layers);
		while (1) {
			for (int index = 0; index < train_images->height * train_images->width; index++) {
				user_nn_model_load_input_feature(nn_gen_layers, user_nn_matrices_ext_matrix_index(train_lables, index));
				user_nn_model_load_target_feature(nn_gen_layers, user_nn_matrices_ext_matrix_index(train_images, index));
				user_nn_model_ffp(nn_gen_layers);
				user_nn_model_bp(nn_gen_layers, 0.01f);
				if (index % 1000 == 0) {
					printf("\ncount:%d,loss:%f", index, user_nn_model_return_loss(nn_gen_layers));
				}
				user_nn_model_display_feature(nn_gen_layers);//��ʾͼ��
			}
			printf("\nloss:%f", user_nn_model_return_loss(nn_gen_layers));
			user_nn_model_save_model(nn_gen_layers, 1);
		}
	}
	_getch();
}

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
		'h', 392, //������ ���� ���߶ȣ�
		'o', 10 //����� ���� ���߶ȣ�
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0, exit_train_count = 100000;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----ѵ�����ӻ�-----\n");
	printf("\n1.����");
	printf("\n2.�رգ���������������");
	printf("\n���������֣�");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//����ģ��
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//����ģ��
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	user_nn_matrix *input_matirx = user_nn_app_set_input();
	while (1){
		for (int index = 0; index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//������������
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_lables, index));//����Ŀ������	
			user_nn_model_ffp(nn_layers);//�������һ��
			user_nn_model_bp(nn_layers, 0.01f);//�������һ��
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target || exit_train_count-- <= 0) {
				user_nn_model_save_model(nn_layers, 0);//����ģ��
				break;
			}
			printf("\ntarget:%f loss:%f", loss_target, loss_function);
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(nn_layers, 0);//����һ��ģ��
				start_time = clock();
			}
		}
		if (loss_function <= loss_target || exit_train_count-- <= 0) {
			break;
		}
	}
}
void user_nn_app_ident(int argc, const char** argv) {
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//����ģ��
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		system("pause");
		return;
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
	printf("\n-----����ѡ��-----\n");
	printf("\n1.ѵ������");
	printf("\n2.��������");
	printf("\n3.ʶ������");
	printf("\n���������֣�");
	switch (_getch()) {
	case '1':user_nn_app_train(argc, argv); break;
	case '2':train_mnist_gen_network(argc, argv); break;
	case '3':user_nn_app_ident(argc, argv); break;
	default: break;
	}
}