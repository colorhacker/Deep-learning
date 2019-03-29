
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
		'h', 784, //������ ���� ���߶ȣ�
		'h', 784, //������ ���� ���߶ȣ�
		'o', 784 //����� ���� ���߶ȣ�
	};

	float loss_function = 1.0f,loss_target= 0.0003f;
	int save_model_count = 0;
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//����ģ��
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//����ģ��
	}
	user_nn_model_info_layer(nn_layers);
	while (1) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//������������
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//����Ŀ������	
			user_nn_model_ffp(nn_layers);//�������һ��
			user_nn_model_bp(nn_layers, 0.01f);//�������һ��
			loss_function = user_nn_model_return_loss(nn_layers);
			//user_nn_model_display_feature(nn_layers);
			if (loss_function <= loss_target) {
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//����ģ��
				break;
			}
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				printf("\ntarget:%f loss:%f", loss_target, loss_function);
				user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//����һ��ģ��
			}
		}
		if (loss_function < loss_target) {
			break;//����ѵ��
		}
	}
	system("pause");
}
void user_nn_app_test(int argc, const char** argv) {
	user_nn_app_train(argc,argv);
}