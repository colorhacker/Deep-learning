
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	//srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 28, 28, //����� ��������ȡ��߶ȣ�
		'h', 28, //������ ���� ���߶ȣ�
		'o', 28 //����� ���� ���߶ȣ�
	};

	float loss_function = 1.0f;
	bool model_is_exist = false;
	//����mnist����
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//����ģ��
	user_nn_matrix *input_mnist_data = user_nn_matrix_create(28,28);
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		nn_layers = user_nn_model_create(user_layers);//����ģ��
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	if (model_is_exist == false) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_matrix_cpy_matrix(input_mnist_data, user_nn_matrices_ext_matrix_index(train_images, index));
			//user_nn_matrix_divi_constant(input_mnist_data, 255.0);
			user_nn_model_load_input_feature(nn_layers, input_mnist_data);//������������
			user_nn_model_load_target_feature(nn_layers, input_mnist_data);//����Ŀ������									   
			user_nn_model_ffp(nn_layers);//�������һ��
			user_nn_model_bp(nn_layers, 0.5f);//�������һ��
			loss_function = user_nn_model_return_loss(nn_layers);
			user_nn_model_display_feature(nn_layers);
			printf("\n%d loss:%f",  index, loss_function);
			if (loss_function <= 0.000001f) {
				//user_nn_model_save_model(user_nn_model_nn_file_name, nn_layers);//����ģ��
				break;
			}
		}
	}
	system("pause");
}
