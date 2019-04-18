
#include "user_nn_app.h"

void user_nn_app_train(int argc, const char** argv) {
	srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 784, //����� ��������ȡ��߶ȣ�
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
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//����ģ��
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//����ģ��
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	train_lables = user_nn_matrices_create(20000, 1, 1, 784);
	train_images = user_nn_matrices_create(20000, 1, 1, 784);
	//user_nn_matrices_init_vaule(rand_matrix_list,3,3);
	user_nn_matrix *images_matrix = train_images->matrix;
	user_nn_matrix *lables_matrix = train_lables->matrix;
	user_nn_matrix *kernel_matrix = user_nn_matrix_create(4, 4);//�������
	user_nn_matrix *same_matrix = NULL;//�������
	user_nn_matrix_memset(kernel_matrix, 0.9f);
	for (int count = 0; count < train_images->height*train_images->width; count++) {
		images_matrix->width = 28;
		images_matrix->height = 28;
		user_nn_matrix_paint_rectangle(images_matrix,
			(int)(user_nn_init_normal() * (images_matrix->width - 2)),
			(int)(user_nn_init_normal() * (images_matrix->height - 2)),
			(int)(user_nn_init_normal() * (images_matrix->width - 2)),
			(int)(user_nn_init_normal() * (images_matrix->height - 2)), 1.0f);//������
		same_matrix = user_nn_matrix_conv2(images_matrix, kernel_matrix, u_nn_conv2_type_same);
		images_matrix->width = 1;
		images_matrix->height = 784;
		same_matrix->width = 1;
		same_matrix->height = 784;
		user_nn_matrix_cpy_matrix(lables_matrix, same_matrix);
		user_nn_matrix_delete(same_matrix);

		images_matrix = images_matrix->next;
		lables_matrix = lables_matrix->next;
	}
	while (1) {
		for (int index = 0;index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_lables, index));//������������
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//����Ŀ������	
			user_nn_model_ffp(nn_layers);//�������һ��
			user_nn_model_bp(nn_layers, 0.01f);//�������һ��
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target) {
				user_nn_model_save_model(nn_layers,0);//����ģ��
				break;
			}
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(nn_layers,0);//����һ��ģ��
				start_time = clock();
			}
			
		}
		if (loss_function < loss_target) {
			break;//����ѵ��
		}
	}
	system("pause");
}
void user_nn_app_ident(int argc, const char** argv) {
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//����ģ��
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		system("pause");
		return ;
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
	printf("\n2.ʶ������");
	printf("\n���������֣�");
	switch (_getch()) {
	case '1':user_nn_app_train(argc, argv); break;
	case '2':user_nn_app_ident(argc, argv); break;
	default: break;
	}
}