
#include "user_cnn_app.h"

void user_cnn_mnist_train() {
	int user_layers[] = {
		'i', 28, 28, 1,//����� ��������ȡ��߶ȡ�������
		'c', 5, 5, 6,//����� ��������ȡ��߶ȡ�������
		's', 2, 2,//�Ӳ����� ��������ȡ��߶ȡ�������
		'c', 5, 5, 12,
		's', 2, 2,
		'f',//ȫ���Ӳ�---����ȫ���Ӳ�֮�� ѵ���ٶ����Խ������
		'o', 10//����� ���������������
	};
	bool sw_display = false;
	int save_model_count = 0;
	float loss_function = 0,target_loss= 0.001f;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----ѵ�����ӻ�-----\n");
	printf("\n1.����");
	printf("\n2.�رգ���������������");
	printf("\n���������֣�");
	sw_display = (_getch() == '1') ? true : false;
	//����mnist����
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	if (train_images == NULL) {
		printf("not found mnist files!");
		system("pause");
		return ;
	}
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(0);//����ģ��
	if (cnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		cnn_layers = user_cnn_model_create(user_layers);//����ģ��
	}
	user_cnn_model_info_layer(cnn_layers);
	start_time = clock();
	while (1) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_cnn_model_load_input_feature(cnn_layers, user_nn_matrices_ext_matrix_index(train_images, train_index), 1);
			user_cnn_model_load_target_feature(cnn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index));//����Ŀ�����
			user_cnn_model_ffp(cnn_layers);//�������һ��
			user_cnn_model_bp(cnn_layers, 0.01f);//����ѵ��һ��
			loss_function = user_cnn_model_return_loss(cnn_layers);//��ȡ��ʧ����
			//printf("\n%f", loss_function);
			if (sw_display) {
				user_cnn_model_display_feature(cnn_layers);//��ʾ������������
			}
			//�����ʧ����С������ֱֵ���˳�
			if (loss_function < target_loss) {
				break;//��������
			}
			if (save_model_count++ > 100) {
				save_model_count = 0;
				printf("train count:%d,loss:%f\n", train_index,loss_function);
				//user_cnn_model_save_model(cnn_layers,0);//����һ��ģ��
			}
		}
		printf("target:%f loss:%f\n", target_loss, loss_function);
		//�����ʧ����С������ֱֵ���˳�
		if (loss_function < target_loss) {
			break;//����ѵ��
		}
	}
	end_time = (clock() - start_time) / 1000 / 60;//��ȡ����ʱ��

	user_nn_debug_printf("%s","\nģ��ѵ����������ʧֵ:");
	user_nn_debug_printf("%f",(void *)&loss_function);
	user_nn_debug_printf("%s","��ʱ��:");
	user_nn_debug_printf("%d", (void *)&end_time);
	user_nn_debug_printf("%s","����");
	user_cnn_model_save_model(cnn_layers,0);//����ģ��
	printf("\n\n");
	system("pause");
}

void user_cnn_mnist_test() {
	char model_path[256] = "";
	sprintf_s(model_path, "%s\\%s", user_cnn_model_get_exe_path(), user_nn_model_cnn_file_name);
	printf("%s\n", model_path);
	user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(0);//����ģ��
	if (cnn_layers == NULL) {
		printf("\n����ģ��ʧ��!\n\n");
		system("pause");
		return;
	}
	//���в���
	float error_count = 0;
	for (int test_index = 0; test_index < test_images->height * test_images->width; test_index++) {
		user_cnn_model_load_input_feature(cnn_layers, user_nn_matrices_ext_matrix_index(test_images, test_index), 1);
		user_cnn_model_ffp(cnn_layers);//�������һ��
		if (user_cnn_model_return_class(cnn_layers) != user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index))) {
			error_count++;
			user_nn_debug_printf("%s","\nʶ�����ͼ������:");
			user_nn_debug_printf("%d", (void *)user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index)));
			user_nn_debug_printf("%s","ʶ��Ϊ:");
			user_nn_debug_printf("%s", (void *)user_cnn_model_return_class(cnn_layers));
		}
	}
	user_nn_debug_printf("%s","\n\nʶ��ɹ���:");
	error_count = ((float)1 - (float)error_count / (test_images->height * test_images->width)) * 100;
	user_nn_debug_printf("%d", (void *)&error_count);
	user_nn_debug_printf("%s","%");
	system("pause");
}


bool user_cnn_load_ident(int argc, const char** argv) {
	char model_path[256] = "";
	sprintf_s(model_path, "%s\\%s", user_cnn_model_get_exe_path(), user_nn_model_cnn_file_name);
	printf("%s\n", model_path);
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(0);//����ģ��
	if (cnn_layers != NULL) {
		printf("loading model success\n");
		if (argv[1] == NULL) {
			printf("path error\n");
		}
		else {
			printf("\n%s\n", argv[1]);
			const char *full_path = argv[1];
			user_cnn_model_load_input_image(cnn_layers, (char *)full_path, 1);//����ͼ���������ĵ�һ��������
			user_cnn_model_ffp(cnn_layers);//ʶ��
			printf("ʶ����Ϊ��%d\n", user_cnn_model_return_class(cnn_layers));
			user_nn_matrix_printf(NULL, ((user_cnn_output_layers *)user_cnn_model_return_layer(cnn_layers, u_cnn_layer_type_output)->content)->feature_matrix);//��ӡ����
		}
		getchar();
		return true;
	}
	else {
		printf("loading model faile\n");
		return false;
	}
}

void user_cnn_app_test(int argc, const char** argv) {
	printf("\n-----����ѡ��-----\n");
	printf("\n1.ѵ��mnist����");
	printf("\n2.����mnist����");
	printf("\n3.ʶ��28*28ͼ��\n");
	printf("\n���������֣�");
	switch (_getch()) {
		case '1':user_cnn_mnist_train(); break;
		case '2':user_cnn_mnist_test(); break;
		case '3':user_cnn_load_ident(argc, argv); break;
		default: break;
	}
}