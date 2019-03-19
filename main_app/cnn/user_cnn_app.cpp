
#include "user_cnn_app.h"

const char *files_list[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

void cnn_detect(int argc, const char** argv) {
	bool sw_display = false;
	//char scanf_char;
	printf("\n�Ƿ���ѵ��������ʾ����ʼ������ʾѵ��ʱ��ϳ�����\n ���� y �س����п�����ֱ�ӻس�Ĭ�Ϲر�\n\n");
	sw_display = (getchar() == 'y') ? true : false;

	int user_layers[] = {
		'i', 28, 28, 1,//����� ��������ȡ��߶ȡ�������
		'c', 5, 5, 6,//����� ��������ȡ��߶ȡ�������
		's', 2, 2,//�Ӳ����� ��������ȡ��߶ȡ�������
		'c', 5, 5, 12,
		's', 2, 2,
		'f',//ȫ���Ӳ�---����ȫ���Ӳ�֮�� ѵ���ٶ����Խ������
		'o', 10//����� ���������������
	};

	char *full_path = NULL;
	bool model_is_exist = false;
	float loss_function = 0;
	//����mnist����
	user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	if (train_images == NULL) {
		printf("not found mnist files!");
		system("pause");
		return ;
	}

	srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(user_nn_model_cnn_file_name);//����ģ��
	if (cnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		cnn_layers = user_cnn_model_create(user_layers);//����ģ��
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}

	clock_t start_time = clock();
	int loss_info = 0;
	//����ѵ��
	while (!model_is_exist) {
		for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
			user_cnn_model_load_input_mnist(train_images, train_index, cnn_layers, 1);//����ͼ��
			user_cnn_model_ffp(cnn_layers);//�������һ��
			user_cnn_model_bp(cnn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index), 0.5f);//����ѵ��һ��
			loss_function = user_cnn_model_return_loss(cnn_layers);//��ȡ��ʧ����
			if (sw_display) {
				user_cnn_model_display_feature(cnn_layers);//��ʾ������������
			}
			//�����ʧ����С������ֱֵ���˳�
			if (loss_function < 0.001f) {
				break;//��������
			}
			if (loss_info++ > 1000) {
				loss_info = 0;
				printf("train count:%d,loss:%f\n", train_index,loss_function);
			}
		}
		printf("target loss:0.001,loss:%f\n", loss_function);
		//�����ʧ����С������ֱֵ���˳�
		if (loss_function < 0.001f) {
			break;//����ѵ��
		}
	}
	clock_t end_time = (clock() - start_time) / 1000 / 60;//��ȡ����ʱ��
	user_model_save_string("\nģ��ѵ����������ʧֵ:");
	user_model_save_float(loss_function);
	user_model_save_string("\n��ʱ��:");
	user_model_save_int(end_time);
	user_model_save_string("����");
	//���в���
	int error_count = 0;
	for (int test_index = 0; test_index < test_images->height * test_images->width; test_index++) {
		user_cnn_model_load_input_mnist(test_images, test_index, cnn_layers, 1);//����ͼ��
		user_cnn_model_ffp(cnn_layers);//�������һ��
		if (user_cnn_model_return_class(cnn_layers) != user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index))) {
			error_count++;
			user_model_save_string("\n\nʶ�����\n �ļ�·��:");
			user_model_save_string("\n������ֵΪ:");
			user_model_save_int(user_nn_matrix_return_max_index(user_nn_matrices_ext_matrix_index(test_lables, test_index)));
			user_model_save_string("\nʶ��Ϊ:");
			user_model_save_int(user_cnn_model_return_class(cnn_layers));
		}
	}
		
	user_model_save_string("\n\nʶ��ɹ���:");
	user_model_save_float(((float)1 - (float)error_count / 500) * 100);
	user_model_save_string("%");
	user_cnn_model_save_model(user_nn_model_cnn_file_name, cnn_layers);//����ģ��
	//system("pause");
}

bool cnn_test(int argc, const char** argv) {
	/*
	Mat a = Mat::ones(2048, 2048, CV_32FC1);
	Mat b = Mat::ones(2048, 2048, CV_32FC1);
	Mat c;
	printf("start:\n");
	InitMat(a, 2.5f);
	InitMat(b, 1.0f);
	clock_t test_start_time = clock();
	c = a*b;
	clock_t test_end_time = (clock() - test_start_time) / 1000;//��ȡ����ʱ��
	printf("total time : %d\n", test_end_time);
	*/
	char model_path[256] = "";
	sprintf_s(model_path, "%s\\%s", user_cnn_model_get_exe_path(), user_nn_model_cnn_file_name);
	printf("%s\n", model_path);
	//printf("%s\n", argv[0]);//�����exe�ļ�·��
	user_cnn_layers *cnn_layers = user_cnn_model_load_model(model_path);//����ģ��
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
			printf("ʶ����Ϊ��%d,����Ϊ��%s\n", user_cnn_model_return_class(cnn_layers), files_list[user_cnn_model_return_class(cnn_layers)]);
			//
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
	if (cnn_test(argc, argv) == false) {
		cnn_detect(argc, argv);
	}
}