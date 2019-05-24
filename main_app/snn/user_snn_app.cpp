
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
	int user_layers[] = {
		'i', 1, 2, //����� ��������ȡ��߶ȣ�
		'h', 2, //������ ���� ���߶ȣ�
		'h', 2, //������ ���� ���߶ȣ�
		'o', 2 //����� ���� ���߶ȣ�
	};
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	user_snn_layers *snn_layers = user_snn_model_load_model(0);//����ģ��
	if (snn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		snn_layers = user_snn_model_create(user_layers);//����ģ��
	}
	user_nn_matrix *src_matrix = user_nn_matrix_create(1, 2);//�������
	src_matrix->data[0] = 1.5f;
	src_matrix->data[1] = 0.5f;
	for (;;) {
		for (int train_index = 0; train_index < 1000; train_index++) {
			user_snn_model_load_input_feature(snn_layers, src_matrix);//������������
			user_snn_model_load_target_feature(snn_layers, src_matrix);//����Ŀ������	
			user_snn_model_ffp(snn_layers);
			user_snn_model_bp(snn_layers);
		}
		break;
	}
	user_snn_model_ffp(snn_layers);
	user_snn_output_layers *snn_output_layer = (user_snn_output_layers *)user_snn_model_return_layer(snn_layers, u_snn_layer_type_output)->content;
	user_nn_matrix_printf(NULL, snn_output_layer->feature_matrix);
	//for (;;) {
	//	for (int train_index = 0; train_index < train_images->height * train_images->width; train_index++) {
	//		user_snn_model_load_input_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_images, train_index));//������������
	//		user_snn_model_load_target_feature(snn_layers, user_nn_matrices_ext_matrix_index(train_lables, train_index));//����Ŀ������	
	//		user_snn_model_ffp(snn_layers);
	//		user_snn_model_bp(snn_layers);
	//	}
	//}
	system("pause");
}
void user_snn_app_ident(int argc, const char** argv) {

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

