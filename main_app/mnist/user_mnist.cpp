
#include "user_mnist.h"

user_nn_matrix *numpy_load(char *file_name) {
	FILE *file_handle = NULL;
	user_numpy_npy_header *numpy_header = NULL;
	char data_type = 0, *descr_data;
	int data_size = 0, data_shape[2];
	numpy_header = (user_numpy_npy_header *)malloc(sizeof(user_numpy_npy_header));//����ռ�
	fopen_s(&file_handle, file_name, "rb");
	fread(numpy_header, sizeof(user_numpy_npy_header), 1, file_handle);//��ȡͷ��Ϣ
	descr_data = (char *)malloc(numpy_header->heade_len_descr + 1);//����ռ� ����ֽڱ��������
	memset(descr_data, 0, numpy_header->heade_len_descr + 1);
	fread(descr_data, numpy_header->heade_len_descr, 1, file_handle);//��Ч��ͷ��Ϣ��16����

	sscanf(descr_data,"{'descr': '<%c%d', 'fortran_order': False, 'shape': (%d, %d), }", &data_type, &data_size, &data_shape[0], &data_shape[1]);
	printf("\nmagic_string[0]:%02x", numpy_header->magic_string[0]);
	printf("\nmagic_string[1-5]:%c", numpy_header->magic_string[1]);
	printf("\nmajor_version:%d", numpy_header->major_version);
	printf("\nminor_version:%d", numpy_header->minor_version);
	printf("\nheade_len_descr:%d", numpy_header->heade_len_descr);
	printf("\nend_data:%s", descr_data);
	printf("\ndata_type:%c%d data_shape:(%d %d)", data_type, data_size, data_shape[0], data_shape[1]);
	free(descr_data);

	if (data_size != 4) {
		return NULL;
	}
	user_nn_matrix *result = user_nn_matrix_create(data_shape[1], data_shape[0]);
	fread(result->data, data_shape[0]* data_shape[1]* data_size, 1, file_handle);
	
	return result;
}

void mnist_conv_list_matrix(char *file_name) {
	char save_file_name[258] = "";
	float *matrix_data = NULL;
	FILE *file_handle = NULL;
	user_mnist_header *mnist_header = NULL;
	user_nn_list_matrix *mnist_data_list = NULL;//������������
	mnist_header = (user_mnist_header *)malloc(sizeof(user_mnist_header));//����ռ�
	
	fopen_s(&file_handle, file_name, "rb");
	fread(mnist_header, sizeof(user_mnist_header), 1, file_handle);//��ȡͷ��Ϣ
	mnist_header->magic_number = _byteswap_ulong(mnist_header->magic_number);//��С��ת��
	if (mnist_header->magic_number == 0x00000801) {//��ǩTRAINING SET LABEL FILE (train-labels-idx1-ubyte):
		mnist_header->number_of_items = _byteswap_ulong(mnist_header->number_of_items);//��С��ת��
		mnist_data_list = user_nn_matrices_create(mnist_header->number_of_items, 1, 1, 10);
		fseek(file_handle,8, SEEK_SET);
		for (int index = 0; index < mnist_header->number_of_items; index++) {
			//matrix_data = user_nn_matrices_ext_matrix_index(mnist_data_list, index)->data;
			//*matrix_data = (float)fgetc(file_handle);
			*(user_nn_matrices_ext_matrix_index(mnist_data_list, index)->data + (unsigned char)fgetc(file_handle)) = 1;
		}
	}
	if (mnist_header->magic_number == 0x00000803) {//ͼ��TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
		mnist_header->number_of_items = _byteswap_ulong(mnist_header->number_of_items);//��С��ת��
		mnist_header->number_of_columns = _byteswap_ulong(mnist_header->number_of_columns);//��С��ת��
		mnist_header->number_of_rows = _byteswap_ulong(mnist_header->number_of_rows);//��С��ת��
		mnist_data_list = user_nn_matrices_create(mnist_header->number_of_items, 1, mnist_header->number_of_rows, mnist_header->number_of_columns);
		for (int index = 0; index < mnist_header->number_of_items; index++) {
			matrix_data = user_nn_matrices_ext_matrix_index(mnist_data_list, index)->data;
			for (int size = 0; size < mnist_header->number_of_columns*mnist_header->number_of_rows; size++) {
				*matrix_data++ = (float)(fgetc(file_handle) / 255.0);
			}
		}
	}
	if (mnist_data_list != NULL) {
		sprintf(save_file_name, "%s.bx", file_name);
		user_nn_model_file_save_matrices(save_file_name, 0, mnist_data_list);
		user_nn_matrices_delete(mnist_data_list);
		printf("\n%s to %s\n", file_name, save_file_name);
	}
	else {
		printf("\n%s error\n", file_name);
	}
	fclose(file_handle);
}

//��ȡͼ����� ����ָ����С��step����������ȡ
//src_matrix Դ����
//f_width ��Ҫ���ɵľ�����
//f_height ��Ҫ���ɾ���ĸ߶�
//step ����ÿ���ƶ���С
//������������
user_nn_list_matrix *user_nn_matrix_generate_feature(user_nn_list_matrix *save_featrue, user_nn_matrix *src_matrix, int f_width, int f_height, int step) {
	user_nn_list_matrix *featrue_list = save_featrue == NULL ? user_nn_matrices_create_head(1, 1) : save_featrue;//�����ھʹ���
	for (int height = 0; height < src_matrix->height - f_height + 1; height += step) {
		for (int width = 0; width < src_matrix->width - f_width + 1; width += step) {
			user_nn_matrices_add_matrix(featrue_list, user_nn_matrix_ext_matrix(src_matrix, width, height, f_width, f_height));
			//printf("\n x:%d,y:%d,w:%d,h:%d", width, height, f_width, f_height);
		}
	}
	return featrue_list;
}
//ͨ��k-means���ľ��������µľ�������
//class_featrue k-means�������ľ���
//src_matrix ��Ҫ���ع��ľ���
//w_step ���ÿ�β��ƶ�����
//h_step �߶�ÿ�β��ƶ�����
//���� �ع���ľ���
user_nn_matrix *user_nn_matrix_kmeans_paste_refactor(user_nn_list_matrix *class_featrue, user_nn_matrix *src_matrix, int w_step, int h_step) {
	user_nn_matrix *matrix_temp = NULL;
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);//��������
	for (int height = 0; height < src_matrix->height - class_featrue->matrix->height + 1; height += h_step) {
		for (int width = 0; width < src_matrix->width - class_featrue->matrix->width + 1; width += w_step) {
			matrix_temp = user_nn_matrix_ext_matrix(src_matrix, width, height, class_featrue->matrix->width, class_featrue->matrix->height);//��ȡָ������
			user_nn_matrix_add_paste_matrix(result, user_nn_matrices_ext_matrix_index(class_featrue, user_nn_matrix_k_means_discern(class_featrue, matrix_temp)), width, height);//ճ��ָ������
			user_nn_matrix_delete(matrix_temp);//ɾ���������
		}
	}
	return result;
}
//�ָ�mnist����ͼ��
void user_nn_mnist_cut_matrices(int c_width, int c_heigth, int step) {
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_list_matrix *featrue_list = NULL;//����ͷ
	for (int count = 0; count < train_images->width*train_images->height; count++) {
		featrue_list = user_nn_matrix_generate_feature(featrue_list, user_nn_matrices_ext_matrix_index(train_images, count), c_width, c_heigth, step);//�ָ�ͼ��
		printf("\n::%d", count);
	}
	user_nn_model_file_save_matrices("./mnist/files/train-images.idx3-ubyte.bx.c", 0, featrue_list);//�������
}

