
#include "user_mnist.h"

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


