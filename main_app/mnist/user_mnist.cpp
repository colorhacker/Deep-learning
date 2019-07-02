
#include "user_mnist.h"

user_nn_matrix *numpy_load(char *file_name) {
	FILE *file_handle = NULL;
	user_numpy_npy_header *numpy_header = NULL;
	char data_type = 0, *descr_data;
	int data_size = 0, data_shape[2];
	numpy_header = (user_numpy_npy_header *)malloc(sizeof(user_numpy_npy_header));//分配空间
	fopen_s(&file_handle, file_name, "rb");
	fread(numpy_header, sizeof(user_numpy_npy_header), 1, file_handle);//读取头信息
	descr_data = (char *)malloc(numpy_header->heade_len_descr + 1);//分配空间 最后字节保存结束符
	memset(descr_data, 0, numpy_header->heade_len_descr + 1);
	fread(descr_data, numpy_header->heade_len_descr, 1, file_handle);//有效的头信息被16整除

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
	user_nn_list_matrix *mnist_data_list = NULL;//创建连续矩阵
	mnist_header = (user_mnist_header *)malloc(sizeof(user_mnist_header));//分配空间
	
	fopen_s(&file_handle, file_name, "rb");
	fread(mnist_header, sizeof(user_mnist_header), 1, file_handle);//读取头信息
	mnist_header->magic_number = _byteswap_ulong(mnist_header->magic_number);//大小端转化
	if (mnist_header->magic_number == 0x00000801) {//标签TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
		mnist_header->number_of_items = _byteswap_ulong(mnist_header->number_of_items);//大小端转化
		mnist_data_list = user_nn_matrices_create(mnist_header->number_of_items, 1, 1, 10);
		fseek(file_handle,8, SEEK_SET);
		for (int index = 0; index < mnist_header->number_of_items; index++) {
			//matrix_data = user_nn_matrices_ext_matrix_index(mnist_data_list, index)->data;
			//*matrix_data = (float)fgetc(file_handle);
			*(user_nn_matrices_ext_matrix_index(mnist_data_list, index)->data + (unsigned char)fgetc(file_handle)) = 1;
		}
	}
	if (mnist_header->magic_number == 0x00000803) {//图像TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
		mnist_header->number_of_items = _byteswap_ulong(mnist_header->number_of_items);//大小端转化
		mnist_header->number_of_columns = _byteswap_ulong(mnist_header->number_of_columns);//大小端转化
		mnist_header->number_of_rows = _byteswap_ulong(mnist_header->number_of_rows);//大小端转化
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

//提取图像矩阵 按照指定大小与step进行数据提取
//src_matrix 源矩阵
//f_width 需要生成的矩阵宽度
//f_height 需要生成矩阵的高度
//step 矩阵每次移动大小
//返回特征矩阵
user_nn_list_matrix *user_nn_matrix_generate_feature(user_nn_list_matrix *save_featrue, user_nn_matrix *src_matrix, int f_width, int f_height, int step) {
	user_nn_list_matrix *featrue_list = save_featrue == NULL ? user_nn_matrices_create_head(1, 1) : save_featrue;//不存在就创建
	for (int height = 0; height < src_matrix->height - f_height + 1; height += step) {
		for (int width = 0; width < src_matrix->width - f_width + 1; width += step) {
			user_nn_matrices_add_matrix(featrue_list, user_nn_matrix_ext_matrix(src_matrix, width, height, f_width, f_height));
			//printf("\n x:%d,y:%d,w:%d,h:%d", width, height, f_width, f_height);
		}
	}
	return featrue_list;
}
//通过k-means中心矩阵生成新的矩阵数据
//class_featrue k-means分类中心矩阵
//src_matrix 需要被重构的矩阵
//w_step 宽度每次步移动距离
//h_step 高度每次步移动距离
//返回 重构后的矩阵
user_nn_matrix *user_nn_matrix_kmeans_paste_refactor(user_nn_list_matrix *class_featrue, user_nn_matrix *src_matrix, int w_step, int h_step) {
	user_nn_matrix *matrix_temp = NULL;
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);//创建矩阵
	for (int height = 0; height < src_matrix->height - class_featrue->matrix->height + 1; height += h_step) {
		for (int width = 0; width < src_matrix->width - class_featrue->matrix->width + 1; width += w_step) {
			matrix_temp = user_nn_matrix_ext_matrix(src_matrix, width, height, class_featrue->matrix->width, class_featrue->matrix->height);//截取指定矩阵
			user_nn_matrix_add_paste_matrix(result, user_nn_matrices_ext_matrix_index(class_featrue, user_nn_matrix_k_means_discern(class_featrue, matrix_temp)), width, height);//粘贴指定矩阵
			user_nn_matrix_delete(matrix_temp);//删除缓冲矩阵
		}
	}
	return result;
}
//分割mnist矩阵图像
void user_nn_mnist_cut_matrices(int c_width, int c_heigth, int step) {
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_list_matrix *featrue_list = NULL;//创建头
	for (int count = 0; count < train_images->width*train_images->height; count++) {
		featrue_list = user_nn_matrix_generate_feature(featrue_list, user_nn_matrices_ext_matrix_index(train_images, count), c_width, c_heigth, step);//分割图像
		printf("\n::%d", count);
	}
	user_nn_model_file_save_matrices("./mnist/files/train-images.idx3-ubyte.bx.c", 0, featrue_list);//保存矩阵
}

