#ifndef _user_mnist_H
#define _user_mnist_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_matrix_file.h"

typedef struct _numpy_npy {
	char magic_string[6];//标志 0x93+NUMPY
	char major_version;//0x01 0x02 0x03
	char minor_version;//0x00
	unsigned short heade_len_descr;//little-endian : unsigned short. = len(magic string) + 2 + len(length) + HEADER_LEN
}user_numpy_npy_header;

typedef struct _mnist_heard{
	int magic_number;//标志 0x00000803(2051)
	int number_of_items;//60000
	int number_of_rows;//
	int number_of_columns;//
}user_mnist_header;

user_nn_matrix *numpy_load(char *file_name);

void mnist_conv_list_matrix(char *file_name);//转化mnist数据为矩阵数据

#endif


/*
int main(int argc, const char** argv){
mnist_conv_list_matrix("./mnist/files/t10k-labels.idx1-ubyte");//转化测试lable
mnist_conv_list_matrix("./mnist/files/t10k-images.idx3-ubyte");//转化测试图像
mnist_conv_list_matrix("./mnist/files/train-labels.idx1-ubyte");//转化训练lable
mnist_conv_list_matrix("./mnist/files/train-images.idx3-ubyte");//转化训练图像

user_nn_list_matrix *test_lables = user_nn_model_file_read_matrices("./mnist/files/t10k-labels.idx1-ubyte.bx", 0);
user_nn_list_matrix *test_images = user_nn_model_file_read_matrices("./mnist/files/t10k-images.idx3-ubyte.bx", 0);
user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

for (int index = 0; index < 20; index++) {
user_nn_matrix_printf(NULL, user_nn_matrices_ext_matrix_index(train_lables, index));
user_opencv_show_matrix("test_image:0", user_nn_matrices_ext_matrix_index(train_images, index),100,100,1);
getchar();
}
}
*/
/*
user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

user_nn_list_matrix *featrue_list = NULL;//创建头
user_nn_list_matrix *kclass_list = NULL;
user_nn_list_matrix *kclass_list_temp = NULL;

int cut_width = 7;//剪切宽度
int cut_heigth = 7;//剪切高度
int cut_step = 1;//剪切精度
int class_size = 255;//分类大小
					 //for (int count = 0; count < train_images->width*train_images->height; count++) {
for (int count = 0; count < 1000; count++) {
	featrue_list = user_nn_matrix_generate_feature(NULL, user_nn_matrices_ext_matrix_index(train_images, count), cut_width, cut_heigth, cut_step);//分割图像
	kclass_list = user_nn_matrix_k_means(kclass_list, featrue_list, class_size, 100);
	user_nn_matrices_delete(featrue_list);
	printf("::%d\n", count);
}
printf("k class size:%d\n", kclass_list->height * kclass_list->width);
for (int count = 1000; count < 2000; count++) {
	user_nn_matrix *matrix_temp = user_nn_matrix_kmeans_paste_refactor(kclass_list, user_nn_matrices_ext_matrix_index(train_images, count), cut_heigth, cut_step);
	user_opencv_show_matrix("f:1", user_nn_matrices_ext_matrix_index(train_images, count), 100, 100, 1);
	user_opencv_show_matrix("f:0", matrix_temp, 300, 100, 1);
	_getch();
	user_nn_matrix_delete(matrix_temp);
}
_getch();
return 0;
*/