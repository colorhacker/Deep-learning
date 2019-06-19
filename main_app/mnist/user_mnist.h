#ifndef _user_mnist_H
#define _user_mnist_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_matrix_file.h"

typedef struct _mnist_heard{
	int magic_number;//标志 0x00000803(2051)
	int number_of_items;//60000
	int number_of_rows;//
	int number_of_columns;//
}user_mnist_header;

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