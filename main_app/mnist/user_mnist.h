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
