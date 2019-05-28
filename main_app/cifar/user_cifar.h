#ifndef _user_cifar_H
#define _user_cifar_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"
#include "../matrix/user_nn_matrix_file.h"

typedef struct _mnist_heard{
	unsigned char lable_id;//标志
	unsigned char red_channel[1024];
	unsigned char green_channel[1024];
	unsigned char blue_channel[1024];
}user_cifar_header;

void cifar_conv_list_matrix(char *file_name);//转化mnist数据为矩阵数据

#endif
