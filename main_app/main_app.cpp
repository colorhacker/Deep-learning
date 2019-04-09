﻿#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"

#include "other\rgb_hsl.h"

int main(int argc, const char** argv){
	user_nn_rgb rgb = { 0xFF,0xA0,0x23 };
	user_nn_hsl hsl = {0,0,0};

	RGB_to_HSL(&rgb, &hsl);
	HSL_to_RGB(&hsl, &rgb);
	return 0;

#ifdef _OPENMP
	omp_set_num_threads(64);
#endif
/*	user_nn_matrix *src_matrix = NULL;
	user_nn_matrix *sub_matrix = NULL;
	user_nn_matrix *res_matrix = NULL;

	src_matrix = user_nn_matrix_create(5, 8);
	sub_matrix = user_nn_matrix_create(5, 8);

	for (int count = 0; count < (src_matrix->width * src_matrix->height); count++) {
		src_matrix->data[count] = (float)count * 0.1f;	
	}
	for (int count = 0; count < (sub_matrix->width * sub_matrix->height); count++) {
		sub_matrix->data[count] = user_nn_init_glorot_uniform(3,3); 
		//sub_matrix->data[count] = (float)(((float)(rand()*2.0 / RAND_MAX) - 1.0f))*sqrt(6.0 / 6.0);
	}
	if (sub_matrix != NULL) {
		user_nn_matrix_printf(NULL, sub_matrix);//打印矩阵
	}
	else {
		printf("null\n");
	}
	getchar();
	return 1;*/

	printf("\n-----功能选择-----\n");
	printf("\n1.cnn测试");
	printf("\n2.rnn测试");
	printf("\n3.nn测试\n");
	//printf("\n随机码：%d\n", (unsigned)time(NULL));
	srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	printf("\n请输入数字：");
	switch (_getch()) {
		case '1':user_cnn_app_test(argc, argv); break;
		case '2':user_rnn_app_test(argc, argv); break;
		case '3':user_nn_app_test(argc, argv); break;
		default: break;
	}
	_getch();
	return 0;
}
