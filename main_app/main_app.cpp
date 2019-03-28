#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"


int main(int argc, const char** argv){
	user_nn_matrix *src_matrix = NULL;
	user_nn_matrix *sub_matrix = NULL;
	user_nn_matrix *res_matrix = NULL;

	src_matrix = user_nn_matrix_create(10, 10);
	sub_matrix = user_nn_matrix_create(10, 10);

	user_nn_matrix_memset(src_matrix, 2.5);//设置矩阵值
	user_nn_matrix_memset(sub_matrix, 1.0);//设置矩阵值


	clock_t start_time;
	start_time = clock();
	res_matrix = user_nn_matrix_mult_matrix(src_matrix, sub_matrix);//矩阵相乘
	if (res_matrix != NULL) {
		user_nn_matrix_printf(NULL, res_matrix);//打印矩阵
	}
	else {
		printf("null\n");
	}
	printf("%f", (float)(clock() - start_time) / 1000);
	return 1;
	//omp_set_num_threads(16);
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
