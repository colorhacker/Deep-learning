#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"

int main(int argc, const char** argv){
#ifdef _OPENMP
	omp_set_num_threads(28);
#endif
/*
#pragma omp parallel for 
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			for (int z = 0; z < 3; z++) {
				printf("\n%d:%d %d %d",x*3*3+y*3+z,x,y,z);
			}
		}
	}
*/
	_getch();
	return 1;
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
