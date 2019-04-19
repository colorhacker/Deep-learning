#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"
#include "other\user_nn_opencv.h"

void user_nn_app_set_data() {
	user_nn_list_matrix *train_lables = user_nn_matrices_create(20000, 1, 1, 784);
	user_nn_list_matrix *train_images = user_nn_matrices_create(20000, 1, 1, 784);
	user_nn_matrix *images_matrix = train_images->matrix;
	user_nn_matrix *lables_matrix = train_lables->matrix;
	user_nn_matrix *kernel_matrix = user_nn_matrix_create(2, 2);//卷积矩阵
	user_nn_matrix *same_matrix1 = NULL;//卷积矩阵
	user_nn_matrix *same_matrix2 = NULL;//卷积矩阵
	user_nn_matrix *temp_matrix1 = user_nn_matrix_create(28, 28);//卷积矩阵
	user_nn_matrix *temp_matrix2 = user_nn_matrix_create(28, 28);//卷积矩阵
	user_nn_matrix_memset(kernel_matrix, 0.9f);
	for (int count = 0; count < train_images->height*train_images->width; count++) {
		user_nn_matrix_memset(temp_matrix1, 0.0f);
		user_nn_matrix_memset(temp_matrix2, 0.0f);
		user_nn_matrix_paint_rectangle(temp_matrix1,
			(int)(user_nn_init_normal() * 26),
			(int)(user_nn_init_normal() * 26),
			(int)(user_nn_init_normal() * 26),
			(int)(user_nn_init_normal() * 26), 1.0f);//画矩形

		user_nn_matrix_cpy_matrix(temp_matrix2, temp_matrix1);
		int x, y, mx, my, min;
		x = (int)(user_nn_init_normal() * 26);
		y = (int)(user_nn_init_normal() * 26);
		mx = 26 - x;
		my = 26 - y;
		min = x;
		min = min < y ? min : y;
		min = min <mx ? min : mx;
		min = min < my ? min : my;
		min = min > 0 ? min : 1;
		user_nn_matrix_paint_circle(temp_matrix1, x, y, min, 1.0f);//画圆

		same_matrix1 = user_nn_matrix_conv2(temp_matrix1, kernel_matrix, u_nn_conv2_type_same);
		user_nn_matrix_memcpy(images_matrix, same_matrix1->data);
		user_nn_matrix_delete(same_matrix1);

		same_matrix2 = user_nn_matrix_conv2(temp_matrix2, kernel_matrix, u_nn_conv2_type_same);
		user_nn_matrix_memcpy(lables_matrix, same_matrix2->data);
		user_nn_matrix_delete(same_matrix2);

		user_opencv_show_matrix("a", images_matrix, 100, 100,1);
		user_opencv_show_matrix("b", lables_matrix, 500, 100,1);
		_getch();
		images_matrix = images_matrix->next;
		lables_matrix = lables_matrix->next;

	}
}
int main(int argc, const char** argv){
#ifdef _OPENMP
	omp_set_num_threads(64);
#endif
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
