#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"
#include "other\user_nn_opencv.h"

int main(int argc, const char** argv){
	user_nn_list_matrix *train_lables = user_nn_matrices_create(50, 1, 1, 784);
	user_nn_list_matrix *train_images = user_nn_matrices_create(50, 1, 1, 784);
	//user_nn_matrices_init_vaule(rand_matrix_list,3,3);
	user_nn_matrix *images_matrix = train_images->matrix;
	user_nn_matrix *lables_matrix = train_lables->matrix;
	user_nn_matrix *kernel_matrix = user_nn_matrix_create(4, 4);//卷积矩阵
	user_nn_matrix *same_matrix = NULL;//卷积矩阵
	user_nn_matrix_memset(kernel_matrix, 1.0f);
	for (int count = 0; count < train_images->height*train_images->width; count++) {
		images_matrix->width = 28;
		images_matrix->height = 28;
		user_nn_matrix_paint_rectangle(images_matrix,
			(int)(user_nn_init_normal() * (images_matrix->width - 2)),
			(int)(user_nn_init_normal() * (images_matrix->height - 2)),
			(int)(user_nn_init_normal() * (images_matrix->width - 2)),
			(int)(user_nn_init_normal() * (images_matrix->height - 2)), 1.0f);//画矩形
		images_matrix->width = 1;
		images_matrix->height = 784;

		same_matrix = user_nn_matrix_conv2(images_matrix, kernel_matrix, u_nn_conv2_type_same);
		same_matrix->width = 1;
		same_matrix->height = 784;
		user_nn_matrix_cpy_matrix(lables_matrix, same_matrix);
		user_nn_matrix_delete(same_matrix);

		user_opencv_show_matrix("p", lables_matrix,10, 10, 1);
		user_opencv_show_matrix("s", images_matrix, 100, 10, 1);
		_getch();
		images_matrix = images_matrix->next;
		lables_matrix = lables_matrix->next;
	}
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
