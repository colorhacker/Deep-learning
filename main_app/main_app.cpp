#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

#include "snn\user_snn_app.h"
#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"
#include "other\user_nn_opencv.h"

//提取图像矩阵 按照指定大小与step进行数据提取
//src_matrix 源矩阵
//f_width 需要生成的矩阵宽度
//f_height 需要生成矩阵的高度
//step 矩阵每次移动大小
//返回特征矩阵
user_nn_list_matrix *user_nn_matrix_generate_feature(user_nn_matrix *src_matrix,int f_width,int f_height,int step) {
	user_nn_list_matrix *featrue_list = user_nn_matrices_create_head(1, 1);
	for (int height = 0; height < src_matrix->height - f_height + 1; height+= step) {
		for (int width = 0; width < src_matrix->width - f_width + 1; width+= step) {
			user_nn_matrices_add_matrix(featrue_list, user_nn_matrix_ext_matrix(src_matrix, width, height, f_width, f_height));
			printf("\n x:%d,y:%d,w:%d,h:%d", width, height, f_width, f_height);
		}
	}
	return featrue_list;
}

int main(int argc, const char** argv){
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	//user_opencv_show_matrix("test_image:0", user_nn_matrices_ext_matrix_index(train_images, 0), 100, 100, 1);
	user_nn_matrix *data_matrix = user_nn_matrices_ext_matrix_index(train_images, 1);

	user_nn_list_matrix *featrue_list = user_nn_matrix_generate_feature(data_matrix,8,8,2);

	for (int index = 0; index < featrue_list->height*featrue_list->width;index++) {
		user_opencv_show_matrix("f:0", user_nn_matrices_ext_matrix_index(featrue_list, index), 100, 100, 1);
		printf("\n i:%d,t:%d", index, featrue_list->height*featrue_list->width);
		_getch();
	}
	
	return 0;
	user_snn_app_test(argc,argv);
	return 0;
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
