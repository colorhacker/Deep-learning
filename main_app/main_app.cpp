#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

//#include "snn\user_snn_app.h"
//#include "nn\user_nn_app.h"
//#include "rnn\user_rnn_app.h"
//#include "cnn\user_cnn_app.h"
//#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"
#include "other\user_nn_opencv.h"

//提取图像矩阵 按照指定大小与step进行数据提取
//src_matrix 源矩阵
//f_width 需要生成的矩阵宽度
//f_height 需要生成矩阵的高度
//step 矩阵每次移动大小
//返回特征矩阵
user_nn_list_matrix *user_nn_matrix_generate_feature(user_nn_list_matrix *save_featrue,user_nn_matrix *src_matrix,int f_width,int f_height,int step) {
	user_nn_list_matrix *featrue_list = save_featrue == NULL ? user_nn_matrices_create_head(1, 1) : save_featrue;//不存在就创建
	for (int height = 0; height < src_matrix->height - f_height + 1; height+= step) {
		for (int width = 0; width < src_matrix->width - f_width + 1; width+= step) {
			user_nn_matrices_add_matrix(featrue_list, user_nn_matrix_ext_matrix(src_matrix, width, height, f_width, f_height));
			//printf("\n x:%d,y:%d,w:%d,h:%d", width, height, f_width, f_height);
		}
	}
	return featrue_list;
}
//通过k-means中心矩阵生成新的矩阵数据
//class_featrue k-means分类中心矩阵
//src_matrix 需要被重构的矩阵
//w_step 宽度每次步移动距离
//h_step 高度每次步移动距离
//返回 重构后的矩阵
user_nn_matrix *user_nn_matrix_kmeans_paste_refactor(user_nn_list_matrix *class_featrue, user_nn_matrix *src_matrix,int w_step,int h_step) {
	user_nn_matrix *matrix_temp = NULL;
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);//创建矩阵
	for (int height = 0; height < src_matrix->height - class_featrue->matrix->height + 1; height += h_step) {
		for (int width = 0; width < src_matrix->width - class_featrue->matrix->width + 1; width += w_step) {
			//printf("=>x:%d,y:%d,w:%d,h:%d\n", width, height, w_step, h_step);
			matrix_temp = user_nn_matrix_ext_matrix(src_matrix, width, height, class_featrue->matrix->width, class_featrue->matrix->height);//截取指定矩阵
			//user_opencv_show_matrix("f:0", matrix_temp, 100, 100, 1);
			//_getch();
			//printf("=>%d\n", user_nn_matrix_k_means_discern(class_featrue, matrix_temp));
			//user_nn_matrix_paste_matrix(result, user_nn_matrices_ext_matrix_index(class_featrue, user_nn_matrix_k_means_discern(class_featrue, matrix_temp)), width, height);//粘贴指定矩阵
			user_nn_matrix_add_paste_matrix(result, user_nn_matrices_ext_matrix_index(class_featrue, user_nn_matrix_k_means_discern(class_featrue, matrix_temp)), width, height);//粘贴指定矩阵
			user_nn_matrix_delete(matrix_temp);//删除缓冲矩阵
		}
	}
	return result;
}

int main(int argc, const char** argv){
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);

	user_nn_list_matrix *featrue_list = NULL;//创建头
	user_nn_list_matrix *kclass_list = NULL;
	user_nn_list_matrix *kclass_list_temp = NULL;

	int cut_width = 7;//剪切宽度
	int cut_heigth = 7;//剪切高度
	int cut_step = 1;//剪切精度
	int class_size = 16;//分类大小
	//for (int count = 0; count < train_images->width*train_images->height; count++) {
	for (int count = 0; count < 100; count++) {
		featrue_list = user_nn_matrix_generate_feature(NULL, user_nn_matrices_ext_matrix_index(train_images, count), cut_width, cut_heigth, cut_step);//分割图像
		if (kclass_list_temp == NULL) {
			kclass_list_temp = user_nn_matrix_k_means(NULL, featrue_list, class_size, 100);
		}else {
			user_nn_matrices_splice_matrices(kclass_list_temp, user_nn_matrix_k_means(NULL, featrue_list, class_size, 100));//
		}
		////如果超出分类进行二次分类
		if (kclass_list_temp->height*kclass_list_temp->width > 1000) {
			kclass_list = user_nn_matrix_k_means(NULL, kclass_list_temp, class_size, 100);
			break;
		}
		user_nn_matrices_delete(featrue_list);
		printf("::%d\n", count);
	}
	printf("k class size:%d\n", kclass_list_temp->height * kclass_list_temp->width);
	//printf("feature size:%d\n", featrue_list->height * featrue_list->width);
	//kclass_list = user_nn_matrix_k_means(kclass_list, featrue_list, class_size, 5000);//分类数据
	//printf("k class size:%d\n", kclass_list->height * kclass_list->width);

	for (int count = 100; count < 200; count++) {
		user_nn_matrix *matrix_temp = user_nn_matrix_kmeans_paste_refactor(kclass_list,user_nn_matrices_ext_matrix_index(train_images, count), cut_heigth, cut_step);
		user_opencv_show_matrix("f:1", user_nn_matrices_ext_matrix_index(train_images, count), 100, 100, 1);
		user_opencv_show_matrix("f:0", matrix_temp, 300, 100, 1);
		_getch();
		user_nn_matrix_delete(matrix_temp);
	}
	_getch();
	return 0;
/*
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
*/
}
