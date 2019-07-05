#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"

//#include "snn\user_snn_app.h"
#include "nn\user_nn_app.h"
//#include "rnn\user_rnn_app.h"
//#include "cnn\user_cnn_app.h"
//#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"
#include "other\user_nn_opencv.h"


int main(int argc, const char** argv){
	
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 16, //输入层 特征（宽度、高度）
		'h', 128, //隐含层 特征 （高度）
		'h', 256, //隐含层 特征 （高度）
		'h', 128, //隐含层 特征 （高度）
		'o', 10 //输出层 特征 （高度）
	};
	bool sw_display = false;
	float loss_function = 1.0f, loss_target = 0.001f;
	int save_model_count = 0, exit_train_count = 10000000;
	clock_t start_time, end_time;
	printf("\n\n");
	printf("\n-----训练可视化-----\n");
	printf("\n1.开启");
	printf("\n2.关闭（或者其他按键）");
	printf("\n请输入数字：");
	sw_display = (_getch() == '1') ? true : false;
	user_nn_matrix *train_lables_m = numpy_load("./train_mnist_feature.npy");
	user_nn_list_matrix *train_images = user_nn_matrices_create(1,60000,1,16);
	user_nn_matrix_to_matrices(train_images, train_lables_m);
	user_nn_list_matrix *train_lables = user_nn_model_file_read_matrices("./mnist/files/train-labels.idx1-ubyte.bx", 0);
	//user_nn_list_matrix *train_images = user_nn_model_file_read_matrices("./mnist/files/train-images.idx3-ubyte.bx", 0);
	user_nn_layers *nn_layers = user_nn_model_load_model(0);//载入模型
	if (nn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		nn_layers = user_nn_model_create(user_layers);//创建模型
	}
	user_nn_model_info_layer(nn_layers);
	start_time = clock();
	while (1) {
		for (int index = 0; index < train_images->height * train_images->width; index++) {
			user_nn_model_load_input_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_images, index));//加载输入数据
			user_nn_model_load_target_feature(nn_layers, user_nn_matrices_ext_matrix_index(train_lables, index));//加载目标数据	
			user_nn_model_ffp(nn_layers);//正向计算一次
			user_nn_model_bp(nn_layers, 0.01f);//反向计算一次
			loss_function = user_nn_model_return_loss(nn_layers);
			if (sw_display) {
				user_nn_model_display_feature(nn_layers);
			}
			if (loss_function <= loss_target || exit_train_count-- <= 0) {
				user_nn_model_save_model(nn_layers, 0);//保存模型
				break;
			}
			printf("\ntarget:%f loss:%f", loss_target, loss_function);
			if (save_model_count++ > 1000) {
				save_model_count = 0;
				end_time = clock();
				printf("\ntarget:%f loss:%f,time:%ds", loss_target, loss_function, (end_time - start_time) / 1000);
				user_nn_model_save_model(nn_layers, 0);//保存一次模型
				start_time = clock();
			}
		}
		if (loss_function <= loss_target || exit_train_count-- <= 0) {
			break;
		}
	}
/*
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
