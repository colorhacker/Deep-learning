#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"
#include "matrix/user_nn_matrix_cuda.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"

int main(int argc, const char** argv){
	//user_nn_app_topic(argc, argv);
	
	/*float content[] = { 
		590,583,932,243,698,
		97,331,488,359,455,
		445,820,411,451,506,
		677,265,332,865,881,
		341,761,573,378,368 };//矩阵数据
	float height_weight[] = { 169,455,74,20,282 };//输出数据的权重
	float width_weight[] = {  210,116,351,20,303 };//输入数据的权重
	//float height_weight[] = { 210,116,351,20,303 };//输出数据的权重
	//float width_weight[] = { 169,455,74,20,282 };//输入数据的权重
	*/
	float content[] = {
		0.59,0.583,0.932,0.243,0.698,0.295,0.747,0.781,0.644,0.249,
		0.097,0.331,0.488,0.359,0.455,0.336,0.804,0.699,0.721,0.222,
		0.445,0.82,0.411,0.451,0.506,0.49,0.548,0.459,0.635,0.4,
		0.677,0.265,0.332,0.865,0.881,0.023,0.425,0.053,0.233,0.302,
		0.341,0.761,0.573,0.378,0.368,0.29,0.022,0.316,0.527,0.434,
		0.138,0.125,0.232,0.116,0.126,0.763,0.399,0.715,0.992,0.285,
		0.135,0.597,0.071,0.899,0.693,0.891,0.813,0.218,0.823,0.05,
		0.249,0.503,0.198,0.204,0.033,0.15,0.056,0.863,0.472,0.227,
		0.718,0.983,0.448,0.334,0.138,0.935,0.643,0.563,0.15,0.4,
		0.959,0.842,0.551,0.738,0.561,0.447,0.693,0.842,0.431,0.965 };//矩阵数据
	//float height_weight[] = { 0.069,0.055,0.074,0.02,0.084,0.077,0.036,0.016,0.107,0.462 };//输出数据的权重
	//float width_weight[] = { 0.11,0.016,0.051,0.02,0.05,0.059,0.05,0.024,0.089,0.531 };//输入数据的权重
	float height_weight[] = { 0.11,0.016,0.051,0.02,0.05,0.059,0.05,0.024,0.089,0.531 };//输出数据的权重
	float width_weight[] = { 0.069,0.055,0.074,0.02,0.084,0.077,0.036,0.016,0.107,0.462 };//输入数据的权重
	float distance = 0.0f;
	int time_count = 0;

	clock_t start_time = clock();

	for (time_count = 0; time_count < 1000; time_count++) {
		distance = user_emd_earth_movers_distance(content, height_weight, sizeof(height_weight) / sizeof(float), width_weight, sizeof(width_weight) / sizeof(float));
	}
	clock_t end_time = clock() - start_time;//获取结束时间
	printf("\nresult:%f,total time:%dms,average time:%dms", distance, end_time, end_time / time_count);
	getchar();
	return 0;
}

/*
	float test_content[] = {
		1,1,0,0,
		0,1,-1,1,
		0,0,1,1,
		1,0,1,0
	};//矩阵数据
	user_nn_matrix *test_matrix = user_nn_matrix_create(4,4);
	user_nn_matrix_memcpy(test_matrix, test_content);

	user_nn_matrix *path_matrix = get_loop_path_list(test_matrix, user_nn_matrix_return_min_index(test_matrix));
	if (path_matrix != NULL)
		user_nn_matrix_printf(NULL, path_matrix);//
	else
		printf("\nmatrix error!\n");
*/