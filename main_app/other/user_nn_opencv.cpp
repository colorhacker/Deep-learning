#include "user_nn_opencv.h"


void user_opencv_show_matrix(char *windows, user_nn_matrix *src_matrix,int x,int y,int delay) {
	//int width = (int)sqrt(src_matrix->height*src_matrix->width);
	//int height = (int)sqrt(src_matrix->height*src_matrix->width);
	int width = src_matrix->height;
	int height = src_matrix->width;
	cv::Mat img(width, height, CV_32FC1, src_matrix->data);
	cv::namedWindow(windows, cv::WINDOW_AUTOSIZE);
	//cv::namedWindow(windows, cv::WINDOW_NORMAL);
	//cv::resizeWindow(window_name, width, height);
	//cv::updateWindow(win);//opengl
	//cv::startWindowThread();
	cv::moveWindow(windows, x, y);
	cv::imshow(windows, img);
	cv::waitKey(delay);
}
void user_opencv_show_rgb(char *windows,user_nn_matrix *src_matrix) {
	user_nn_matrix_divi_constant(src_matrix, 255.0f);
	cv::Mat d_bgr, d_rgb( src_matrix->height, src_matrix->width / 3, CV_32FC3, src_matrix->data);
	cv::namedWindow(windows, cv::WINDOW_NORMAL);
	cv::cvtColor(d_rgb, d_bgr, cv::COLOR_BGR2RGB);
	cv::imshow(windows, d_bgr);
	cv::waitKey(0);
}
user_nn_matrix *user_opencv_read_image(const char *path) {
	cv::Mat rgb,bgr = cv::imread(path, cv::IMREAD_COLOR);//format BGR
	cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);//BGR TO RGB
	user_nn_matrix *result = user_nn_matrix_create(rgb.cols * rgb.channels(), rgb.rows);
	user_nn_matrix_memcpy_uchar(result, rgb.data);
	return result;
}

//把rgb矩阵转化为hls
//src_matrix 输入矩阵 数据类型 float 内部转化为unsigned char
//return 新的矩阵
user_nn_matrix *user_matrix_rgb_hsl(user_nn_matrix *src_matrix) {
	unsigned char rgb_char[3];
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);
	for (int index = 0; index < src_matrix->width*src_matrix->height;index += 3) {
		rgb_char[0] = (unsigned char)src_matrix->data[index];
		rgb_char[1] = (unsigned char)src_matrix->data[index+1];
		rgb_char[2] = (unsigned char)src_matrix->data[index+2];
		rgb_to_hsl(rgb_char, &result->data[index]);
	}
	return result;
}
//把hls转化为rgb
//src_matrix 输入矩阵 数据类型 unsigned char 内部转化为float
//return 新的矩阵
user_nn_matrix *user_matrix_hsl_rgb(user_nn_matrix *src_matrix) {
	unsigned char rgb_char[3];
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);
	for (int index = 0; index < src_matrix->width*src_matrix->height; index += 3) {
		hsl_to_rgb(&src_matrix->data[index], rgb_char);
		result->data[index]		= (float)rgb_char[0];
		result->data[index + 1] = (float)rgb_char[1];
		result->data[index + 2] = (float)rgb_char[2];
	}
	return result;
}

//把hls转化为rgb
//src_matrix 输入矩阵 数据类型 unsigned char 内部转化为float
//return 新的矩阵
user_nn_matrix *user_matrix_hsl_rgb_actor(user_nn_matrix *src_matrix,float s,float l) {
	unsigned char rgb_char[3];
	float hsl_float[3];
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);
	for (int index = 0; index < src_matrix->width*src_matrix->height; index += 3) {
		hsl_float[0] = src_matrix->data[index];
		hsl_float[1] = s;
		hsl_float[2] = l;
		hsl_to_rgb(hsl_float, rgb_char);
		result->data[index] = (float)rgb_char[0];
		result->data[index + 1] = (float)rgb_char[1];
		result->data[index + 2] = (float)rgb_char[2];
	}
	return result;
}

//把rgb矩阵转化为hls
//src_matrix 输入矩阵 数据类型 float 内部转化为unsigned char
//return 新的矩阵
user_nn_matrix *user_matrix_rgb_hsv(user_nn_matrix *src_matrix) {
	unsigned char rgb_char[3];
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);
	for (int index = 0; index < src_matrix->width*src_matrix->height; index += 3) {
		rgb_char[0] = (unsigned char)src_matrix->data[index];
		rgb_char[1] = (unsigned char)src_matrix->data[index + 1];
		rgb_char[2] = (unsigned char)src_matrix->data[index + 2];
		rgb_to_hsv(rgb_char, &result->data[index]);
	}
	return result;
}
//把hls转化为rgb
//src_matrix 输入矩阵 数据类型 unsigned char 内部转化为float
//return 新的矩阵
user_nn_matrix *user_matrix_hsv_rgb(user_nn_matrix *src_matrix) {
	unsigned char rgb_char[3];
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);
	for (int index = 0; index < src_matrix->width*src_matrix->height; index += 3) {
		hsv_to_rgb(&src_matrix->data[index], rgb_char);
		result->data[index] = (float)rgb_char[0];
		result->data[index + 1] = (float)rgb_char[1];
		result->data[index + 2] = (float)rgb_char[2];
	}
	return result;
}

//把hls转化为rgb
//src_matrix 输入矩阵 数据类型 unsigned char 内部转化为float
//return 新的矩阵
user_nn_matrix *user_matrix_hsv_rgb_actor(user_nn_matrix *src_matrix, float s, float l) {
	unsigned char rgb_char[3];
	float hsl_float[3];
	user_nn_matrix *result = user_nn_matrix_create(src_matrix->width, src_matrix->height);
	for (int index = 0; index < src_matrix->width*src_matrix->height; index += 3) {
		hsl_float[0] = src_matrix->data[index];
		hsl_float[1] = s;
		hsl_float[2] = l;
		hsv_to_rgb(hsl_float, rgb_char);
		result->data[index] = (float)rgb_char[0];
		result->data[index + 1] = (float)rgb_char[1];
		result->data[index + 2] = (float)rgb_char[2];
	}
	return result;
}