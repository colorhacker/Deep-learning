#include "user_nn_opencv.h"

/*
cv::Mat hls,img,iiIm = cv::imread("E:/GitHub/_output/Release64/exe/1.jpg", cv::IMREAD_COLOR);
cv::cvtColor(iiIm, hls, cv::COLOR_BGR2HLS);

int cols = hls.cols, rows = hls.rows;
printf("\n%d", hls.depth());
printf("\n%d", hls.channels());
printf("\n%d", hls.type());
for (int row = 0; row < hls.rows; row++){
for (int col = 0; col < hls.cols; col++){
printf("\n%d ", hls.at<cv::Vec3b>(row, col)[0] = 0);
printf(" %d", hls.at<cv::Vec3b>(row, col)[1]);
printf(" %d", hls.at<cv::Vec3b>(row, col)[2] = 255);
		}
	}

	cv::cvtColor(hls, img, cv::COLOR_HLS2BGR);
	//cv::namedWindow("1ao", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("1ao", cv::WINDOW_NORMAL);
	cv::imshow("1ao", img);
	cv::waitKey(0);
	return 0;
*/
user_nn_matrix *user_opencv_read_image(const char *path) {
	cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
	user_nn_matrix *result = user_nn_matrix_create(image.cols * image.channels(), image.rows);
	user_nn_matrix_memcpy_uchar(result, image.data);
	return result;
}

//把rgb矩阵转化为hls
user_nn_matrix *user_matrix_rgb_hls(unsigned char *rgb,int width,int height) {
	user_nn_matrix *result = user_nn_matrix_create(width, height);
	for (int index = 0; index < width*height;index += 3) {
		RGB_to_HSL(&rgb[index], &result->data[index]);
	}
	return result;
}
