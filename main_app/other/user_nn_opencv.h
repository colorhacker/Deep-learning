#ifndef _user_nn_opencv__H
#define _user_nn_opencv__H

#include<stdio.h>
#include<stdlib.h> 
#include<string.h>
#include <math.h>
//#include <opencv.hpp>

#include "rgb_hsl.h"
#include "../matrix/user_nn_matrix.h"

void user_opencv_show_matrix(char *windows, user_nn_matrix *src_matrix, int x, int y);
void user_opencv_show_rgb(char *windows, user_nn_matrix *src_matrix);
user_nn_matrix *user_opencv_read_image(const char *path);

user_nn_matrix *user_matrix_rgb_hsl(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsl_rgb(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsl_rgb_actor(user_nn_matrix *src_matrix, float s, float l);

user_nn_matrix *user_matrix_rgb_hsv(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsv_rgb(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsv_rgb_actor(user_nn_matrix *src_matrix, float s, float l);


#endif

/*
user_nn_matrix *image = user_opencv_read_image("E:/GitHub/_output/Release64/exe/apple.jpg");
user_nn_matrix *hls = user_matrix_rgb_hsv(image);
//user_nn_matrix *rgb = user_matrix_hsv_rgb(hls);
user_nn_matrix *rgb = user_matrix_hsv_rgb_actor(hls,1.0f,0.5f);
user_opencv_show_rgb("n", rgb);
*/
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