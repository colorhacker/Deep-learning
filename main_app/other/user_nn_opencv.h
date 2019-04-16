#ifndef _user_nn_opencv__H
#define _user_nn_opencv__H

#include<stdio.h>
#include<stdlib.h> 
#include<string.h>
#include <math.h>
//#include <opencv.hpp>

#include "rgb_hsl.h"
#include "../matrix/user_nn_matrix.h"

void user_opencv_show_rgb(char *windows, user_nn_matrix *src_matrix);
user_nn_matrix *user_opencv_read_image(const char *path);

user_nn_matrix *user_matrix_rgb_hsl(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsl_rgb(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsl_rgb_actor(user_nn_matrix *src_matrix, float s, float l);

user_nn_matrix *user_matrix_rgb_hsv(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsv_rgb(user_nn_matrix *src_matrix);
user_nn_matrix *user_matrix_hsv_rgb_actor(user_nn_matrix *src_matrix, float s, float l);
#endif