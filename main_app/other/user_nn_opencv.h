#ifndef _user_nn_opencv__H
#define _user_nn_opencv__H

#include<stdio.h>
#include<stdlib.h> 
#include<string.h>
#include <math.h>
//#include <opencv.hpp>

#include "rgb_hsl.h"
#include "../matrix/user_nn_matrix.h"

user_nn_matrix *user_opencv_read_image(const char *path);
user_nn_matrix *user_matrix_rgb_hls(unsigned char *rgb, int width, int height);

#endif