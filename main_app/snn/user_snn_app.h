#ifndef _user_snn_app_H
#define _user_snn_app_H

#include "user_snn_layers.h"

#include "..\matrix\user_nn_initialization.h"
#include "..\matrix\user_nn_matrix_file.h"
#include "..\other\user_nn_opencv.h"

void user_snn_app_train(int argc, const char** argv);
void user_snn_app_test(int argc, const char** argv);

#endif