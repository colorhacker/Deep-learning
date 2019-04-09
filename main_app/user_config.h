#ifndef _user_config_H
#define _user_config_H

#include <string.h>
//#include <stdbool.h>
//#include <cv.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <windows.h>   
#include <direct.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <basetsd.h>
#include <float.h>
#include <conio.h>

#include <omp.h>

#include "user_types.h"
//是否开启OPENMP底层API
#define		_USER_API_OPENMP				true
//配置输出文件夹
#define		user_nn_debug_file				"./debug.txt"
//初始化网络值的方式
//lecun_uniform、glorot_normal、glorot_uniform、he_normal、he_uniform
#define		glorot_uniform					true
//CNN配置开始
#define		user_nn_cnn_softmax				activation_sigmoid
#define		user_nn_model_cnn_file_name		"./model/cnn_model"
#define		user_nn_model_cnn_layer_addr	 0x0		//保存层的基地址
#define		user_nn_model_cnn_content_addr	 0x800		//保存层对象的基地址
#define		user_nn_model_cnn_data_addr		 0x1000		//保存数据的基地址
//CNN配置结束
//RNN配置开始
#define		user_nn_rnn_softmax				activation_sigmoid
#define		user_nn_model_rnn_file_name		"./model/rnn_model"
#define		user_nn_model_rnn_layer_addr	 0x0		//保存层的基地址
#define		user_nn_model_rnn_content_addr	 0x800		//保存层对象的基地址
#define		user_nn_model_rnn_data_addr		 0x1000		//保存数据的基地址
//RNN配置结束
//NN配置开始
#define		user_nn_nn_softmax				activation_tanh
#define		user_nn_model_nn_file_name		"./model/nn_model"
#define		user_nn_model_nn_layer_addr		 0x0		//保存层的基地址
#define		user_nn_model_nn_content_addr	 0x800		//保存层对象的基地址
#define		user_nn_model_nn_data_addr		 0x1000		//保存数据的基地址
//NN配置结束
#endif