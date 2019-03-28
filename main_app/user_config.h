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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <basetsd.h>
#include <float.h>
#include <conio.h>

#include <omp.h>

//#include <cuda_runtime.h>
//#include <cublas_v2.h>
//#include <device_launch_parameters.h>

//#include "CvxText.h"
//#include "user_nn_matrix.h"
//#include "user_cnn_create.h"
#include "user_types.h"

//��ʼ������ֵ�ķ�ʽ
//lecun_uniform��glorot_normal��glorot_uniform��he_normal��he_uniform
#define		user_nn_init_type				glorot_normal

//CNN���ÿ�ʼ
#define		user_nn_cnn_softmax				activation_tanh

#define		user_nn_cnn_training_folder		"digital"
#define		user_nn_cnn_training_type		".jpg"
#define		user_nn_model_cnn_file_name		"./model/cnn_model.bin"
#define		user_nn_model_cnn_layer_addr	 0x0		//�����Ļ���ַ
#define		user_nn_model_cnn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_cnn_data_addr		 0x1000		//�������ݵĻ���ַ
//CNN���ý���
//RNN���ÿ�ʼ
#define		user_nn_rnn_softmax				activation_sigmoid

#define		user_nn_model_rnn_file_name		"./model/rnn_model.bin"
#define		user_nn_model_rnn_layer_addr	 0x0		//�����Ļ���ַ
#define		user_nn_model_rnn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_rnn_data_addr		 0x1000		//�������ݵĻ���ַ
//RNN���ý���
//NN���ÿ�ʼ
#define		user_nn_nn_softmax				activation_prelu

#define		user_nn_model_nn_file_name		"./model/nn_model.bin"
#define		user_nn_model_nn_layer_addr		 0x0		//�����Ļ���ַ
#define		user_nn_model_nn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_nn_data_addr		 0x1000		//�������ݵĻ���ַ
//NN���ý���
#endif