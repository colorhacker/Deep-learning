#ifndef _user_config_H
#define _user_config_H

#include <string.h>
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

#include <opencv.hpp>

#include "user_types.h"

//�Ƿ���OPENMP�ײ�API
#define		_USER_API_OPENMP				true
#define		_USER_API_OPENMP_CONV			false
//��������ļ���
#define		user_nn_debug_file				"./debug.txt"
//��ʼ������ֵ�ķ�ʽ
//lecun_uniform��glorot_normal��glorot_uniform��he_normal��he_uniform
#define		glorot_uniform					true
//CNN���ÿ�ʼ
#define		user_nn_cnn_softmax				activation_tanh
#define		user_nn_model_cnn_file_name		"./model/cnn_model"
#define		user_nn_model_cnn_layer_addr	 0x0		//�����Ļ���ַ
#define		user_nn_model_cnn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_cnn_data_addr		 0x1000		//�������ݵĻ���ַ
//CNN���ý���
//RNN���ÿ�ʼ
#define		user_nn_rnn_softmax				activation_sigmoid
#define		user_nn_model_rnn_file_name		"./model/rnn_model"
#define		user_nn_model_rnn_layer_addr	 0x0		//�����Ļ���ַ
#define		user_nn_model_rnn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_rnn_data_addr		 0x1000		//�������ݵĻ���ַ
//RNN���ý���
//NN���ÿ�ʼ
#define		user_nn_use_bias				true
#define		user_nn_nn_softmax				activation_prelu
#define		user_nn_model_nn_file_name		"./model/nn_model"
#define		user_nn_model_nn_layer_addr		 0x0		//�����Ļ���ַ
#define		user_nn_model_nn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_nn_data_addr		 0x1000		//�������ݵĻ���ַ
//NN���ý���
//SNN���ÿ�ʼ
#define		snn_avg_vaule					1.0f	//���ݾ�ֵ
#define		snn_add_value					0.1f	//ǰ���� �仯ֵ
#define		snn_step_vaule					0.001f	//ÿһ���ƶ���ֵ

#define		snn_thred_none					1.0f	//����Ҫ�仯
#define		snn_thred_add					0.9f	//Ŀ��ֵ�������ֵ
#define		snn_thred_acc					1.1f	//Ŀ��ֵС�����ֵ

#define		user_nn_model_snn_file_name		"./model/snn_model"
#define		user_nn_model_snn_layer_addr	 0x0		//�����Ļ���ַ
#define		user_nn_model_snn_content_addr	 0x800		//��������Ļ���ַ
#define		user_nn_model_snn_data_addr		 0x1000		//�������ݵĻ���ַ
//SNN���ý���
#endif