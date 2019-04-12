#ifndef _rgb_hs__H
#define _rgb_hs__H

#include<stdio.h>
#include<stdlib.h> 
#include<string.h>
#include <math.h>

typedef enum _rgb_conv_type {
	rgb_conv_hsl = 0,//
	rgb_conv_hsv = 1,//
}rgb_conv_type;

typedef struct _nn_rgb {
	unsigned char R;
	unsigned char G;
	unsigned char B;
}user_nn_rgb;

typedef struct _nn_hsl {
	float H;
	float S;
	float L;
}user_nn_hsl;

typedef struct _nn_hsv {
	float H;
	float S;
	float V;
}user_nn_hsv;

void RGB_to_HSL(unsigned char *rgb, float *hsl);
void HSL_to_RGB(float *hsl, unsigned char *rgb);
void RGB_to_HSV(unsigned char *rgb, float *hsv);
void HSV_to_RGB(float *hsv, unsigned char *rgb);

float user_nn_get_rgb_hue(unsigned char *rgb, rgb_conv_type type);

#endif