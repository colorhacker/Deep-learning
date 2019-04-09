#ifndef _rgb_hsl_H
#define _rgb_hsl_H

#include<stdio.h>
#include<stdlib.h> 
#include<string.h>

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

void HSL_to_RGB(user_nn_hsl *hsl, user_nn_rgb *rgb);
void RGB_to_HSL(user_nn_rgb *rgb, user_nn_hsl *hsl);

#endif