
#include "rgb_hsl.h"

static float _hue_to_rgb(float p, float q, float t) {
	if (t < 0.0f) t += 1.0f;
	if (t > 1.0f) t -= 1.0f;
	if (t < 1.0f / 6.0f) return (p + (q - p) * 6 * t);
	if (t < 1.0f / 2.0f) return q;
	if (t < 2.0f / 3.0f) return (p + (q - p) * (2 / 3 - t) * 6);
	return p;
}
static float _max_rgb(float *rgb) {
	float max = rgb[0];
	max = max < rgb[1] ? rgb[1] : max;
	max = max < rgb[2] ? rgb[2] : max;
	return max;
}
static float _min_rgb(float *rgb) {
	float min = rgb[0];
	min = min > rgb[1] ? rgb[1] : min;
	min = min > rgb[2] ? rgb[2] : min;
	return min;
}

/**
* Converts an RGB color value to HSL. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
* Assumes r, g, and b are contained in the set [0, 255] and
* returns h, s, and l in the set [0, 1].
*
* @param   {number}  r       The red color value
* @param   {number}  g       The green color value
* @param   {number}  b       The blue color value
* @return  {Array}           The HSL representation
*/

void RGB_to_HSL(user_nn_rgb *rgb, user_nn_hsl *hsl) {
	float rgb_f[] = { (float)rgb->R / 255, (float)rgb->G / 255, (float)rgb->B / 255};
	float max = _max_rgb(rgb_f), min = _min_rgb(rgb_f);
	hsl->L = (max + min) / 2;
	if (max == min) {
		hsl->H = hsl->S = 0; // achromatic
	}
	else {
		float d = max - min;
		hsl->S = hsl->L > 0.5f ? d / (2.0f - max - min) : d / (max + min);
		if (max == rgb_f[0]) {
			hsl->H = (rgb_f[1] - rgb_f[2]) / d + (rgb_f[1]  < rgb_f[2] ? 6 : 0);
		}else if (max == rgb_f[1]) {
			hsl->H = (rgb_f[2] - rgb_f[0]) / d + 2;
		}else if (max == rgb_f[2]) {
			hsl->H = (rgb_f[0] - rgb_f[1]) / d + 4;
		}
		hsl->H /= 6;
	}
}
/**
* Converts an HSL color value to RGB. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
* Assumes h, s, and l are contained in the set [0, 1] and
* returns r, g, and b in the set [0, 255].
*
* @param   {number}  h       The hue 色相分量
* @param   {number}  s       The saturation 颜色饱和度
* @param   {number}  l       The lightness 颜色亮度
* @return  {Array}           The RGB representation
*/

void HSL_to_RGB(user_nn_hsl *hsl, user_nn_rgb *rgb) {
	user_nn_hsl rgb_f = { 0.0f, 0.0f, 0.0f };
	if (hsl->S == 0) {
		rgb_f.H = rgb_f.S = rgb_f.L = hsl->L; // achromatic
	}
	else {
		float q = hsl->L < 0.5f ? hsl->L * (1.0f + hsl->S) : hsl->L + hsl->S - hsl->L * hsl->S;
		float p = 2.0f * hsl->L - q;
		rgb_f.H = _hue_to_rgb(p, q, hsl->H + 1.0f / 3.0f);
		rgb_f.S = _hue_to_rgb(p, q, hsl->H);
		rgb_f.L = _hue_to_rgb(p, q, hsl->H - 1.0f / 3.0f);
	}
	rgb->R = (unsigned char)(rgb_f.H * 255);
	rgb->G = (unsigned char)(rgb_f.S * 255);
	rgb->B = (unsigned char)(rgb_f.L * 255);
}
/**
* Converts an RGB color value to HSV. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSV_color_space.
* Assumes r, g, and b are contained in the set [0, 255] and
* returns h, s, and v in the set [0, 1].
*
* @param   Number  r       The red color value
* @param   Number  g       The green color value
* @param   Number  b       The blue color value
* @return  Array           The HSV representation
*/
void RGB_to_HSV(user_nn_rgb *rgb, user_nn_hsv *hsv) {
	float rgb_f[] = { (float)rgb->R / 255, (float)rgb->G / 255, (float)rgb->B / 255 };
	float max = _max_rgb(rgb_f), min = _min_rgb(rgb_f);
	float d = max - min;
	hsv->V = max;
	hsv->S = max == 0.0f ? 0.0f : d / max;

	if (max == min) {
		hsv->H = 0; // achromatic
	}
	else {
		if (max == rgb_f[0]) {
			hsv->H = (rgb_f[1] - rgb_f[2]) / d + (rgb_f[1] < rgb_f[2] ? 6.0f : 0.0f);
		}
		else if (max == rgb_f[1]) {
			hsv->H = (rgb_f[2] - rgb_f[0]) / d + 2.0f;
		}
		else if (max == rgb_f[2]) {
			hsv->H = (rgb_f[0] - rgb_f[1]) / d + 4.0f;
		}
		hsv->H /= 6;
	}
}
/**
* Converts an HSV color value to RGB. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSV_color_space.
* Assumes h, s, and v are contained in the set [0, 1] and
* returns r, g, and b in the set [0, 255].
*
* @param   Number  h       The hue 色调
* @param   Number  s       The saturation 饱和度 
* @param   Number  v       The value 明度
* @return  Array           The RGB representation
*/
void HSV_to_RGB(user_nn_hsv *hsv, user_nn_rgb *rgb) {
	user_nn_hsv rgb_f = {0.0f,0.0f,0.0f};

	int i = (int)floor(hsv->H * 6);
	float f = hsv->H * 6 - i;
	float p = hsv->V * (1 - hsv->S);
	float q = hsv->V * (1 - f * hsv->S);
	float t = hsv->V * (1 - (1 - f) * hsv->S);

	switch (i % 6) {
		case 0: rgb_f.H = hsv->V, rgb_f.S = t, rgb_f.V = p; break;
		case 1: rgb_f.H = q, rgb_f.S = hsv->V, rgb_f.V = p; break;
		case 2: rgb_f.H = p, rgb_f.S = hsv->V, rgb_f.V = t; break;
		case 3: rgb_f.H = p, rgb_f.S = q, rgb_f.V = hsv->V; break;
		case 4: rgb_f.H = t, rgb_f.S = p, rgb_f.V = hsv->V; break;
		case 5: rgb_f.H = hsv->V, rgb_f.S = p, rgb_f.V = q; break;
		default:break;
	}

	rgb->R = (unsigned char)(rgb_f.H * 255);
	rgb->G = (unsigned char)(rgb_f.S * 255);
	rgb->B = (unsigned char)(rgb_f.V * 255);
}