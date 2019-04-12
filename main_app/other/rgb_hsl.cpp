
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

void RGB_to_HSL(unsigned char *rgb, float *hsl) {
	float rgb_f[] = { (float)rgb[0] / 255, (float)rgb[1] / 255, (float)rgb[2] / 255};
	float max = _max_rgb(rgb_f), min = _min_rgb(rgb_f);
	hsl[2] = (max + min) / 2;
	if (max == min) {
		hsl[0] = hsl[1] = 0; // achromatic
	}
	else {
		float d = max - min;
		hsl[1] = hsl[2] > 0.5f ? d / (2.0f - max - min) : d / (max + min);
		if (max == rgb_f[0]) {
			hsl[0] = (rgb_f[1] - rgb_f[2]) / d + (rgb_f[1]  < rgb_f[2] ? 6 : 0);
		}else if (max == rgb_f[1]) {
			hsl[0] = (rgb_f[2] - rgb_f[0]) / d + 2;
		}else if (max == rgb_f[2]) {
			hsl[0] = (rgb_f[0] - rgb_f[1]) / d + 4;
		}
		hsl[0] /= 6;
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

void HSL_to_RGB(float *hsl, unsigned char *rgb) {
	float rgb_f[] = { 0.0f, 0.0f, 0.0f };
	if (hsl[1] == 0) {
		rgb_f[0] = rgb_f[1] = rgb_f[2] = hsl[2]; // achromatic
	}
	else {
		float q = hsl[2] < 0.5f ? hsl[2] * (1.0f + hsl[1]) : hsl[2] + hsl[1] - hsl[2] * hsl[1];
		float p = 2.0f * hsl[2] - q;
		rgb_f[0] = _hue_to_rgb(p, q, hsl[0] + 1.0f / 3.0f);
		rgb_f[1] = _hue_to_rgb(p, q, hsl[0]);
		rgb_f[2] = _hue_to_rgb(p, q, hsl[0] - 1.0f / 3.0f);
	}
	rgb[0] = (unsigned char)(rgb_f[0] * 255);
	rgb[1] = (unsigned char)(rgb_f[1] * 255);
	rgb[2] = (unsigned char)(rgb_f[2] * 255);
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
void RGB_to_HSV(unsigned char *rgb, float *hsv) {
	float rgb_f[] = { (float)rgb[0] / 255, (float)rgb[1] / 255, (float)rgb[2] / 255 };
	float max = _max_rgb(rgb_f), min = _min_rgb(rgb_f);
	float d = max - min;
	hsv[2] = max;
	hsv[1] = max == 0.0f ? 0.0f : d / max;

	if (max == min) {
		hsv[0] = 0; // achromatic
	}
	else {
		if (max == rgb_f[0]) {
			hsv[0] = (rgb_f[1] - rgb_f[2]) / d + (rgb_f[1] < rgb_f[2] ? 6.0f : 0.0f);
		}
		else if (max == rgb_f[1]) {
			hsv[0] = (rgb_f[2] - rgb_f[0]) / d + 2.0f;
		}
		else if (max == rgb_f[2]) {
			hsv[0] = (rgb_f[0] - rgb_f[1]) / d + 4.0f;
		}
		hsv[0] /= 6;
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
void HSV_to_RGB(float *hsv, unsigned char *rgb) {
	float rgb_f[] = { 0.0f,0.0f,0.0f };

	int i = (int)floor(hsv[0] * 6);
	float f = hsv[0] * 6 - i;
	float p = hsv[2] * (1 - hsv[1]);
	float q = hsv[2] * (1 - f * hsv[1]);
	float t = hsv[2] * (1 - (1 - f) * hsv[1]);

	switch (i % 6) {
	case 0: rgb_f[0] = hsv[2], rgb_f[1] = t, rgb_f[2] = p; break;
	case 1: rgb_f[0] = q, rgb_f[1] = hsv[2], rgb_f[2] = p; break;
	case 2: rgb_f[0] = p, rgb_f[1] = hsv[2], rgb_f[2] = t; break;
	case 3: rgb_f[0] = p, rgb_f[1] = q, rgb_f[2] = hsv[2]; break;
	case 4: rgb_f[0] = t, rgb_f[1] = p, rgb_f[2] = hsv[2]; break;
	case 5: rgb_f[0] = hsv[2], rgb_f[1] = p, rgb_f[2] = q; break;
	default:break;
	}
	rgb[0] = (unsigned char)(rgb_f[0] * 255);
	rgb[1] = (unsigned char)(rgb_f[1] * 255);
	rgb[2] = (unsigned char)(rgb_f[2] * 255);
}


float user_nn_get_rgb_hue(unsigned char *rgb) {
	float hsl[3];
	RGB_to_HSL(rgb, hsl);
	return hsl[0];
}