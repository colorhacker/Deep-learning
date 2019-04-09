
#include "rgb_hsl.h"

/**
* Converts an HSL color value to RGB. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
* Assumes h, s, and l are contained in the set [0, 1] and
* returns r, g, and b in the set [0, 255].
*
* @param   {number}  h       The hue
* @param   {number}  s       The saturation
* @param   {number}  l       The lightness
* @return  {Array}           The RGB representation
*/

static float hue_to_rgb(float p, float q, float t) {
	if (t < 0.0f) t += 1.0f;
	if (t > 1.0f) t -= 1.0f;
	if (t < 1.0f / 6.0f) return (p + (q - p) * 6 * t);
	if (t < 1.0f / 2.0f) return q;
	if (t < 2.0f / 3.0f) return (p + (q - p) * (2 / 3 - t) * 6);
	return p;
}

void HSL_to_RGB(user_nn_hsl *hsl, user_nn_rgb *rgb) {
	user_nn_hsl rgb_f = { 0.0f, 0.0f, 0.0f };
	if (hsl->S == 0) {
		rgb_f.H = rgb_f.S = rgb_f.L = hsl->L; // achromatic
	}
	else {
		float q = hsl->L < 0.5f ? hsl->L * (1.0f + hsl->S) : hsl->L + hsl->S - hsl->L * hsl->S;
		float p = 2.0f * hsl->L - q;
		rgb_f.H = hue_to_rgb(p, q, hsl->H + 1.0f / 3.0f);
		rgb_f.S = hue_to_rgb(p, q, hsl->H);
		rgb_f.L = hue_to_rgb(p, q, hsl->H - 1.0f / 3.0f);
	}
	rgb->R = (unsigned char)(rgb_f.H * 255);
	rgb->G = (unsigned char)(rgb_f.S * 255);
	rgb->B = (unsigned char)(rgb_f.L * 255);
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
static float _max_rgb(user_nn_hsl *rgb) {
	float max = rgb->H;
	max = max < rgb->S ? rgb->S : max;
	max = max < rgb->L ? rgb->L : max;
	return max;
}
static float _min_rgb(user_nn_hsl *rgb) {
	float min = rgb->H;
	min = min > rgb->S ? rgb->S : min;
	min = min > rgb->L ? rgb->L : min;
	return min;
}
void RGB_to_HSL(user_nn_rgb *rgb, user_nn_hsl *hsl) {
	user_nn_hsl rgb_f = { (float)rgb->R / 255, (float)rgb->G / 255, (float)rgb->B / 255};
	float max = _max_rgb(&rgb_f), min = _min_rgb(&rgb_f);
	hsl->L = (max + min) / 2;
	if (max == min) {
		hsl->H = hsl->S = 0; // achromatic
	}
	else {
		float d = max - min;
		hsl->S = hsl->L > 0.5f ? d / (2.0f - max - min) : d / (max + min);
		if (max == rgb_f.H) {
			hsl->H = (rgb_f.S - rgb_f.L) / d + (rgb_f.S < rgb_f.L ? 6 : 0);
		}else if (max == rgb_f.S) {
			hsl->H = (rgb_f.L - rgb_f.H) / d + 2;
		}else if (max == rgb_f.L) {
			hsl->H = (rgb_f.H - rgb_f.S) / d + 4;
		}
		hsl->H /= 6;
	}
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
void RGB_to_HSL(user_nn_rgb *rgb, user_nn_hsl *hsv) {
	user_nn_hsl rgb_f = { (float)rgb->R / 255, (float)rgb->G / 255, (float)rgb->B / 255 };
	float max = _max_rgb(&rgb_f), min = _min_rgb(&rgb_f);
	hsv->L = (max + min) / 2;
	float d = max - min;
	hsv->S = max == 0.0f ? 0.0f : d / max;

	if (max == min) {
		hsv->H = 0; // achromatic
	}
	else {
		if (max == rgb_f.H) {
			hsv->H = (rgb_f.S - rgb_f.L) / d + (rgb_f.S < rgb_f.L ? 6 : 0);
		}
		else if (max == rgb_f.S) {
			hsv->H = (rgb_f.L - rgb_f.H) / d + 2;
		}
		else if (max == rgb_f.L) {
			hsv->H = (rgb_f.H - rgb_f.S) / d + 4;
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
* @param   Number  h       The hue
* @param   Number  s       The saturation
* @param   Number  v       The value
* @return  Array           The RGB representation
*/
function hsvToRgb(h, s, v) {
	var r, g, b;

	var i = Math.floor(h * 6);
	var f = h * 6 - i;
	var p = v * (1 - s);
	var q = v * (1 - f * s);
	var t = v * (1 - (1 - f) * s);

	switch (i % 6) {
	case 0: r = v, g = t, b = p; break;
	case 1: r = q, g = v, b = p; break;
	case 2: r = p, g = v, b = t; break;
	case 3: r = p, g = q, b = v; break;
	case 4: r = t, g = p, b = v; break;
	case 5: r = v, g = p, b = q; break;
	}

	return[r * 255, g * 255, b * 255];
}