#include"base64.h"

// 全局常量定义
const char *base64char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const char padding_char = '=';

/*编码代码
* const unsigned char * sourcedata， 源数组
* char * base64 ，码字保存
*/
int base64_encode(const char *input, char *base64){
	int i = 0, j = 0;
	unsigned char trans_index = 0;    // 索引是8位，但是高两位都为0
	const int datalength = strlen((const char*)input);
	for (; i < datalength; i += 3) {
		// 每三个一组，进行编码
		// 要编码的数字的第一个
		trans_index = ((input[i] >> 2) & 0x3f);
		base64[j++] = base64char[(int)trans_index];
		// 第二个
		trans_index = ((input[i] << 4) & 0x30);
		if (i + 1 < datalength) {
			trans_index |= ((input[i + 1] >> 4) & 0x0f);
			base64[j++] = base64char[(int)trans_index];
		}else {
			base64[j++] = base64char[(int)trans_index];
			base64[j++] = padding_char;
			base64[j++] = padding_char;
			break;   // 超出总长度，可以直接break
		}
		// 第三个
		trans_index = ((input[i + 1] << 2) & 0x3c);
		if (i + 2 < datalength) { // 有的话需要编码2个
			trans_index |= ((input[i + 2] >> 6) & 0x03);
			base64[j++] = base64char[(int)trans_index];
			trans_index = input[i + 2] & 0x3f;
			base64[j++] = base64char[(int)trans_index];
		}else {
			base64[j++] = base64char[(int)trans_index];
			base64[j++] = padding_char;
			break;
		}
	}
	base64[j] = 0x00;
	return j;
}
/** 在字符串中查询特定字符位置索引
* const char *str ，字符串
* char c，要查找的字符
*/
inline int num_strchr(const char *str, char c){
	const char *pindex = strchr(str, c);
	if (NULL == pindex) {
		return -1;
	}
	return pindex - str;
}
/* 解码
* const char * base64 码字
* unsigned char * dedata， 解码恢复的数据
*/
int base64_decode(const char * base64, char * output){
	int i = 0, j = 0;
	int trans[4] = { 0,0,0,0 };
	for (; base64[i] != 0x00; i += 4) {
		// 每四个一组，译码成三个字符
		trans[0] = num_strchr(base64char, base64[i]);
		trans[1] = num_strchr(base64char, base64[i + 1]);
		// 1/3
		output[j++] = ((trans[0] << 2) & 0xfc) | ((trans[1] >> 4) & 0x03);
		if (base64[i + 2] == '=') {
			continue;
		}else {
			trans[2] = num_strchr(base64char, base64[i + 2]);
		}
		// 2/3
		output[j++] = ((trans[1] << 4) & 0xf0) | ((trans[2] >> 2) & 0x0f);
		if (base64[i + 3] == '=') {
			continue;
		}else {
			trans[3] = num_strchr(base64char, base64[i + 3]);
		}
		// 3/3
		output[j++] = ((trans[2] << 6) & 0xc0) | (trans[3] & 0x3f);
	}
	output[j] = 0x00;
	return j;
}
//字符串交换
char char_exchange(char input) {
	static char map_a_char[] = "acehjlnqsuwy12345";
	static char map_b_char[] = "bdfikmprtvxz98760";
	for (int i = 0; i < strlen(map_a_char); i++) {
		if (input == map_a_char[i]) {
			return map_b_char[i];
		}else if (input == map_b_char[i]) {
			return map_a_char[i];
		}
	}
	return input;
}
//加密
void encode_xor_map_base64(char *input, char *output) {
	for (int i = 0; i < strlen(input); i++) {
		input[i] = char_exchange(input[i]) ^ xor_code;
	}
	base64_encode(input, output);//生成base64码
}

//解密
void decode_base64_map_xor(char *input, char *output) {
	base64_decode(input, output);//解密base64
	for (int i = 0; i < strlen(output); i++) {
		output[i] = char_exchange(output[i] ^ xor_code);
	}
}