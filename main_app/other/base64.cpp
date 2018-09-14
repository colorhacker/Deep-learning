#include"base64.h"

// ȫ�ֳ�������
const char *base64char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const char padding_char = '=';

/*�������
* const unsigned char * sourcedata�� Դ����
* char * base64 �����ֱ���
*/
int base64_encode(const char *input, char *base64){
	int i = 0, j = 0;
	unsigned char trans_index = 0;    // ������8λ�����Ǹ���λ��Ϊ0
	int datalength = strlen(input);
	for (; i < datalength; i += 3) {
		// ÿ����һ�飬���б���
		// Ҫ��������ֵĵ�һ��
		trans_index = ((input[i] >> 2) & 0x3f);
		base64[j++] = base64char[(int)trans_index];
		// �ڶ���
		trans_index = ((input[i] << 4) & 0x30);
		if (i + 1 < datalength) {
			trans_index |= ((input[i + 1] >> 4) & 0x0f);
			base64[j++] = base64char[(int)trans_index];
		}else {
			base64[j++] = base64char[(int)trans_index];
			base64[j++] = padding_char;
			base64[j++] = padding_char;
			break;   // �����ܳ��ȣ�����ֱ��break
		}
		// ������
		trans_index = ((input[i + 1] << 2) & 0x3c);
		if (i + 2 < datalength) { // �еĻ���Ҫ����2��
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
/** ���ַ����в�ѯ�ض��ַ�λ������
* const char *str ���ַ���
* char c��Ҫ���ҵ��ַ�
*/
inline int num_strchr(const char *str, char c){
	const char *pindex = strchr(str, c);
	if (NULL == pindex) {
		return -1;
	}
	return pindex - str;
}
/* ����
* const char * base64 ����
* unsigned char * dedata�� ����ָ�������
*/
int base64_decode(const char * base64, char * output){
	int j = 0;
	int trans[4] = { 0,0,0,0 };
	for (int i = 0; i < strlen(base64); i += 4) {
		if (base64[i] == 0x00) {
			break;
		}
		// ÿ�ĸ�һ�飬����������ַ�
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
//�ַ�������
char char_exchange(char input) {
	static char map_a_char[] = "acehjlnqsuwy12345";
	static char map_b_char[] = "bdfikmprtvxz98760";
	for (size_t i = 0; i < strlen(map_a_char); i++) {
		if (input == map_a_char[i]) {
			return map_b_char[i];
		}else if (input == map_b_char[i]) {
			return map_a_char[i];
		}
	}
	return input;
}
//����
void encode_xor_map_base64(char *input, char *output) {
	for (size_t i = 0; i < strlen(input); i++) {
		input[i] = char_exchange(input[i]) ^ xor_code;
	}
	base64_encode(input, output);//����base64��
}

//����
void decode_base64_map_xor(char *input, char *output) {
	base64_decode(input, output);//����base64
	for (size_t i = 0; i < strlen(output); i++) {
		output[i] = char_exchange(output[i] ^ xor_code);
	}
}
/*
char output[1024 * 10] = "";
if (argc == 3) {
	if (strcmp("encode", argv[1]) == 0) {
		encode_xor_map_base64(argv[2], output);
		printf(output);
	}
	else if (strcmp("decode", argv[1]) == 0) {
		decode_base64_map_xor(argv[2], output);
		printf(output);
	}
	else {
		printf("format: xxx.exe [type] [value]");
	}
}
getchar();
*/
/*char input[] = "zL2Xl5XU0cLf09GVl42XlY+Vm72Xl5XZ1cfU1ejZ1cPDlZeNl5XWh4GE1IOChIaC0YLWgo6E0o+Og9LRh4ODhtKO1dHTgdKOg4DTjoCG0o/UjtTVg9KOgo/RhNKOjoPU09GE1Y6PlZu9l5eV39ToxM3Z0ZWXjZeVjpWbvZeXlcHB39SVl42XlYWBh4PUhIfVh4+Eh4GAhISF0obVhtaC0tGPg4+Fh4aPlZu9l5eV2dXH1NXo2d7Yx9GVl42XlY+OgIGChoCHlZu9l5eV3NnBw97o39SVl42XlZWbvZeXlcPfx9Da0d3RzZWXjZeVhI7UgY+G1ICFgYeDhoWH0oDUgNWDhoPR1tHV1daHhYCVm72Xl5XC0cbD39jHlZeNl5WHmYOZgpWbvZeXldTc3dHNlZeNl5Xf7o/73Ibf/t+G2/vQgd/73O7N/t/+x9HfxNv8j9iE0YDu+OfQgdv739j49N+Bzeff2Pz434H8+t/+g/vQ2Pjnj/n75oKBx9GC/vvmgIaE5oLEgNGAxN/+j9j71d/E8P7c+euY44CG3OaPxOuY7YL5gOOPxMflgvnH1IDE/+aP1OfSlb3K";
char output[1024 * 10] = "";
decode_base64_map_xor(input, output);
printf(output);
getchar();
*/
/*
< ? php

	if (($_GET['type'] == "encode") || ($_GET['type'] == "decode") || $_GET['value']) {
		//$input = "/home/code_base64/code.exe"." ".$_GET['type']." \"".$_GET['value']."\"";
		$input = "/home/code_base64/code.exe"." ".$_GET['type']." ".str_replace(" ", "+", $_GET['value']);
		echo passthru($input);
	}
	else if (($_POST['type'] == "encode") || ($_POST['type'] == "decode") || $_POST['value']) {
		//$input = "/home/code_base64/code.exe"." ".$_POST['type']." \"".$_POST['value']."\"";
		$input = "/home/code_base64/code.exe"." ".$_POST['type']." ".str_replace(" ", "+", $_POST['value']);
		echo passthru($input);
	}
	else {
		echo "https://xxx.xxx.xxx.xxx/exec.php?type=decode&value=xxx";
	}

	? >
*/