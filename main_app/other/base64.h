#ifndef _base64_H
#define _base64_H

#include<stdio.h>
#include<stdlib.h> 
#include<string.h>

#define xor_code 0xb7

int base64_encode(const char *input, char *base64);//base加密
int base64_decode(const char *base64, char *output);//base解密

void encode_xor_map_base64(char *input, char *output);//加密数据
void decode_base64_map_xor(char *input, char *output);//解密数据

#endif