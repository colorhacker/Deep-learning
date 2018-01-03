#ifndef _user_words_vector_pro_H
#define _user_words_vector_pro_H

#include <string.h>  
#include <math.h>  
#include <malloc.h>  
#include <stdio.h>  
#include "../matrix/user_nn_matrix.h"

float user_w2v_cos_dist(float *a, float *b, int n);//计算cos夹角
float user_w2v_eu_dist(float *a, float *b, int n);//计算欧式距离

//创建一个list结构体用于保存单词或句子的维度信息
typedef struct words_vector {
	struct words_vector *prior;//上一个ceng
	int index;//指数
	char *words_string;//对象字符串
	int class_id;//分类类别
	int  vector_number;//向量个数
	float *vector_data;//向量数据
	struct words_vector *next;//下一层
}user_w2v_words_vector;

user_w2v_words_vector *user_w2v_words_vector_create(void);//创建一个词向量的结构体
void user_w2v_words_vector_delete(user_w2v_words_vector *dest);//删除一个词向量的结构体
void user_w2v_words_vector_all_delete(user_w2v_words_vector *dest);//删除所有连续的词向量结构体
bool user_w2v_words_vector_add(user_w2v_words_vector *dest, char *words, int classid, int size, float *vector);//保存词向量


enum distance_type {
	cosine = 0,//cos距离
	euclidean = 1//欧式距离
};
typedef struct similar_list {
	struct similar_list *prior;//上一个
	int index;//指数
	struct words_vector *result_addr;//保存的单词对象地址
	float  result_similar;//结果距离
	struct similar_list *next;//下一个
}user_w2v_similar_list;
bool user_w2v_read_words_string_txt_utf8(FILE *f, char *string);//在一个读取一个字符串 空格结束
bool user_w2v_read_words_vector_txt_utf8(FILE *f, int count, float *mem);//读取一组词向量 换行符为结束
user_w2v_words_vector *load_words_vector_model(char *path);//加载一个模型文件 到双向链表里面
user_w2v_similar_list *user_w2v_similar_list_create(void);//创建一个保存距离的链表
void user_w2v_similar_list_delete(user_w2v_similar_list *dest);//删除一个保存距离的链表
void user_w2v_similar_list_all_delete(user_w2v_similar_list *dest);//删除所有保存距离的链表
float user_w2v_words_similar(user_w2v_words_vector *model, distance_type type, char *stra, char *strb);//计算两个词语之间的距离
user_w2v_similar_list *user_w2v_words_most_similar(user_w2v_words_vector *model, distance_type type, char *string, float thrd);//找出指定词语相近的一系列词语并且返回
user_w2v_similar_list *user_w2v_similar_sorting(user_w2v_similar_list *dest, sorting_type type);//分配列表 降序排列

void user_w2v_similar_list_printf(bool to_file, user_w2v_similar_list *similar_list);//打印数据

#endif