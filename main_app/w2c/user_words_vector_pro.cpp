

#include "user_words_vector_pro.h"


//计算COS夹角 cosine angle
//公式：cos(ai,bi)=向量a*向量b/|向量a|*|向量b|=(a1*b1+a2*b2+...+ai*bi)/(sqrt(a1*a1+a2*a2+...ai*ai)*sqrt(b1*b1+b2*b2+...bi*bi))
float user_w2v_cos_dist(float *a, float *b, int n) {
	float num = 0, dena = 0, denb = 0;

	while (n--) {
		num += *a * *b;//num = (a1*b1+a2*b2+...+ai*bi)
		dena += *a * *a;//dena = sqrt(a1*a1+a2*a2+...ai*ai)
		denb += *b * *b;//denb = sqrt(a1*a1+a2*a2+...ai*ai)
		a++; b++;
	}
	dena = (float)sqrt(dena);
	denb = (float)sqrt(denb);

	return (float)(num / (dena*denb));//cos(ai,bi)=num/(dena*denb)
}
//欧式距离 euclidean metric
//公式：dist(a,b)=sqrt((a1-b1)*(a1-b1)+(a2-b2)*(a1-b2)+...+(ai-bi)*(ai-bi))
float user_w2v_eu_dist(float *a, float *b, int n) {
	float dest = 0;
	while (n--) {
		dest += (*a - *b)*(*a - *b);//
		a++; b++;
	}
	return (float)sqrt(dest);
}


//创建一个用于保存单词组向量的对象
user_w2v_words_vector *user_w2v_words_vector_create(void) {
	user_w2v_words_vector *words_vector = NULL;
	words_vector = (user_w2v_words_vector *)malloc(sizeof(user_w2v_words_vector));
	words_vector->prior = NULL;
	words_vector->index = 0;
	words_vector->words_string = NULL;
	words_vector->class_id = 0;
	words_vector->vector_number = 0;
	words_vector->vector_data = NULL;
	words_vector->next = NULL;

	return words_vector;
}
//删除一个向量并且释放内存
void user_w2v_words_vector_delete(user_w2v_words_vector *dest) {
	if (dest != NULL) {
		if (dest->vector_data != NULL) {
			free(dest->vector_data);//释放数据
		}
		if (dest->words_string != NULL) {
			free(dest->words_string);//释放数据
		}
		free(dest);//释放结构体
	}
}
//删除所有连续向量
void user_w2v_words_vector_all_delete(user_w2v_words_vector *dest) {
	user_w2v_words_vector *next = dest->next;
	while (dest != NULL) {
		next = dest->next;
		user_w2v_words_vector_delete(dest);//删除当前指针数据
		dest = next;//继续处理下一个
	}
}
//填充一个向量结构体
//dest：填充对象
//index：填充序号
//words：字符串对象
//size：向量个数
//vector：向量指针
//返回：成功或失败
bool user_w2v_words_vector_fill(user_w2v_words_vector *dest,int index, char *words,int classid, int size, float *vector) {
	if (dest == NULL) {
		return false;
	}else{
		dest->index = index;//设置起始位置
		dest->words_string = (char *)malloc(strlen(words) + 1);//分配内存 预留添加结束符字节
		memcpy(dest->words_string, words, strlen(words) + 1);//保存数据 添加结束符
		dest->words_string[strlen(words)] = '\0';//设置结束符0
		dest->class_id = classid;//填充分类ID

		dest->vector_number = size;//保存大小
		dest->vector_data = (float *)malloc((int)size * sizeof(float));//分配数据内存
		memcpy(dest->vector_data, vector, (int)size * sizeof(float));//保存数据		
	}
	return true;
}

//添加一个用于保存单词组向量的对象
//dest：填充对象
//words：字符串对象
//size：向量个数
//classid:词向量类别
//vector：向量指针
//返回：成功或失败
bool user_w2v_words_vector_add(user_w2v_words_vector *dest,char *words,int classid,int size,float *vector) {
	if (dest == NULL) {
		return false;
	}
	else {
			//如果内存为空那么直接进行填充
			if (dest->index == 0) {
				if (user_w2v_words_vector_fill(dest,1, words, classid, size, vector)) {
					return true;
				}
			}
			else {
				while (dest->next != NULL) {
					dest = dest->next;//指向下一个结构体
				}
				dest->next = user_w2v_words_vector_create();//分配数据空间
				dest->next->prior = dest;//设置上一个结构体的位置
				if (user_w2v_words_vector_fill(dest->next, dest->index+1, words, classid, size, vector)) {
					return true;
				}
			}
	}
	return false;
}

//计算两个words之间的cos夹角
//model：模型对象
//type：计算方式
//stra：数据字符串a
//strb：数据字符串b
//返回：距离
float user_w2v_words_similar(user_w2v_words_vector *model, distance_type type,char *stra, char *strb) {
	user_w2v_words_vector *word_a = NULL;
	user_w2v_words_vector *word_b = NULL;
	while (1) {
		if ((word_a != NULL) && (word_b != NULL)) {
			if (word_a->vector_number == word_b->vector_number) {
				/*//测试开始
				float result = user_w2v_cos_dist(word_a->vector_data, word_b->vector_data, word_a->vector_number);
				printf("\n %s to %s :%f", stra, strb, result);
				return result;
				//测试结束*/
				if (type == cosine) {
					return user_w2v_cos_dist(word_a->vector_data, word_b->vector_data, word_a->vector_number);
				}
				else if (type == euclidean) {
					return user_w2v_eu_dist(word_a->vector_data, word_b->vector_data, word_a->vector_number);
				}
			}
			else {
				return 0;
			}
		}
		else {
			if (model == NULL) {
				return 0;//找不到模型
			}
			if (strcmp(model->words_string, stra) == 0) {
				word_a = model;
			}
			if (strcmp(model->words_string, strb) == 0) {
				word_b = model;
			}
			model = model->next;//继续检索下一个模型
		}
	}
	
	return 0;
}

//读取标签
//f：文件指针
//string：保存数据的指针
//返回：成功或失败
bool user_w2v_read_words_string_txt_utf8(FILE *f, char *string) {
	fscanf(f, "%*[\n|' ']%s", string);//utf8编码 读取标签
	return true;
}
//读取词向量
//f：文件指针
//count：读取float数据个数
//mem：用于保存数据的内存
//返回：成功或失败
bool user_w2v_read_words_vector_txt_utf8(FILE *f, int count, float *mem) {
	while (count--) {
		fscanf_s(f, "%*[\n|' ']%f", mem++);//读取词向量数据
	}
	return true;
}

//加载一个文本格式的词向量到内存中 并且返回双向链表
//path：文件路径
//返回：词向量结构体链表
user_w2v_words_vector *load_words_vector_model(char *path) {
	user_w2v_words_vector *words_vector_list = NULL;
	FILE *file = fopen(path, "r");   //使用二进制打开一个模型文件
	int words = 0, size = 0;

	if (file == NULL) {
		printf("Input file not found\n");
		return NULL;
	}
	//读取模型文件数据，转化为：名称<->数据
	fscanf(file, "%d", &words);    //读取所有单词个数
	fscanf(file, "%d", &size);     //读取每个单词的维度大小 
	printf("\ntotal number words:%d \ndimension  size : %d ", words, size);

	words_vector_list = user_w2v_words_vector_create();//创建一个结构体

	char  *words_string = (char *)malloc(1024);//分配1K内存用于保持单词标签
	float *words_vector = (float *)malloc((int)size * sizeof(float));//分配数据内存

	for (int i = 0; i<words; i++) {
		memset(words_string, 0, sizeof(words_string));//清空数据
		user_w2v_read_words_string_txt_utf8(file, words_string);//读取标签
		user_w2v_read_words_vector_txt_utf8(file, (int)size, words_vector);//读取词向量
		user_w2v_words_vector_add(words_vector_list, words_string, 0, (int)size, words_vector);//添加数据
	}
	free(words_string);
	free(words_vector);
	fclose(file);

	return words_vector_list;
}

//创建一个用于保存链表向量的对象
//返回：分配内存后的相似度链表
user_w2v_similar_list *user_w2v_similar_list_create(void) {
	user_w2v_similar_list *similar_list = NULL;
	similar_list = (user_w2v_similar_list *)malloc(sizeof(user_w2v_similar_list));
	similar_list->prior = NULL;
	similar_list->index = 0;
	similar_list->result_addr = NULL;
	similar_list->result_similar = 0;
	similar_list->next = NULL;

	return similar_list;
}
//填充list表
//dest：链表对象
//index：序号
//addr：字符串指针
//similar：相似度值
//返回：成功或失败
bool user_w2v_similar_list_fill(user_w2v_similar_list *dest,int index,user_w2v_words_vector *addr, float similar) {
	if (dest == NULL) {
		return false;
	}
	else {
		dest->index = index;//设置起始位置
		dest->result_addr = addr;
		dest->result_similar = similar;
	}
	return true;
}
//添加一个用于保存单词组向量的对象
//dest：链表对象
//addr：字符串指针
//similar：相似度值
//返回：成功或失败
bool user_w2v_similar_list_add(user_w2v_similar_list *dest, user_w2v_words_vector *addr,float similar) {

	if (dest == NULL) {
		return false;
	}
	else {
		//如果内存为空那么直接进行填充
		if (dest->index == 0) {
			if (user_w2v_similar_list_fill(dest, 1, addr, similar)) {
				return true;
			}
		}
		else {
			while (dest->next != NULL) {
				dest = dest->next;//指向下一个结构体
			}
			dest->next = user_w2v_similar_list_create();//分配数据空间
			dest->next->prior = dest;//设置上一个结构体的位置
			if (user_w2v_similar_list_fill(dest->next, dest->index + 1, addr, similar)) {
				return true;
			}
		}

	}
	return false;
}
//删除一个链表并且释放内存
//dest：删除对象
//返回值：无
void user_w2v_similar_list_delete(user_w2v_similar_list *dest) {
	if (dest != NULL) {
		free(dest);//释放结构体
	}
}
//删除所有连续的链表对象
//dest：删除的起始对象
//返回值：无
void user_w2v_similar_list_all_delete(user_w2v_similar_list *dest) {
	user_w2v_similar_list *next = dest->next;
	while (dest != NULL) {
		next = dest->next;
		user_w2v_similar_list_delete(dest);//删除当前指针数据
		dest = next;//继续处理下一个
	}
}
//找出模型中阈值以内的字符
//model：模型对象
//type：距离方式
//string：目标单词
//thrd：阈值
//返回：结果
user_w2v_similar_list *user_w2v_words_most_similar(user_w2v_words_vector *model, distance_type type, char *string,float thrd) {
	user_w2v_words_vector *word_a = model;
	user_w2v_words_vector *word_b = model;
	user_w2v_similar_list *result = user_w2v_similar_list_create();
	float similar_vaule = 0;

	while (word_a != NULL) {
		if (strcmp(word_a->words_string, string) == 0) {
			while (word_b != NULL) {
				if (word_a->vector_number == word_b->vector_number) {
					if (type == cosine) {//cosine距离是夹角距离那么越相似数据越大，那么阈值是最小值
						similar_vaule = user_w2v_cos_dist(word_a->vector_data, word_b->vector_data, word_a->vector_number);//进行计算
						if (similar_vaule >= thrd) {
							user_w2v_similar_list_add(result, word_b, similar_vaule);//添加数据
						}
					}
					else if (type == euclidean) {//欧式距离是移动距离，月相似距离越小，那么阈值应该是最大值
						similar_vaule = user_w2v_eu_dist(word_a->vector_data, word_b->vector_data, word_a->vector_number);//进行计算
						//similar_vaule = 1 - 1 / similar_vaule;
						//printf("\n%f\n", similar_vaule);
						if (similar_vaule <= thrd) {
							user_w2v_similar_list_add(result, word_b, similar_vaule);//添加数据
						}
					}

				}
				word_b = word_b->next;//继续检索下一个模型
			}
			break;
		}
		word_a = word_a->next;//继续检索下一个模型
	}
	return result;
}


//排序链表
//dest：目标对象 排序后会自动删除
//type：表示降序或者升序
//返回：排序后的链表
//
user_w2v_similar_list *user_w2v_similar_sorting(user_w2v_similar_list *dest, sorting_type type) {
	user_w2v_similar_list *result_count = dest;
	user_w2v_similar_list *result_nor = NULL;
	user_w2v_similar_list *result = user_w2v_similar_list_create();
	float similar_nor = 0;//临时保存最大距离
	int index_max = 0;//临时保存最大距离位置
	user_w2v_words_vector *addr_max = NULL;//保存地址

	while (result_count != NULL){
		result_count = result_count->next;
		//开始查询一次极限值
		if (type == sorting_down) {
			similar_nor = -FLT_MAX;//设置极限标准值
		}
		else if (type == sorting_up) {
			similar_nor = FLT_MAX;//设置极限标准值
		}
		result_nor = dest;//初始化值到开头
		while (result_nor != NULL) {
			if (type == sorting_down) {
				if (similar_nor <= result_nor->result_similar) {
					index_max = result_nor->index;//获取指数
					similar_nor = result_nor->result_similar;//这是最大值
					addr_max = result_nor->result_addr;
				}
			}
			else if (type == sorting_up) {
				if ( result_nor->result_similar <= similar_nor) {
					index_max = result_nor->index;//获取指数
					similar_nor = result_nor->result_similar;//这是最大值
					addr_max = result_nor->result_addr;
				}
			}

			result_nor = result_nor->next;//继续检索下一个
		}
		//结束查询极限值
		//开始删除极限值
		result_nor = dest;//初始化值到开头
		while (result_nor != NULL) {
			if (index_max == result_nor->index) {
				if (type == sorting_down) {
					result_nor->result_similar = -FLT_MAX;//设置极限标准值
				}
				else if (type == sorting_up) {
					result_nor->result_similar = FLT_MAX;//设置极限标准值
				}
			}
			result_nor = result_nor->next;//继续检索下一个
		}
		//结束删除最大值
		//保存更新后的值
		user_w2v_similar_list_add(result, addr_max, similar_nor);//添加数据
		//printf("\nv:%s,    %f", addr_max->words_string, similar_nor);
	}
	user_w2v_similar_list_all_delete(dest);//删除结果链表
	return result;
}
//打印链表数据
//参数
//to_file：是否输出到文件
//similar_list：链表数据
//返回 无
void user_w2v_similar_list_printf(bool to_file, user_w2v_similar_list *similar_list) {
	FILE *debug_file = NULL;
	//debug_file = fopen("debug.txt", "w+");
	if (to_file == true)
		debug_file = fopen("words_vec_debug.txt", "w+");

	while (similar_list != NULL) {
		printf("%s\t\t%-10.6f \n", similar_list->result_addr->words_string, similar_list->result_similar);
		if (debug_file != NULL)
			fprintf(debug_file, "%s\t\t%-10.6f \n", similar_list->result_addr->words_string, similar_list->result_similar);
		similar_list = similar_list->next;
	}
	printf("\n\n");
	if (debug_file != NULL)
		fprintf(debug_file, "\n");
	//fclose(debug_file);
	if (debug_file != NULL)
		fflush(debug_file);
}
