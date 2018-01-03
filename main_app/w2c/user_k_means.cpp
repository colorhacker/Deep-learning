

#include "user_k_means.h"


//1.创建N个分类器
//2.对N个分类器初始化值
//3.对样本数据进行1-N进行分类计算，并且标记
//4.对已分类的数据进行中心查找
//5.更新分类器数据
//6.检测所有数据是否被重新分配，若没有退出，否则跳转值第3步

//创建分类器
//target 分类对象
//num 分类个数
//返回 空或链表
//注意：返回的class结构体里面的class_id 是保存的分类类别的元素个数 和数据结构体保存分类ID不一样
user_w2v_words_vector *user_k_means_create_n_class(user_w2v_words_vector *target,int classnum) {
	user_w2v_words_vector *result = NULL;
	int classid = 0;
	if (classnum <= 0) {
		return NULL;
	}
	result = user_w2v_words_vector_create();//创建分类器
	while(classnum--) {
		user_w2v_words_vector_add(result, "class", 0, target->vector_number, target->vector_data);//设置参数 设置元素为0
		target = target->next;//更新目标
	}
	return result;
}

//通过分类器对所有数据进行标记 并且提取新的分类数据
//target：被分类的对象
//nclass：分类器分类点
//type：距离计算方式
//返回：所有点是否转移完成，如果完成返回true
bool user_k_means_mark_data(user_w2v_words_vector *target, user_w2v_words_vector *nclass, distance_type type) {
	user_w2v_words_vector *target_object = target;
	user_w2v_words_vector *class_object = nclass;
	float distance_max = FLT_MAX;//记录最大值
	int class_id = 0;//记录类别
	float distance_vaule = 0;//距离值
	bool result = true;//设置转化完成

	//清除分类元素个数 让其重新分配
	class_object = nclass;//初始化值为0
	while (class_object != NULL) {
		class_object->class_id = 0;//设置类别元素为0
		class_object = class_object->next;//
	}
	//进行数据元素重新分配
	target_object = target;
	while (target_object != NULL) {
		//查找元素属于哪一类
		class_object = nclass;//初始化到起点分类
		distance_max = FLT_MAX;//设置最大值
		while (class_object != NULL) {
			//计算距离
			if (type == cosine) {
				distance_vaule = user_w2v_cos_dist(target_object->vector_data, class_object->vector_data, class_object->vector_number);
			}
			else if (type == euclidean) {
				distance_vaule = user_w2v_eu_dist(target_object->vector_data, class_object->vector_data, class_object->vector_number);
			}
			//找出最小距离的值
			if (distance_vaule < distance_max) {
				distance_max = distance_vaule;//记录最小值
				class_id = class_object->index;//记录ID
			}
			class_object = class_object->next;//指向下一个类别
		}

		//记录类里面的元素个数
		class_object = nclass;//初始化到起点分类
		while (class_object != NULL) {
			if (class_object->index == class_id) {
				class_object->class_id++;//分类中心的类别记录的是分类下面总元素个数
			}
			class_object = class_object->next;//指向下一个类别
		}
		//对数据进行标记
		if (target_object->class_id != class_id) {
			target_object->class_id = class_id;//对类别进行标记
			result = false;
		}	

		target_object = target_object->next;//计算下一个
	}
	return result;
}
//计算一次分类器中心
//target：目标对象
//nclass：分类对象
//返回 成功或失败
bool user_k_means_compute_class(user_w2v_words_vector *target, user_w2v_words_vector *nclass) {
	user_w2v_words_vector *target_object = target;
	user_w2v_words_vector *class_object = nclass;
	float *target_vaule = NULL;
	float *nclass_vaule = NULL;
	int count = 0;

	class_object = nclass;//初始化值为0
	while (class_object != NULL) {
		memset(class_object->vector_data, 0, class_object->vector_number * sizeof(float));//清空内存
		class_object = class_object->next;//
	}
	//计算新的中心
	target_object = target;
	while (target_object != NULL) {
		target_vaule = target_object->vector_data;//取得当前数据的值
		class_object = nclass;//初始化到起点
		while (class_object != NULL) {
			if (target_object->class_id == class_object->index) {//数据类别等于主分类 进行中心值求解 求平均书
				nclass_vaule = class_object->vector_data;//取得当前质心的值
				//平均值作为中心
				for (count = 0; count < target_object->vector_number; count++) {
					*nclass_vaule++ += (float) (*target_vaule++ / class_object->class_id);//更新值
				}
			}
			class_object = class_object->next;//
		}
		target_object = target_object->next;//计算下一个
	}
	return true;
}

//保存分类后的数据
//
void user_k_means_class_fprintf(char *path,user_w2v_words_vector *target,int classnum) {
	user_w2v_words_vector *target_object = target;
	FILE *debug_file = NULL;
	int count = 0;
	debug_file = fopen(path, "w+");
	
	for (count = 1; count < classnum + 1; count++) {
		fprintf(debug_file, "class id :%d \n", count);
		target_object = target;
		while (target_object != NULL) {
			if (target_object->class_id == count) {
				fprintf(debug_file,"%s \n",target_object->words_string);
			}
			target_object = target_object->next;
		}
	}
	fclose(debug_file);
}


/*
	int class_number = 30;//分类个数
	user_w2v_words_vector *model = load_words_vector_model("word2vec_model.bin");//加载模型
	user_w2v_words_vector *class_center = user_k_means_create_n_class(model, class_number);

	while (user_k_means_mark_data(model, class_center, euclidean) == false) {
	user_w2v_words_vector *nc = class_center;
	printf("\n ");
	while (nc!=NULL) {
	printf("%d ",nc->class_id);
	nc = nc->next;
	}
	user_k_means_compute_class(model, class_center);//重新计算中心值
	}
	user_k_means_class_fprintf("k-means.txt", model, class_number);
	printf("\n k-means class compute complete!!");
	user_w2v_words_vector_all_delete(model);//删除模型
	user_w2v_words_vector_all_delete(class_center);//删除分类中心点
	getchar();
	return 0;
*/