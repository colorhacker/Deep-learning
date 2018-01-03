
#include "user_nn_matrix_file.h"

//在指定位置保存一个矩阵
//file 文件对象
//offset 偏移地址
//matrix 保存的对象
//返回 文件指针位置
long user_nn_model_save_matrix(FILE *file, long offset, user_nn_matrix *matrix) {
	int count = matrix->width * matrix->height;
	fseek(file, offset, SEEK_SET);
	fwrite(matrix->data, count * sizeof(float), 1, file);//写入层
	return ftell(file);
}
//在指定位置保存连续矩阵
//file 文件对象
//offset 偏移地址
//matrices 保存的对象
//返回 文件指针位置
long user_nn_model_save_matrices(FILE *file, long offset, user_nn_list_matrix *matrices) {
	int count = matrices->width * matrices->height;
	user_nn_matrix *matrix = matrices->matrix;
	while (count--) {
		//printf("%d\n", offset);
		offset = user_nn_model_save_matrix(file, offset, matrix);
		//offset = offset + matrix->width * matrix->height * sizeof(float);//移动位置
		matrix = matrix->next;//指向下一个保存的位置
	}
	return ftell(file);
}
//在指定位置读取一个矩阵
//file 文件对象
//offset 偏移地址
//matrix 保存的对象
//返回 文件指针位置
long user_nn_model_read_matrix(FILE *file, long offset, user_nn_matrix *matrix) {
	int count = matrix->width * matrix->height;
	fseek(file, offset, SEEK_SET);
	fread(matrix->data, count * sizeof(float), 1, file);//写入层
	return ftell(file);
}
//在指定位置读取一个连续矩阵
//file 文件对象
//offset 偏移地址
//matrices 保存的对象
//返回 文件指针位置
long user_nn_model_read_matrices(FILE *file, long offset, user_nn_list_matrix *matrices) {
	int count = matrices->width * matrices->height;
	user_nn_matrix *matrix = matrices->matrix;
	while (count--) {
		offset = user_nn_model_read_matrix(file, offset, matrix);
		//offset = offset + matrix->width * matrix->height * sizeof(float);//移动位置
		matrix = matrix->next;//指向下一个保存的位置
	}
	return ftell(file);
}

//连续矩阵保存至文件
//path 文件路径
//offset 偏移地址
//matrices 记录的对象
//返回 成功或失败
bool user_nn_model_file_save_matrices(const char *path, long offset, user_nn_list_matrix *matrices) {
	FILE *model_file = NULL;
	user_nn_matrix *matrix = NULL;
	int matrices_count = 0;
	int matrix_count = 0;
	fopen_s(&model_file, path, "wb+");//采用读写打开模型文件
	if (model_file == NULL)return NULL;

	fseek(model_file, offset, SEEK_SET);//移动至偏移地址
	fprintf(model_file,"M");//写入层
	fwrite(&matrices->width, sizeof(int), 1, model_file);//写入矩阵的高度
	fwrite(&matrices->height, sizeof(int), 1, model_file);//写入矩阵的高度
	//user_nn_model_save_matrices(model_file, offset, matrices);//写入矩阵
	matrices_count = matrices->height * matrices->width;//计算连续矩阵的数目
	matrix = matrices->matrix;//获取矩阵
	while (matrices_count--) {
		//fprintf(model_file, "%d %d ", matrix->height, matrix->width);//写入层
		matrix_count = matrix->width * matrix->height;//获取大小
		fprintf(model_file, "m");//写入标志
		fwrite(&matrix->width, sizeof(int), 1, model_file);//写入矩阵的高度
		fwrite(&matrix->height, sizeof(int), 1, model_file);//写入矩阵的高度	
		fwrite(matrix->data, matrix_count * sizeof(float), 1, model_file);//写入矩阵数据
		matrix = matrix->next;//指向下一个矩阵
	}
	fclose(model_file);
	return true;
}
//从文件载入连续矩阵
//path 文件路径
//offset 偏移地址
//返回 记录的连续矩阵
user_nn_list_matrix *user_nn_model_file_read_matrices(const char *path, long offset) {
	user_nn_list_matrix *matrices=NULL;
	user_nn_matrix *matrix = NULL;
	user_nn_matrix *matrices_matrix = NULL;
	FILE *model_file = NULL;
	char flag = 0;
	int matrices_count=0;
	int matrices_width = 0;
	int matrices_height = 0;

	int matrix_count = 0;
	int matrix_width = 0;
	int matrix_height = 0;

	fopen_s(&model_file, path, "rb");//采用读写打开模型文件
	if (model_file == NULL)return NULL;

	fseek(model_file, offset, SEEK_SET);//移动至偏移地址
	fread(&flag, sizeof(char), 1, model_file);//获取标志位
	if (flag != 'M') {
		fclose(model_file);
		return NULL;
	}
	fread(&matrices_width, sizeof(int), 1, model_file);//写入矩阵的高度
	fread(&matrices_height, sizeof(int), 1, model_file);//写入矩阵的高度
	matrices = (user_nn_list_matrix *)malloc(sizeof(user_nn_list_matrix));//分配空间
	matrices->width = matrices_width;//设置总矩阵宽度
	matrices->height = matrices_height;//设置总矩阵高度

	while (TRUE) {
		fread(&flag, sizeof(char), 1, model_file);//获取标志位
		if (feof(model_file) || (flag != 'm')) {//文件读取结束退出或者读取错误退出
			break;
		}
		fread(&matrix_width, sizeof(int), 1, model_file);//写入矩阵的高度
		fread(&matrix_height, sizeof(int), 1, model_file);//写入矩阵的高度	

		matrix = user_nn_matrix_create(matrix_width, matrix_height);//创建矩阵
		fread(matrix->data, matrix_width * matrix_height * sizeof(float), 1, model_file);//读取矩阵数据
		
		if (matrices_matrix == NULL) {
			matrices_matrix = matrix;
			matrices->matrix = matrices_matrix;//记录数据到连续矩阵中
		}
		else {
			matrices_matrix->next = matrix;
			matrices_matrix = matrices_matrix->next;
		}
	}
	fclose(model_file);
	return matrices;
}

/*
user_nn_list_matrix *input_data = user_nn_matrices_create(1,2,2,2);//创建输入数据
user_nn_matrices_rand_vaule(input_data, 1.0f);//初始化矩阵
user_nn_matrices_printf(NULL, "input", input_data);//打印
user_nn_model_file_save_matrices("test.bin",0, input_data);//保存
input_data  = user_nn_model_file_read_matrices("test.bin", 0);//读取
user_nn_matrices_printf(NULL,"output", input_data);//打印
getchar();
return 0;
*/
