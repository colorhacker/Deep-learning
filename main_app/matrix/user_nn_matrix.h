#ifndef _user_nn_matrix_H
#define _user_nn_matrix_H

#include "../user_config.h"
//#define USE_CUDA

typedef enum _nn_conv2_type{
	u_nn_conv2_type_null = 0,
	u_nn_conv2_type_full,
	u_nn_conv2_type_same,
	u_nn_conv2_type_valid
}user_nn_conv2_type;

typedef enum _nn_pooling_type{
	u_nn_pooling_type_null = 0,
	u_nn_pooling_type_mean,
	u_nn_pooling_type_max
}user_nn_pooling_type;

typedef struct _nn_bias{
	float bias;//存放参数
	struct _nn_bias *next;//指向下一个矩阵结构体
}user_nn_bias;
typedef struct _nn_list_biases{
	int biasescount;//总参数的个数
	user_nn_bias *bias;//指向下一个矩阵结构体
}user_nn_list_biases;
//矩阵描述
typedef struct _nn_matrix{
	int width;//矩阵的宽度
	int height;//矩阵的高度
	//float biases;//偏置参数
	float *data;//数据内容
	struct _nn_matrix *next;//指向下一个矩阵结构体
}user_nn_matrix;
//连续矩阵描述
typedef struct _nn_list_matrix{
	int width;//二维矩阵的宽度
	int height;//二维矩阵的高度
	user_nn_matrix *matrix;
}user_nn_list_matrix;
//排序类型
typedef enum _sorting_type {
	sorting_up = 0,//升序
	sorting_down = 1//降序
}sorting_type;
//求取QR矩阵的方式
typedef enum _eigs_type {
	qr_householder = 0,//
	qr_givens = 1//
}eigs_type;

user_nn_matrix *user_nn_matrix_create(int width, int height);//创建一个宽度为width 高度为height的矩阵
user_nn_matrix *user_nn_matrix_cpy_create(user_nn_matrix *dest_matrix);//复制一个矩阵，返回新的矩阵
void user_nn_matrix_transpose(user_nn_matrix *src_matrix);//矩阵转置 交换一个矩阵的长度和宽度大小并且交换数据
float *user_nn_matrix_ext_value_index(user_nn_matrix *dest, int post_index);//获取矩阵中的一个值的指针，按照一维矩阵的方式来获取
float *user_nn_matrix_ext_value(user_nn_matrix *dest, int postx, int posty);//获取矩阵中的一个值的指针，按照二维矩阵的方式来获取

int user_nn_matrix_return_max_index(user_nn_matrix *dest);//返回矩阵中的最大值在矩阵中的位置index
float *user_nn_matrix_return_max_addr(user_nn_matrix *dest);//返回矩阵中的最大值的地址
int user_nn_matrix_return_min_index(user_nn_matrix *dest);//返回矩阵中的最小值的位置index
float *user_nn_matrix_return_min_addr(user_nn_matrix *dest);//返回矩阵中的最小值的地址

void user_nn_matrix_delete(user_nn_matrix *dest);//删除一个矩阵 释放内存
user_nn_list_matrix *user_nn_matrices_create(int total_w, int total_h, int matrix_w, int matrix_h);//创建一个连续的矩阵连续个数为 total_w*total_h，其中每个矩阵大小为matrix_w*matrix_h
user_nn_list_matrix *user_nn_matrices_create_head(int total_w, int total_h);//创建一个连续矩阵头，内部不包含矩阵
void user_nn_matrices_delete(user_nn_list_matrix *src_matrices);//删除连续矩阵
bool user_nn_matrices_add_matrix(user_nn_list_matrix *list_matrix, user_nn_matrix *end_matirx);//在连续矩阵末尾处添加一个矩阵
user_nn_matrix *user_nn_matrices_ext_matrix(user_nn_list_matrix *list_matrix, int postx, int posty);//从连续矩阵中返回指定位置（postx，posty）矩阵指针
user_nn_matrix *user_nn_matrices_ext_matrix_index(user_nn_list_matrix *list_matrix, int index);//从连续矩阵中返回指定位置index矩阵指针
void user_nn_matrices_to_matrix(user_nn_matrix *src_matrix, user_nn_list_matrix *sub_matrices);//把一个连续矩阵转化为一个矩阵，其中的值为连续矩阵的所有值，此转化后的矩阵的宽度为1，高度为数据个数总和
void user_nn_matrix_to_matrices(user_nn_list_matrix *src_matrices, user_nn_matrix *sub_matrix);//把矩阵src_matrix转化为src_matrices的连续矩阵
void user_nn_matrices_cpy_matrices(user_nn_list_matrix *src_matrices, user_nn_list_matrix *dest_matrices);//拷贝连续矩阵dest_matrices到连续矩阵src_matrices中
bool user_nn_matrix_cpy_array(float *dest_data,user_nn_matrix *src_matrix, int startx, int starty, int width, int height); //拷贝src_matrix矩阵(x,y)起点大小为width*height的数据至dest_data,其中，返回失败或成功
bool user_nn_matrix_cpy_array_mult_constant(float *dest_data, user_nn_matrix *src_matrix, int startx, int starty, int width, int height, float constant); //拷贝src_matrix*constant矩阵(x,y)起点大小为width*height的数据至dest_data,其中，返回失败或成功
user_nn_matrix *user_nn_matrix_expand_mult_constant(user_nn_matrix *src_matrix, int width, int height, float constant);//扩充矩阵src_matrix， 每个像素按照给定扩大width和height倍 扩充之后乘以系数bias
user_nn_matrix *user_nn_matrix_expand(user_nn_matrix *src_matrix, int above, int below, int left, int right);//扩充矩阵src_matrix，按照上下、左右各扩充指定大小

user_nn_matrix *user_nn_matrix_rotate180(user_nn_matrix *src_matrix);//矩阵src_matrix旋转180°
void user_nn_matrix_poit_mult_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//矩阵src_matrix.*sub_matrix 结果保存在src_matrix里面
user_nn_matrix *user_nn_matrix_mult_matrix(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//矩阵src_matrix*sub_matrix 返回乘积后的矩阵
void user_nn_matrix_mult_constant(user_nn_matrix *src_matrix, float constant);//矩阵src_matrix*v  结果保存在src_matrix里面
void user_nn_matrix_divi_constant(user_nn_matrix *src_matrix, float constant);//矩阵src_matrix/v 结果保存在src_matrix里面
void user_nn_matrix_sum_constant(user_nn_matrix *src_matrix, float constant);//矩阵src_matrix的每个数据加上constant
void user_nn_matrix_sum_matrix_mult_alpha(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, float alpha);//矩阵save_matrix求和矩阵src_matrix与alpha的乘积
void user_nn_matrix_memset(user_nn_matrix *save_matrix, float constant);//设置矩阵src_matrix的值 结果保存在src_matrix里面
void user_nn_matrix_memcpy(user_nn_matrix *save_matrix, float *data);//拷贝数据至矩阵 大小大于等于矩阵大小
void user_nn_matrix_memcpy_uchar_mult_constant(user_nn_matrix *save_matrix, unsigned char *input_array, float constant);//拷贝数据至矩阵 大小大于等于矩阵大小
void user_nn_matrix_uchar_memcpy(unsigned char *save_array,user_nn_matrix *src_matrix);//拷贝数据至矩阵 大小大于等于矩阵大小
user_nn_matrix *user_nn_matrix_sorting(user_nn_matrix *src_matrix, sorting_type type);//对矩阵数据进行排序,返回一个新的矩阵，不会删掉原来的矩阵

float user_nn_matrix_cum_element(user_nn_matrix *src_matrix);//求和矩阵里面所有值 加在一起
void user_nn_matrix_cum_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//求和两个矩阵 矩阵src_matrix与矩阵sub_matrix每个元素进行加法运算 结果保存在src_matrix里面
void user_nn_matrix_cum_matrix_alpha(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, float alpha);//求和两个矩阵 矩阵src_matrix与矩阵sub_matrix*alpha每个元素进行加法运算 结果保存在src_matrix里面
void user_nn_matrix_cum_matrix_mult_alpha(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, float alpha);//求和两个矩阵 矩阵src_matrix与矩阵sub_matrix*alpha每个元素进行加法运算 结果保存在src_matrix里面
void user_nn_matrix_cpy_matrix(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix);//拷贝矩阵sub_matrix数据到矩阵src_matrix中
void user_nn_matrix_cpy_matrix_p(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix);//指向矩阵sub_matrix数据到矩阵src_matrix中
void user_nn_matrix_cpy_matrix_sum_constant(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix, int index, float constant);//拷贝矩阵到指定矩阵中，在给定位置地方进行求和constant
float user_nn_matrix_mult_cum_matrix(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//矩阵src_matrix.*sub_matrix 然后求和矩阵里面所有值 （条件要求矩阵大小完全一致）
user_nn_matrix *user_nn_matrix_ext_matrix(user_nn_matrix *src_matrix, int startx, int starty, int width, int height);//在矩阵src_matrix 中提取指定（startx,starty）位置为基点 的指定width、height大小的矩阵区域
bool user_nn_matrix_save_array(user_nn_matrix *src_matrix, float *save_data, int startx, int starty, int width, int height); //在src_matrix矩阵指定(x, y)位置保存save_datad的内存数据，返回失败或成功
bool user_nn_matrix_save_float(user_nn_matrix *src_matrix, int startx, int starty, float vaule);//在指定位置保存一个值，返回失败或成功
bool user_nn_matrix_save_matrix(user_nn_matrix *src_matrix, user_nn_matrix *save_matrix, int startx, int starty);//在src_matrix矩阵指定(x,y)位置保存save_matrix矩阵
bool user_nn_matrix_sum_array_mult_alpha(user_nn_matrix *dest_matrix, float *src_data, float alpha, int startx, int starty, int width, int height);//在dest_matrix矩阵中指定位置叠加src_data*alpha的矩阵
void user_nn_matrix_pooling(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *kernel_matrix);//池化矩阵  矩阵sub_matrix卷积src_matrix每个区域，区域不重叠 返回新的结果矩阵
user_nn_matrix *user_nn_matrix_conv2(user_nn_matrix *src_matrix, user_nn_matrix *kernel_matrix, user_nn_conv2_type type);//卷积矩阵  矩阵sub_matrix卷积src_matrix 返回新的结果矩阵

float user_nn_matrix_get_mse(user_nn_matrix *src_matrix);//求取均方误差
float user_nn_matrix_get_rmse(user_nn_matrix *src_matrix);//求取均方根误差

//matlab函数
user_nn_matrix *user_nn_matrix_repmat(user_nn_matrix *dest, int m, int n);//矩阵复制
void user_nn_matrix_eye(user_nn_matrix *dest);//设置矩阵对角线为1
user_nn_matrix *user_nn_givens(float x, float y);//求解givens 旋转后的值
user_nn_list_matrix *user_nn_householder_qr(user_nn_matrix *dest);//求解A=QR
user_nn_list_matrix *user_nn_givens_qr(user_nn_matrix *dest);//求解A=QR
user_nn_list_matrix *user_nn_eigs(user_nn_matrix *dest, float epsilon, eigs_type type);//反复迭代A=QR求取矩阵特征值
user_nn_matrix *user_nn_matrix_mean(user_nn_matrix *src_matrix);//求解矩阵的平均值
user_nn_matrix *user_nn_matrix_cov(user_nn_matrix *src_matrix);//求解协方差矩阵
user_nn_matrix *user_nn_tril_indices(int width, int height, float details);//求取方阵的三角矩阵坐标 返回左下角所有坐标不包含中线
float user_nn_matrix_trace(user_nn_matrix *src_matrix);//求和对角线元素
user_nn_matrix *user_nn_matrix_diag(user_nn_matrix *src_matrix);//返回对角线元素生成新的矩阵
float user_nn_matrix_norm(user_nn_matrix *src_matrix);//求解矩阵的范数 2D
user_nn_matrix *user_nn_matrix_outer(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//计算两个矩阵向量的outer值
//
user_nn_matrix *user_nn_matrix_cut_vector(user_nn_matrix *src_matrix, user_nn_matrix *diag_matrix,float epsilon);//通过对角矩阵的排序来获取新的矩阵
//

void user_nn_matrix_printf(FILE *debug_file, user_nn_matrix *src_matrix);//打印矩阵
void user_nn_matrices_printf(FILE *debug_file, char *title, user_nn_list_matrix *src_matrix);//打印连续矩阵

#endif




/*
//把一个矩阵数据拷贝到另外一个矩阵中
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(6, 6);//创建3*3大小的二维矩阵
dest   = user_nn_matrix_create(2, 2);//创建2*2大小的二维矩阵
user_nn_matrix_rand_vaule(matrix,1);

user_nn_matrix_printf(NULL, matrix);//打印矩阵
bool is_success = user_nn_matrix_save_matrix(matrix, dest, 1, 4);
printf("\n%s\n", is_success==true?"true":"false");

user_nn_matrix_printf(NULL, matrix);//打印矩阵

getchar();
return 0;

*/
/*
//把一个内存数据拷贝到另外一个矩阵中 按照规定参数
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(6, 6);//创建3*3大小的二维矩阵
dest = user_nn_matrix_create(2, 2);//创建2*2大小的二维矩阵
user_nn_matrix_rand_vaule(matrix, 1);

user_nn_matrix_printf(NULL, matrix);//打印矩阵
bool is_success = user_nn_matrix_save_array(matrix, dest->data, 2, 3, dest->width, dest->height);
printf("\n%s\n", is_success == true ? "true" : "false");

user_nn_matrix_printf(NULL, matrix);//打印矩阵

getchar();
return 0;
*/

/* 池化矩阵
user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;
float *result = NULL;


src_matrix = user_nn_matrix_create(28, 28);
sub_matrix = user_nn_matrix_create(2, 2);

user_nn_matrix_memset(src_matrix, 2.5);//设置矩阵值
user_nn_matrix_memset(sub_matrix, 0.25);//设置矩阵值

result = user_nn_matrix_ext_value(src_matrix, 27, 27);
*result = 1;

res_matrix = user_nn_matrix_pool(src_matrix, sub_matrix);
if (res_matrix != NULL){
user_nn_matrix_printf(NULL,res_matrix);//打印矩阵
}
else{
printf("null\n");
}
*/
/* 矩阵卷积测试
user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;

src_matrix = user_nn_matrix_create(28, 28);
sub_matrix = user_nn_matrix_create(5, 5);

user_nn_matrix_memset(src_matrix, 2.5);//设置矩阵值
user_nn_matrix_memset(sub_matrix, 1.0);//设置矩阵值

res_matrix = user_nn_matrix_conv2(src_matrix, sub_matrix, u_nn_conv2_type_valid);
if (res_matrix != NULL){
user_nn_matrix_printf(NULL,res_matrix);//打印矩阵
}
else{
printf("null\n");
}
*/
/* 二维矩阵 提取
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;

list = user_nn_matrices_create(2, 2, 1, 1);//创建2*2个1*1大小的二维矩阵
user_nn_matrix_memset(list->matrix, 1);
user_nn_matrix_memset(list->matrix->next, 2);
user_nn_matrix_memset(list->matrix->next->next, 3);
user_nn_matrix_memset(list->matrix->next->next->next, 4);

dest = user_nn_matrices_ext_matrix(list, 1, 0);//获取其中一个矩阵

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//打印矩阵
}
else{
printf("null\n");
}
*/
/* 矩阵截取测试
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(2, 2);//创建2*2大小的二维矩阵
matrix->data[0] = 1.0;
matrix->data[1] = 2.0;
matrix->data[2] = 3.0;
matrix->data[3] = 4.0;

dest = user_nn_matrix_ext_matrix(matrix, 1, 1, 1, 1);//截取矩阵

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//打印矩阵
}
else{
printf("null\n");
}
*/


/*获取矩阵中的一个值
user_nn_matrix *matrix = NULL;
float *p;

matrix = user_nn_matrix_create(2, 2);//创建2*2大小的二维矩阵
matrix->data[0] = 1.0;
matrix->data[1] = 2.0;
matrix->data[2] = 3.0;
matrix->data[3] = 4.0;

p = user_nn_matrix_ext_value(matrix, 1, 1);//获取矩阵值
if (p != NULL)
printf("%f \n", *p);
*/
/*偏置参数测试
user_nn_list_biases *biases = NULL;
user_nn_bias *dest = NULL;

biases = user_nn_biases_create(4);

biases->bias->bias = 1.0;
biases->bias->next->bias = 2.0;
biases->bias->next->next->bias = 3.0;
biases->bias->next->next->next->bias = 4.0;

dest = user_nn_biases_ext_bias(biases,0);
if (dest != NULL)
printf("%f \n", dest->bias);
*/
/* 矩阵乘法
user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;

src_matrix = user_nn_matrix_create(1280, 1280);
sub_matrix = user_nn_matrix_create(1280, 1280);

for (int count = 0; count < (src_matrix->width * src_matrix->height); count++) {
src_matrix->data[count] = (float)count * 0.01f;
sub_matrix->data[count] = (float)count * 0.01f;
}
res_matrix = user_nn_matrix_mult_matrix(src_matrix, sub_matrix);//矩阵相乘
if (res_matrix != NULL) {
user_nn_matrix_printf(NULL, res_matrix);//打印矩阵
}
else {
printf("null\n");
}
printf("\nend");
_getch();
return 1;
*/
/*单个矩阵转化为 高度为1的连续链表矩阵
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;

dest = user_nn_matrix_create(2,2);
dest->data[0] = 1.0;
dest->data[1] = 2.0;
dest->data[2] = 3.0;
dest->data[3] = 4.0;

list = user_nn_matrix_to_matrices(dest,1,1);//获取其中一个矩阵

if (list != NULL){
user_nn_matrix_printf(NULL,list->matrix);//打印矩阵
user_nn_matrix_printf(NULL,list->matrix->next);//打印矩阵
user_nn_matrix_printf(NULL,list->matrix->next->next);//打印矩阵
user_nn_matrix_printf(NULL,list->matrix->next->next->next);//打印矩阵
}
else{
printf("null\n");
}
*/
/*链表矩阵转化为高度为1的单个矩阵
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;

list = user_nn_matrices_create(5, 2, 1, 1);//创建2*2个1*1大小的二维矩阵
user_nn_matrix_memset(list->matrix, 1);
user_nn_matrix_memset(list->matrix->next, 2);
user_nn_matrix_memset(list->matrix->next->next, 3);
user_nn_matrix_memset(list->matrix->next->next->next, 4);

dest = user_nn_matrices_to_matrix(list);//获取其中一个矩阵

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//打印矩阵
}
else{
printf("null\n");
}
*/
/*链表偏置参数转矩阵测试
user_nn_list_biases *list = NULL;
user_nn_matrix *dest = NULL;

list = user_nn_biases_create(1);

list->bias->bias = 1.0;
list->bias->next->bias = 2.0;
list->bias->next->next->bias = 3.0;
list->bias->next->next->next->bias = 4.0;
list->bias->next->next->next->next->bias = 5.0;

dest = user_nn_biases_to_matrix(list);

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//打印矩阵
}
else{
printf("null\n");
}
*/
/*矩阵均值扩充
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(2, 2);//创建2*2大小的二维矩阵
matrix->data[0] = 1.0;
matrix->data[1] = 2.0;
matrix->data[2] = 3.0;
matrix->data[3] = 4.0;

dest = user_nn_matrix_expand_matrix(matrix, 2, 3);//

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//打印矩阵
}
else{
printf("null\n");
}
*/
/*矩阵扩充 边框扩充
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(1, 1);//创建2*2大小的二维矩阵
matrix->data[0] = 1.0;
//matrix->data[1] = 2.0;
//matrix->data[2] = 3.0;
//matrix->data[3] = 4.0;

dest = user_nn_matrix_expand(matrix, 1, 1);//

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//打印矩阵
}
else{
printf("null\n");
}
*/
/*连续矩阵拷贝
user_nn_list_matrix *list = NULL;
user_nn_list_matrix *dest = NULL;

list = user_nn_matrices_create(5, 2, 1, 1);//创建2*2个1*1大小的二维矩阵
dest = user_nn_matrices_create(5, 2, 1, 1);//创建2*2个1*1大小的二维矩阵

user_nn_matrix_memset(list->matrix, 1);
user_nn_matrix_memset(list->matrix->next, 2);
user_nn_matrix_memset(list->matrix->next->next, 3);
user_nn_matrix_memset(list->matrix->next->next->next, 4);

user_nn_matrices_cpy_matrices(dest, list);//获取其中一个矩阵

if (dest != NULL){
user_nn_matrices_printf(NULL,"TEST", dest);//打印矩阵
}
else{
printf("null\n");
}
*/
/* 矩阵交换
unsigned char src[] = { 1, 2, 3, 4, 5, 6 };
unsigned char sub[] = { 1, 2, 3 };

user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;

src_matrix = user_nn_matrix_create(2, 3);
sub_matrix = user_nn_matrix_create(1, 3);

user_nn_matrix_memcpy_char(src_matrix, src);
user_nn_matrix_memcpy_char(sub_matrix, sub);

user_nn_matrix_exc_width_height(src_matrix);//交换output_kernel_maps 的 width与height
res_matrix = user_nn_matrix_mult_matrix(src_matrix, sub_matrix);//计算feature vector delta  ######如果不能相乘需要变化横纵大小######
user_nn_matrix_exc_width_height(src_matrix);//交换output_kernel_maps 的 width与height 交换回来

if (res_matrix != NULL){
user_nn_matrix_printf(NULL,res_matrix);//打印矩阵
}
else{
printf("null\n");
}

*/
/* 矩阵与连续矩阵变化
unsigned char src[192] ;
int i = 0;
for (i = 0; i < 192; i++){
src[i] = i;
}
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;
dest = user_nn_matrix_create(1, 192);
user_nn_matrix_memcpy_char(dest, src);
list = user_nn_matrix_to_matrices(dest, 4, 4);//获取其中一个矩阵
dest = user_nn_matrices_to_matrix(list);
if (dest != NULL){
user_nn_matrix_printf(NULL, dest);//打印矩阵
}
else{
printf("null\n");
}

*/
/*矩阵返回最大值测试
user_nn_matrix *matrix = NULL;

matrix = user_nn_matrix_create(6, 6);//创建2*2大小的二维矩阵
user_nn_matrix_rand_vaule(matrix, 1);

int max_value_index = user_nn_matrix_return_max_index(matrix);//
int min_value_index = user_nn_matrix_return_min_index(matrix);//

user_nn_matrix_printf(NULL,matrix);//打印矩阵

printf("\nmax_value_index=%d,vaule=%f\n",max_value_index,*user_nn_matrix_return_max_addr(matrix));
printf("\nmin_value_index=%d,vaule=%f\n",min_value_index,*user_nn_matrix_return_min_addr(matrix));
*/


/* 矩阵转置时间测试

user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
int matrix_w = 128, matrix_h = 128;
src_matrix = user_nn_matrix_create(matrix_w, matrix_h);
sub_matrix = user_nn_matrix_create(matrix_w, matrix_h);
for (int count = 0; count < (src_matrix->width * src_matrix->height); count++) {
src_matrix->data[count] = (float)count * 0.01f;
sub_matrix->data[count] = (float)count * 0.01f;
}
user_nn_matrix_transpose(src_matrix);
user_nn_matrix_transpose(src_matrix);
for (int count = 0; count < (src_matrix->width * src_matrix->height); count++) {
printf("\n%f %f", src_matrix->data[count],sub_matrix->data[count]);
if (src_matrix->data[count] != sub_matrix->data[count]) {
printf(" error");
break;
}
}
printf("end");
getchar();
return 1;

*/
