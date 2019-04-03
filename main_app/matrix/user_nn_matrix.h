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
	float bias;//��Ų���
	struct _nn_bias *next;//ָ����һ������ṹ��
}user_nn_bias;
typedef struct _nn_list_biases{
	int biasescount;//�ܲ����ĸ���
	user_nn_bias *bias;//ָ����һ������ṹ��
}user_nn_list_biases;
//��������
typedef struct _nn_matrix{
	int width;//����Ŀ��
	int height;//����ĸ߶�
	//float biases;//ƫ�ò���
	float *data;//��������
	struct _nn_matrix *next;//ָ����һ������ṹ��
}user_nn_matrix;
//������������
typedef struct _nn_list_matrix{
	int width;//��ά����Ŀ��
	int height;//��ά����ĸ߶�
	user_nn_matrix *matrix;
}user_nn_list_matrix;
//��������
typedef enum _sorting_type {
	sorting_up = 0,//����
	sorting_down = 1//����
}sorting_type;
//��ȡQR����ķ�ʽ
typedef enum _eigs_type {
	qr_householder = 0,//
	qr_givens = 1//
}eigs_type;

user_nn_matrix *user_nn_matrix_create(int width, int height);//����һ�����Ϊwidth �߶�Ϊheight�ľ���
user_nn_matrix *user_nn_matrix_cpy_create(user_nn_matrix *dest_matrix);//����һ�����󣬷����µľ���
void user_nn_matrix_transpose(user_nn_matrix *src_matrix);//����ת�� ����һ������ĳ��ȺͿ�ȴ�С���ҽ�������
float *user_nn_matrix_ext_value_index(user_nn_matrix *dest, int post_index);//��ȡ�����е�һ��ֵ��ָ�룬����һά����ķ�ʽ����ȡ
float *user_nn_matrix_ext_value(user_nn_matrix *dest, int postx, int posty);//��ȡ�����е�һ��ֵ��ָ�룬���ն�ά����ķ�ʽ����ȡ

int user_nn_matrix_return_max_index(user_nn_matrix *dest);//���ؾ����е����ֵ�ھ����е�λ��index
float *user_nn_matrix_return_max_addr(user_nn_matrix *dest);//���ؾ����е����ֵ�ĵ�ַ
int user_nn_matrix_return_min_index(user_nn_matrix *dest);//���ؾ����е���Сֵ��λ��index
float *user_nn_matrix_return_min_addr(user_nn_matrix *dest);//���ؾ����е���Сֵ�ĵ�ַ

void user_nn_matrix_delete(user_nn_matrix *dest);//ɾ��һ������ �ͷ��ڴ�
user_nn_list_matrix *user_nn_matrices_create(int total_w, int total_h, int matrix_w, int matrix_h);//����һ�������ľ�����������Ϊ total_w*total_h������ÿ�������СΪmatrix_w*matrix_h
user_nn_list_matrix *user_nn_matrices_create_head(int total_w, int total_h);//����һ����������ͷ���ڲ�����������
void user_nn_matrices_delete(user_nn_list_matrix *src_matrices);//ɾ����������
bool user_nn_matrices_add_matrix(user_nn_list_matrix *list_matrix, user_nn_matrix *end_matirx);//����������ĩβ�����һ������
user_nn_matrix *user_nn_matrices_ext_matrix(user_nn_list_matrix *list_matrix, int postx, int posty);//�����������з���ָ��λ�ã�postx��posty������ָ��
user_nn_matrix *user_nn_matrices_ext_matrix_index(user_nn_list_matrix *list_matrix, int index);//�����������з���ָ��λ��index����ָ��
void user_nn_matrices_to_matrix(user_nn_matrix *src_matrix, user_nn_list_matrix *sub_matrices);//��һ����������ת��Ϊһ���������е�ֵΪ�������������ֵ����ת����ľ���Ŀ��Ϊ1���߶�Ϊ���ݸ����ܺ�
void user_nn_matrix_to_matrices(user_nn_list_matrix *src_matrices, user_nn_matrix *sub_matrix);//�Ѿ���src_matrixת��Ϊsrc_matrices����������
void user_nn_matrices_cpy_matrices(user_nn_list_matrix *src_matrices, user_nn_list_matrix *dest_matrices);//������������dest_matrices����������src_matrices��
bool user_nn_matrix_cpy_array(float *dest_data,user_nn_matrix *src_matrix, int startx, int starty, int width, int height); //����src_matrix����(x,y)����СΪwidth*height��������dest_data,���У�����ʧ�ܻ�ɹ�
bool user_nn_matrix_cpy_array_mult_constant(float *dest_data, user_nn_matrix *src_matrix, int startx, int starty, int width, int height, float constant); //����src_matrix*constant����(x,y)����СΪwidth*height��������dest_data,���У�����ʧ�ܻ�ɹ�
user_nn_matrix *user_nn_matrix_expand_mult_constant(user_nn_matrix *src_matrix, int width, int height, float constant);//�������src_matrix�� ÿ�����ذ��ո�������width��height�� ����֮�����ϵ��bias
user_nn_matrix *user_nn_matrix_expand(user_nn_matrix *src_matrix, int above, int below, int left, int right);//�������src_matrix���������¡����Ҹ�����ָ����С

user_nn_matrix *user_nn_matrix_rotate180(user_nn_matrix *src_matrix);//����src_matrix��ת180��
void user_nn_matrix_poit_mult_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//����src_matrix.*sub_matrix ���������src_matrix����
user_nn_matrix *user_nn_matrix_mult_matrix(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//����src_matrix*sub_matrix ���س˻���ľ���
void user_nn_matrix_mult_constant(user_nn_matrix *src_matrix, float constant);//����src_matrix*v  ���������src_matrix����
void user_nn_matrix_divi_constant(user_nn_matrix *src_matrix, float constant);//����src_matrix/v ���������src_matrix����
void user_nn_matrix_sum_constant(user_nn_matrix *src_matrix, float constant);//����src_matrix��ÿ�����ݼ���constant
void user_nn_matrix_sum_matrix_mult_alpha(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, float alpha);//����save_matrix��;���src_matrix��alpha�ĳ˻�
void user_nn_matrix_memset(user_nn_matrix *save_matrix, float constant);//���þ���src_matrix��ֵ ���������src_matrix����
void user_nn_matrix_memcpy(user_nn_matrix *save_matrix, float *data);//�������������� ��С���ڵ��ھ����С
void user_nn_matrix_memcpy_uchar_mult_constant(user_nn_matrix *save_matrix, unsigned char *input_array, float constant);//�������������� ��С���ڵ��ھ����С
void user_nn_matrix_uchar_memcpy(unsigned char *save_array,user_nn_matrix *src_matrix);//�������������� ��С���ڵ��ھ����С
user_nn_matrix *user_nn_matrix_sorting(user_nn_matrix *src_matrix, sorting_type type);//�Ծ������ݽ�������,����һ���µľ��󣬲���ɾ��ԭ���ľ���

float user_nn_matrix_cum_element(user_nn_matrix *src_matrix);//��;�����������ֵ ����һ��
void user_nn_matrix_cum_matrix(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//����������� ����src_matrix�����sub_matrixÿ��Ԫ�ؽ��мӷ����� ���������src_matrix����
void user_nn_matrix_cum_matrix_alpha(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, float alpha);//����������� ����src_matrix�����sub_matrix*alphaÿ��Ԫ�ؽ��мӷ����� ���������src_matrix����
void user_nn_matrix_cum_matrix_mult_alpha(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix, float alpha);//����������� ����src_matrix�����sub_matrix*alphaÿ��Ԫ�ؽ��мӷ����� ���������src_matrix����
void user_nn_matrix_cpy_matrix(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix);//��������sub_matrix���ݵ�����src_matrix��
void user_nn_matrix_cpy_matrix_p(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix);//ָ�����sub_matrix���ݵ�����src_matrix��
void user_nn_matrix_cpy_matrix_sum_constant(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix, int index, float constant);//��������ָ�������У��ڸ���λ�õط��������constant
float user_nn_matrix_mult_cum_matrix(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//����src_matrix.*sub_matrix Ȼ����;�����������ֵ ������Ҫ������С��ȫһ�£�
user_nn_matrix *user_nn_matrix_ext_matrix(user_nn_matrix *src_matrix, int startx, int starty, int width, int height);//�ھ���src_matrix ����ȡָ����startx,starty��λ��Ϊ���� ��ָ��width��height��С�ľ�������
bool user_nn_matrix_save_array(user_nn_matrix *src_matrix, float *save_data, int startx, int starty, int width, int height); //��src_matrix����ָ��(x, y)λ�ñ���save_datad���ڴ����ݣ�����ʧ�ܻ�ɹ�
bool user_nn_matrix_save_float(user_nn_matrix *src_matrix, int startx, int starty, float vaule);//��ָ��λ�ñ���һ��ֵ������ʧ�ܻ�ɹ�
bool user_nn_matrix_save_matrix(user_nn_matrix *src_matrix, user_nn_matrix *save_matrix, int startx, int starty);//��src_matrix����ָ��(x,y)λ�ñ���save_matrix����
bool user_nn_matrix_sum_array_mult_alpha(user_nn_matrix *dest_matrix, float *src_data, float alpha, int startx, int starty, int width, int height);//��dest_matrix������ָ��λ�õ���src_data*alpha�ľ���
void user_nn_matrix_pooling(user_nn_matrix *save_matrix, user_nn_matrix *src_matrix, user_nn_matrix *kernel_matrix);//�ػ�����  ����sub_matrix���src_matrixÿ�����������ص� �����µĽ������
user_nn_matrix *user_nn_matrix_conv2(user_nn_matrix *src_matrix, user_nn_matrix *kernel_matrix, user_nn_conv2_type type);//�������  ����sub_matrix���src_matrix �����µĽ������

float user_nn_matrix_get_mse(user_nn_matrix *src_matrix);//��ȡ�������
float user_nn_matrix_get_rmse(user_nn_matrix *src_matrix);//��ȡ���������

//matlab����
user_nn_matrix *user_nn_matrix_repmat(user_nn_matrix *dest, int m, int n);//������
void user_nn_matrix_eye(user_nn_matrix *dest);//���þ���Խ���Ϊ1
user_nn_matrix *user_nn_givens(float x, float y);//���givens ��ת���ֵ
user_nn_list_matrix *user_nn_householder_qr(user_nn_matrix *dest);//���A=QR
user_nn_list_matrix *user_nn_givens_qr(user_nn_matrix *dest);//���A=QR
user_nn_list_matrix *user_nn_eigs(user_nn_matrix *dest, float epsilon, eigs_type type);//��������A=QR��ȡ��������ֵ
user_nn_matrix *user_nn_matrix_mean(user_nn_matrix *src_matrix);//�������ƽ��ֵ
user_nn_matrix *user_nn_matrix_cov(user_nn_matrix *src_matrix);//���Э�������
user_nn_matrix *user_nn_tril_indices(int width, int height, float details);//��ȡ��������Ǿ������� �������½��������겻��������
float user_nn_matrix_trace(user_nn_matrix *src_matrix);//��ͶԽ���Ԫ��
user_nn_matrix *user_nn_matrix_diag(user_nn_matrix *src_matrix);//���ضԽ���Ԫ�������µľ���
float user_nn_matrix_norm(user_nn_matrix *src_matrix);//������ķ��� 2D
user_nn_matrix *user_nn_matrix_outer(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);//������������������outerֵ
//
user_nn_matrix *user_nn_matrix_cut_vector(user_nn_matrix *src_matrix, user_nn_matrix *diag_matrix,float epsilon);//ͨ���ԽǾ������������ȡ�µľ���
//

void user_nn_matrix_printf(FILE *debug_file, user_nn_matrix *src_matrix);//��ӡ����
void user_nn_matrices_printf(FILE *debug_file, char *title, user_nn_list_matrix *src_matrix);//��ӡ��������

#endif




/*
//��һ���������ݿ���������һ��������
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(6, 6);//����3*3��С�Ķ�ά����
dest   = user_nn_matrix_create(2, 2);//����2*2��С�Ķ�ά����
user_nn_matrix_rand_vaule(matrix,1);

user_nn_matrix_printf(NULL, matrix);//��ӡ����
bool is_success = user_nn_matrix_save_matrix(matrix, dest, 1, 4);
printf("\n%s\n", is_success==true?"true":"false");

user_nn_matrix_printf(NULL, matrix);//��ӡ����

getchar();
return 0;

*/
/*
//��һ���ڴ����ݿ���������һ�������� ���չ涨����
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(6, 6);//����3*3��С�Ķ�ά����
dest = user_nn_matrix_create(2, 2);//����2*2��С�Ķ�ά����
user_nn_matrix_rand_vaule(matrix, 1);

user_nn_matrix_printf(NULL, matrix);//��ӡ����
bool is_success = user_nn_matrix_save_array(matrix, dest->data, 2, 3, dest->width, dest->height);
printf("\n%s\n", is_success == true ? "true" : "false");

user_nn_matrix_printf(NULL, matrix);//��ӡ����

getchar();
return 0;
*/

/* �ػ�����
user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;
float *result = NULL;


src_matrix = user_nn_matrix_create(28, 28);
sub_matrix = user_nn_matrix_create(2, 2);

user_nn_matrix_memset(src_matrix, 2.5);//���þ���ֵ
user_nn_matrix_memset(sub_matrix, 0.25);//���þ���ֵ

result = user_nn_matrix_ext_value(src_matrix, 27, 27);
*result = 1;

res_matrix = user_nn_matrix_pool(src_matrix, sub_matrix);
if (res_matrix != NULL){
user_nn_matrix_printf(NULL,res_matrix);//��ӡ����
}
else{
printf("null\n");
}
*/
/* ����������
user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;

src_matrix = user_nn_matrix_create(28, 28);
sub_matrix = user_nn_matrix_create(5, 5);

user_nn_matrix_memset(src_matrix, 2.5);//���þ���ֵ
user_nn_matrix_memset(sub_matrix, 1.0);//���þ���ֵ

res_matrix = user_nn_matrix_conv2(src_matrix, sub_matrix, u_nn_conv2_type_valid);
if (res_matrix != NULL){
user_nn_matrix_printf(NULL,res_matrix);//��ӡ����
}
else{
printf("null\n");
}
*/
/* ��ά���� ��ȡ
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;

list = user_nn_matrices_create(2, 2, 1, 1);//����2*2��1*1��С�Ķ�ά����
user_nn_matrix_memset(list->matrix, 1);
user_nn_matrix_memset(list->matrix->next, 2);
user_nn_matrix_memset(list->matrix->next->next, 3);
user_nn_matrix_memset(list->matrix->next->next->next, 4);

dest = user_nn_matrices_ext_matrix(list, 1, 0);//��ȡ����һ������

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//��ӡ����
}
else{
printf("null\n");
}
*/
/* �����ȡ����
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(2, 2);//����2*2��С�Ķ�ά����
matrix->data[0] = 1.0;
matrix->data[1] = 2.0;
matrix->data[2] = 3.0;
matrix->data[3] = 4.0;

dest = user_nn_matrix_ext_matrix(matrix, 1, 1, 1, 1);//��ȡ����

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//��ӡ����
}
else{
printf("null\n");
}
*/


/*��ȡ�����е�һ��ֵ
user_nn_matrix *matrix = NULL;
float *p;

matrix = user_nn_matrix_create(2, 2);//����2*2��С�Ķ�ά����
matrix->data[0] = 1.0;
matrix->data[1] = 2.0;
matrix->data[2] = 3.0;
matrix->data[3] = 4.0;

p = user_nn_matrix_ext_value(matrix, 1, 1);//��ȡ����ֵ
if (p != NULL)
printf("%f \n", *p);
*/
/*ƫ�ò�������
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
/* ����˷�
user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;

src_matrix = user_nn_matrix_create(1280, 1280);
sub_matrix = user_nn_matrix_create(1280, 1280);

for (int count = 0; count < (src_matrix->width * src_matrix->height); count++) {
src_matrix->data[count] = (float)count * 0.01f;
sub_matrix->data[count] = (float)count * 0.01f;
}
res_matrix = user_nn_matrix_mult_matrix(src_matrix, sub_matrix);//�������
if (res_matrix != NULL) {
user_nn_matrix_printf(NULL, res_matrix);//��ӡ����
}
else {
printf("null\n");
}
printf("\nend");
_getch();
return 1;
*/
/*��������ת��Ϊ �߶�Ϊ1�������������
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;

dest = user_nn_matrix_create(2,2);
dest->data[0] = 1.0;
dest->data[1] = 2.0;
dest->data[2] = 3.0;
dest->data[3] = 4.0;

list = user_nn_matrix_to_matrices(dest,1,1);//��ȡ����һ������

if (list != NULL){
user_nn_matrix_printf(NULL,list->matrix);//��ӡ����
user_nn_matrix_printf(NULL,list->matrix->next);//��ӡ����
user_nn_matrix_printf(NULL,list->matrix->next->next);//��ӡ����
user_nn_matrix_printf(NULL,list->matrix->next->next->next);//��ӡ����
}
else{
printf("null\n");
}
*/
/*�������ת��Ϊ�߶�Ϊ1�ĵ�������
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;

list = user_nn_matrices_create(5, 2, 1, 1);//����2*2��1*1��С�Ķ�ά����
user_nn_matrix_memset(list->matrix, 1);
user_nn_matrix_memset(list->matrix->next, 2);
user_nn_matrix_memset(list->matrix->next->next, 3);
user_nn_matrix_memset(list->matrix->next->next->next, 4);

dest = user_nn_matrices_to_matrix(list);//��ȡ����һ������

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//��ӡ����
}
else{
printf("null\n");
}
*/
/*����ƫ�ò���ת�������
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
user_nn_matrix_printf(NULL,dest);//��ӡ����
}
else{
printf("null\n");
}
*/
/*�����ֵ����
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(2, 2);//����2*2��С�Ķ�ά����
matrix->data[0] = 1.0;
matrix->data[1] = 2.0;
matrix->data[2] = 3.0;
matrix->data[3] = 4.0;

dest = user_nn_matrix_expand_matrix(matrix, 2, 3);//

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//��ӡ����
}
else{
printf("null\n");
}
*/
/*�������� �߿�����
user_nn_matrix *matrix = NULL;
user_nn_matrix *dest = NULL;

matrix = user_nn_matrix_create(1, 1);//����2*2��С�Ķ�ά����
matrix->data[0] = 1.0;
//matrix->data[1] = 2.0;
//matrix->data[2] = 3.0;
//matrix->data[3] = 4.0;

dest = user_nn_matrix_expand(matrix, 1, 1);//

if (dest != NULL){
user_nn_matrix_printf(NULL,dest);//��ӡ����
}
else{
printf("null\n");
}
*/
/*�������󿽱�
user_nn_list_matrix *list = NULL;
user_nn_list_matrix *dest = NULL;

list = user_nn_matrices_create(5, 2, 1, 1);//����2*2��1*1��С�Ķ�ά����
dest = user_nn_matrices_create(5, 2, 1, 1);//����2*2��1*1��С�Ķ�ά����

user_nn_matrix_memset(list->matrix, 1);
user_nn_matrix_memset(list->matrix->next, 2);
user_nn_matrix_memset(list->matrix->next->next, 3);
user_nn_matrix_memset(list->matrix->next->next->next, 4);

user_nn_matrices_cpy_matrices(dest, list);//��ȡ����һ������

if (dest != NULL){
user_nn_matrices_printf(NULL,"TEST", dest);//��ӡ����
}
else{
printf("null\n");
}
*/
/* ���󽻻�
unsigned char src[] = { 1, 2, 3, 4, 5, 6 };
unsigned char sub[] = { 1, 2, 3 };

user_nn_matrix *src_matrix = NULL;
user_nn_matrix *sub_matrix = NULL;
user_nn_matrix *res_matrix = NULL;

src_matrix = user_nn_matrix_create(2, 3);
sub_matrix = user_nn_matrix_create(1, 3);

user_nn_matrix_memcpy_char(src_matrix, src);
user_nn_matrix_memcpy_char(sub_matrix, sub);

user_nn_matrix_exc_width_height(src_matrix);//����output_kernel_maps �� width��height
res_matrix = user_nn_matrix_mult_matrix(src_matrix, sub_matrix);//����feature vector delta  ######������������Ҫ�仯���ݴ�С######
user_nn_matrix_exc_width_height(src_matrix);//����output_kernel_maps �� width��height ��������

if (res_matrix != NULL){
user_nn_matrix_printf(NULL,res_matrix);//��ӡ����
}
else{
printf("null\n");
}

*/
/* ��������������仯
unsigned char src[192] ;
int i = 0;
for (i = 0; i < 192; i++){
src[i] = i;
}
user_nn_list_matrix *list = NULL;
user_nn_matrix *dest = NULL;
dest = user_nn_matrix_create(1, 192);
user_nn_matrix_memcpy_char(dest, src);
list = user_nn_matrix_to_matrices(dest, 4, 4);//��ȡ����һ������
dest = user_nn_matrices_to_matrix(list);
if (dest != NULL){
user_nn_matrix_printf(NULL, dest);//��ӡ����
}
else{
printf("null\n");
}

*/
/*���󷵻����ֵ����
user_nn_matrix *matrix = NULL;

matrix = user_nn_matrix_create(6, 6);//����2*2��С�Ķ�ά����
user_nn_matrix_rand_vaule(matrix, 1);

int max_value_index = user_nn_matrix_return_max_index(matrix);//
int min_value_index = user_nn_matrix_return_min_index(matrix);//

user_nn_matrix_printf(NULL,matrix);//��ӡ����

printf("\nmax_value_index=%d,vaule=%f\n",max_value_index,*user_nn_matrix_return_max_addr(matrix));
printf("\nmin_value_index=%d,vaule=%f\n",min_value_index,*user_nn_matrix_return_min_addr(matrix));
*/


/* ����ת��ʱ�����

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
