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