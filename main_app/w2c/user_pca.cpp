

#include "user_pca.h"


//第一步 求解所有样本每一维度的平均值，进行对每个维度减去平均值
//第二步 求特征协方差矩阵
//第三步 求协方差的特征值和特征向量
//第四步 特征值降序排列 选择K个最大值 把K作为列向量组成特征向量矩阵
//第五步 进行矩阵投影得到降低维度后数据

//flag1:x11 x12 x13 ... x1n
//flag2:x21 x22 x23 ... x2n
//flag3:x31 x32 x33 ... x3n
//.
//.
//flagi:xi1 xi2 xi3 ... xin
//
//PCA降维算法
//src_matrix：目标矩阵
//返回 计算后的特征矩阵 依次保存COEFF, LATENT,SCORE 
//参考matlab
user_nn_list_matrix *user_pca_process(user_nn_matrix *src_matrix, float epsilon, eigs_type type) {
	user_nn_list_matrix *result = NULL;//

	user_nn_matrix *result_coeff = NULL;//特征向量
	user_nn_matrix *result_latent = NULL;//特征值对角元素
	user_nn_matrix *result_score = NULL;//特征数据
	
	user_nn_matrix *src_matrix_cov = NULL;//保存协方差矩阵
	user_nn_matrix *src_matrix_mean = NULL;//保存均值矩阵
	user_nn_matrix *src_matrix_mean_ext = NULL;//保存均值矩阵的扩展矩阵
	user_nn_matrix *src_matrix_zero = NULL;//目标矩阵减去目标均值矩阵的结果 归零化
	user_nn_matrix *src_vector = NULL;//特征向量
	user_nn_matrix *src_vector_list = NULL;//对角线特征
	user_nn_list_matrix *src_matrix_qr = NULL;//保存矩阵特征值和特征向量
	
	src_matrix_zero = user_nn_matrix_create(src_matrix->width, src_matrix->height);//创建矩阵
	src_matrix_cov = user_nn_matrix_cov(src_matrix);//求目标矩阵的解协方差矩阵
	src_matrix_mean = user_nn_matrix_mean(src_matrix);//求解目标矩阵的均值
	src_matrix_qr = user_nn_eigs(src_matrix_cov, USER_PCA_EIGS_EPSILON, type);//采用eigs算法 返回特征值和特征向量
	src_matrix_mean_ext = user_nn_matrix_repmat(src_matrix_mean, src_matrix->height, 1);//复制扩展矩阵
	user_nn_matrix_cum_matrix_mult_alpha(src_matrix_zero, src_matrix, src_matrix_mean_ext,-1.0f);//原始矩阵减去平均值矩阵
	src_vector = user_nn_matrix_mult_matrix(src_matrix_zero, src_matrix_qr->matrix->next);//求取降低维度后的特征数据
	src_vector_list = user_nn_matrix_diag(src_matrix_qr->matrix);//获取QR分解后特征值矩阵的对角线元素，拉成一列

	result_latent = user_nn_matrix_cut_vector(src_vector_list, src_vector_list, epsilon);//获取QR分解后特征值矩阵的对角线元素，拉成一列
	result_coeff = user_nn_matrix_cut_vector(src_matrix_qr->matrix->next, src_vector_list, epsilon);//通过对角元素裁剪矩阵
	result_score = user_nn_matrix_cut_vector(src_vector, src_vector_list, epsilon);//通过对角元素裁剪矩阵

	//添加矩阵
	result_coeff->next = result_latent; result_latent->next = result_score;//保存为连续矩阵
	result = (user_nn_list_matrix *)malloc(sizeof(user_nn_list_matrix));//分配空间
	result->width = 3; 
	result->height = 1;
	result->matrix = result_coeff;
	//添加矩阵
	//删除内存
	user_nn_matrix_delete(src_vector);
	user_nn_matrix_delete(src_matrix_cov);
	user_nn_matrix_delete(src_matrix_mean);
	user_nn_matrix_delete(src_matrix_mean_ext);
	user_nn_matrix_delete(src_matrix_zero);
	user_nn_matrices_delete(src_matrix_qr);
	//删除内存
	return result;
}

/*
//srand((unsigned)time(NULL));
	float content[] = {
	0.1,20,3.7,
	3.2,8,6.3,
	0.001,9,1.9 };//矩阵数据
	user_nn_list_matrix *result = NULL;
	user_nn_matrix *matrix = user_nn_matrix_create(10, 10);
	user_nn_matrix_rand_vaule(matrix, 1);
	//user_nn_matrix_memcpy(matrix, content);

	result = user_pca_process(matrix, 1.0f, qr_givens);//自动排序
	user_nn_matrices_printf(NULL, "givens", result);
	user_nn_matrices_delete(result);
	result = user_pca_process(matrix, 1.0f, qr_householder);//无法自动排序
	user_nn_matrices_printf(NULL, "householder", result);
	getchar();
	return 0;
*/