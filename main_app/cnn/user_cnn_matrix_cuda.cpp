
#include "../matrix/user_nn_matrix_cuda.h"

#ifdef WIN64
user_nn_matrix * user_nn_matrix_mult_matrix_cuda(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix){
	cublasHandle_t cuda_handle;
	cublasStatus_t status;
	cudaError_t error;

	user_nn_matrix *result = NULL;//结果矩阵
	int src_count = src_matrix->width * src_matrix->height;
	int sub_count = sub_matrix->width * sub_matrix->height;
	float *src_matrix_cuda = NULL, *sub_matrix_cuda = NULL, *result_cuda = NULL;
	float alpha = 1.0, beta = 0.0;

	if (src_matrix->width != sub_matrix->height){//矩阵乘积只有当第一个矩阵的列数=第二个矩阵的行数才有意义
		return NULL;
	}
	status = cublasCreate_v2(&cuda_handle);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("status error");
	}

	result = user_nn_matrix_create(sub_matrix->width, src_matrix->height);//创建新的矩阵

	error = cudaMalloc((void **)&src_matrix_cuda, src_count * sizeof(float));//分配数据空间
	error = cudaMalloc((void **)&sub_matrix_cuda, sub_count * sizeof(float));//分配数据空间
	error = cudaMalloc((void **)&result_cuda, result->width * result->height * sizeof(float));//分配保存结果的矩阵

	error = cudaMemcpy(src_matrix_cuda, src_matrix->data, src_count * sizeof(float), cudaMemcpyHostToDevice);//拷贝数据
	error = cudaMemcpy(sub_matrix_cuda, sub_matrix->data, sub_count * sizeof(float), cudaMemcpyHostToDevice);//拷贝数据

	//user_nn_matrix_memset(result,1.2f);
	//cudaMemcpy(result_cuda, result->data, result->width * result->height * sizeof(float), cudaMemcpyHostToDevice);//拷贝数据

	//公式：C = alpha*op(A)xop(B)+beta*C
	//句柄、A是否转置、B是否转置、矩阵A长、矩阵A宽（矩阵A长）、矩阵B宽、alpha、A指针、lda、B、ldb、beta、C、ldc
	status = cublasSgemm_v2(cuda_handle, CUBLAS_OP_T, CUBLAS_OP_T, result->height, result->width, src_matrix->width, &alpha, src_matrix_cuda, src_matrix->width, sub_matrix_cuda, sub_matrix->width, &beta, result_cuda, result->height);
	error = cudaThreadSynchronize();

	error = cudaMemcpy(result->data, result_cuda, result->width * result->height * sizeof(float), cudaMemcpyDeviceToHost);//拷贝数据


	cudaFree(src_matrix_cuda);
	cudaFree(sub_matrix_cuda);
	cudaFree(result_cuda);
	cublasDestroy_v2(cuda_handle);

	return result;
}
#endif
