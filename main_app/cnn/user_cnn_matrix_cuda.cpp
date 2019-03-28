
#ifdef WIN64
user_nn_matrix * user_nn_matrix_mult_matrix_cuda(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix){
	cublasHandle_t cuda_handle;
	cublasStatus_t status;
	cudaError_t error;

	user_nn_matrix *result = NULL;//�������
	int src_count = src_matrix->width * src_matrix->height;
	int sub_count = sub_matrix->width * sub_matrix->height;
	float *src_matrix_cuda = NULL, *sub_matrix_cuda = NULL, *result_cuda = NULL;
	float alpha = 1.0, beta = 0.0;

	if (src_matrix->width != sub_matrix->height){//����˻�ֻ�е���һ�����������=�ڶ��������������������
		return NULL;
	}
	status = cublasCreate_v2(&cuda_handle);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("status error");
	}

	result = user_nn_matrix_create(sub_matrix->width, src_matrix->height);//�����µľ���

	error = cudaMalloc((void **)&src_matrix_cuda, src_count * sizeof(float));//�������ݿռ�
	error = cudaMalloc((void **)&sub_matrix_cuda, sub_count * sizeof(float));//�������ݿռ�
	error = cudaMalloc((void **)&result_cuda, result->width * result->height * sizeof(float));//���䱣�����ľ���

	error = cudaMemcpy(src_matrix_cuda, src_matrix->data, src_count * sizeof(float), cudaMemcpyHostToDevice);//��������
	error = cudaMemcpy(sub_matrix_cuda, sub_matrix->data, sub_count * sizeof(float), cudaMemcpyHostToDevice);//��������

	//user_nn_matrix_memset(result,1.2f);
	//cudaMemcpy(result_cuda, result->data, result->width * result->height * sizeof(float), cudaMemcpyHostToDevice);//��������

	//��ʽ��C = alpha*op(A)xop(B)+beta*C
	//�����A�Ƿ�ת�á�B�Ƿ�ת�á�����A��������A������A����������B��alpha��Aָ�롢lda��B��ldb��beta��C��ldc
	status = cublasSgemm_v2(cuda_handle, CUBLAS_OP_T, CUBLAS_OP_T, result->height, result->width, src_matrix->width, &alpha, src_matrix_cuda, src_matrix->width, sub_matrix_cuda, sub_matrix->width, &beta, result_cuda, result->height);
	error = cudaThreadSynchronize();

	error = cudaMemcpy(result->data, result_cuda, result->width * result->height * sizeof(float), cudaMemcpyDeviceToHost);//��������


	cudaFree(src_matrix_cuda);
	cudaFree(sub_matrix_cuda);
	cudaFree(result_cuda);
	cublasDestroy_v2(cuda_handle);

	return result;
}
#endif
