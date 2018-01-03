
#include "user_nn_matrix_cuda.h"
#include "user_nn_matrix.h"

//__global__ 主机调用
//__device__ 设备调用
//transpose_naive << < 1, matrix_h >> >(sub_matrix, src_matrix, matrix_w, matrix_h);
//extern "C" template <int BLOCK_SIZE>
//extern "C"
__global__ void transposeNaive_array(float *odata, float* idata, int width, int height)
{
	//threadId线程的索引 blockDim线程块的维度 gridDim线程格的维度
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;//grid划分成1维，block划分为1维 获取基坐标
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;//
	unsigned int index_in = xIndex + width * yIndex;//设置线程数量 线性地址
	unsigned int index_out = yIndex + height * xIndex;

	unsigned int width_index = 0;
	for (width_index = 0; width_index < height; width_index++){
		odata[index_out + width_index] = idata[index_in + width_index * width];//拷贝一列数据至一行的位置
	}
	/*
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
		unsigned int index_in = xIndex + width * yIndex;
		unsigned int index_out = yIndex + height * xIndex;
		odata[index_out] = idata[index_in];
	}
	*/
}

__global__ void transposeNaive_block(float *odata, float *idata, int width, int height)
{
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int index_in = xIndex + width * yIndex;
	int index_out = yIndex + height * xIndex;

	for (int i = 0; i<TILE_DIM; i += BLOCK_ROWS)
	{
		odata[index_out + i] = idata[index_in + i*width];
	}
}

//矩阵转置
extern "C"
void user_nn_matrix_transpose_cuda(user_nn_matrix *src_matrix){
	int src_count = src_matrix->width * src_matrix->height;
	float *cuda_input_matrix = NULL;
	float *cuda_output_matrix = NULL;

	cudaMalloc((void **)&cuda_input_matrix, src_count * sizeof(float));//分配数据空间
	cudaMalloc((void **)&cuda_output_matrix, src_count * sizeof(float));//分配数据空间

	cudaMemcpy(cuda_input_matrix, src_matrix->data, src_count * sizeof(float), cudaMemcpyHostToDevice);//拷贝数据
	//<<<grid, threads>>> grid表示线程格有多少多少个线程块，threads表示一个线程块里有多少个线程
	if ((src_matrix->width%TILE_DIM == 0) && (src_matrix->height%TILE_DIM == 0) && (src_matrix->width == src_matrix->height)){
		dim3 grid(src_matrix->width / TILE_DIM, src_matrix->height / TILE_DIM), threads(TILE_DIM, BLOCK_ROWS);//动态计算所需要的栅格和线程
		transposeNaive_block << <grid, threads >> >(cuda_output_matrix, cuda_input_matrix, src_matrix->width, src_matrix->height);//进行矩阵转置
	}
	else{
		transposeNaive_array << <1, ((src_matrix->width > src_matrix->height) ? src_matrix->width : src_matrix->height) >> >(cuda_output_matrix, cuda_input_matrix, src_matrix->width, src_matrix->height);//进行矩阵转置
	}
	cudaMemcpy(src_matrix->data, cuda_output_matrix, src_count * sizeof(float), cudaMemcpyDeviceToHost);//拷贝数据
	//交换矩阵位置
	src_matrix->width  = src_matrix->width ^ src_matrix->height;
	src_matrix->height = src_matrix->width ^ src_matrix->height;
	src_matrix->width  = src_matrix->width ^ src_matrix->height;

	cudaFree(cuda_input_matrix);
	cudaFree(cuda_output_matrix);
}
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