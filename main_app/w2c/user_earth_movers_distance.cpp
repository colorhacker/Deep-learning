#include "user_earth_movers_distance.h"


//升序排列定义函数
int user_emd_float_sort(const void *a, const void *b){
	return (*(float*)b < *(float*)a)?1:-1;  
}

//获取总费用
float user_emd_matrix_get_cost_value(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix) {
	int total = src_matrix->width * src_matrix->height;
	float result = 0.0f;
	float *src_data = src_matrix->data;
	float *sub_data = sub_matrix->data;

	while (total--) {
		if (*sub_data != 0) {
			result += *src_data * *sub_data;
		}
		sub_data++;
		src_data++;
	}
	return result;
}
//设置矩阵的行和列的固定值
void user_emd_matrix_set_row_col(user_nn_matrix *save_matrix, float value) {
	int index = 0;
	float *save_data = save_matrix->data;
	for (index=0; index<save_matrix->width; index++) {
		*save_data++ = value;
	}
	for (index = 0; index < save_matrix->height - 1; index++) {
		*save_data = value;
		save_data += save_matrix->width;
	}
}
//通过设置是否为0拷贝矩阵
void user_emd_matrix_unzero_mapping_cpy(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix) {
	int post_y = 0, post_x = 0;
	for (post_y = 1; post_y < save_matrix->height; post_y++) {
		for (post_x = 1; post_x < save_matrix->width; post_x++) {
			if (*user_nn_matrix_ext_value(save_matrix, post_x, post_y) != 0) {
				*user_nn_matrix_ext_value(save_matrix, post_x, post_y) = *user_nn_matrix_ext_value(sub_matrix, post_x, post_y);
			}
		}
	}
}
//通过设置是否为0拷贝矩阵
void user_emd_matrix_zero_mapping_cpy(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix) {
	int post_y = 0, post_x = 0;
	for (post_y = 1; post_y < save_matrix->height; post_y++) {
		for (post_x = 1; post_x < save_matrix->width; post_x++) {
			if (*user_nn_matrix_ext_value(save_matrix, post_x, post_y) == 0) {
				*user_nn_matrix_ext_value(save_matrix, post_x, post_y) = *user_nn_matrix_ext_value(sub_matrix, post_x, post_y);
			}
			else {
				*user_nn_matrix_ext_value(save_matrix, post_x, post_y) = 0;
			}
		}
	}
}

//创建一个emd结构矩阵
//dist_array：距离值
//width：需求地数据
//height：制造数据
//w_size：需求个数
//h_size：目的个数
//返回：新的矩阵
user_nn_matrix *user_emd_object_create(float *dist_array,float *height_array,int height,float *width_array,int width){
	user_nn_matrix *result = NULL;
	float total_input = 0.0f;
	float total_output=0.0f;
	int index = 0;

	//校验输入和输出权重需要一致
	for(index=0;index < height;index++){
		total_input += *height_array++;
	}
	height_array -= height;
	for(index=0;index < width;index++){
		total_output += *width_array++;
	}
	if(total_input != total_output){
		return NULL;
	}
	width_array -= width;
	for(index = 0;index<(height*width);index++){
		if(*dist_array++ < 0.0f){
			return NULL;
		}
	}
	dist_array -= height*width;
	
	result = user_nn_matrix_create(width + 1,height + 1);//创建新的数据矩阵
	user_nn_matrix_save_array(result, dist_array, 1, 1, width, height);//加载距离数据
	user_nn_matrix_save_array(result, height_array, 0, 1, 1, height);//加载输入权重
	user_nn_matrix_save_array(result, width_array, 1, 0, width, 1);//加载输出权重

	return result;
}

/*   
		o1	o2 o3 o4 
	00	w1 w2 w3 w4 00
i1	w1	a1 a2 a3 a4 ta 
i2	w2	b1 b2 b3 b4 tb
i3	w3	c1 c2 c3 c4 tc
	00	t1 t2 t3 t4 00
	ix：表示货物仓库
	ox：表示需求仓库
	wx：表示货物数量
	ax：表示运输成本或者距离
	tx：用于保存行列的最小值与次小值的差
*/
//通过矩阵计算一次处理一次emd距离
//src_matrix：输入矩阵
//返回：距离或者 float的最大值 返回最大值表示无效或者距离以及处理完成
//采用 volge算法 VAM法
//算法实现：
//阶段一获取最大的行或者列 并且提取行列的位置
//阶段二处理所在行传输最短数据
float user_emd_vogel_plan_init_martix(user_nn_matrix *src_matrix, user_nn_matrix *tra_matrix){
	user_nn_matrix *row_matrix = user_nn_matrix_create(src_matrix->width - 1,1);//保存行数据的矩阵
	user_nn_matrix *col_matrix = user_nn_matrix_create(1,src_matrix->height - 1);//保存行数据的矩阵
	float result = FLT_MAX, transfer_value = 0.0f;
	bool max_vaule_is_row = true;//保存是否是行或者列
	int max_vaule_index = 1;//初始化为1 第0行与0列保存的是权重
	float max_vaule = -FLT_MAX;//用于保存值
	int index = 0;

	float constant_value = 0.0f;//不变值
	float *variable_value = NULL;//列表值
	float *distance_vaule = NULL;//距离值
	int min_vaule_index = 0;

	//第一阶段找出需要处理的行
	for(index=0;index<col_matrix->height;index++){
		if(*user_nn_matrix_ext_value(src_matrix,0,index + 1) != 0.0f){//判断权重是否是0 如果是那么跳过这一行
			if(user_nn_matrix_cpy_array(row_matrix->data,src_matrix, 1,index + 1,row_matrix->width,1)){//拷贝一行距离数据 成功进行下一步处理
				qsort(row_matrix->data,row_matrix->width*row_matrix->height,sizeof(row_matrix->data),user_emd_float_sort);//升序排列 距离数据
				//两种情况，一种最小与次最小都有效 另外只有最小有效次最小无效
				if((row_matrix->data[0] != FLT_MAX) && (row_matrix->data[1] != FLT_MAX)){
					//user_nn_matrix_save_float(src_matrix,src_matrix->width - 2 + 1,index + 1,min_vaule);//设置差值 如果需要打开此函数那么需要设置矩阵宽度加2 第一行user_nn_matrix_create(src_matrix->width - 1,1)设置为user_nn_matrix_create(src_matrix->width - 2,1) 并且输入矩阵也需要扩充
					//row_matrix->data[1] - row_matrix->data[0];//取得差值
					if(max_vaule < (row_matrix->data[1] - row_matrix->data[0])){
						max_vaule = (row_matrix->data[1] - row_matrix->data[0]);
						max_vaule_is_row = true;
						max_vaule_index = index + 1;
						//printf("\nmin_vaule %d,%f\n",index+1,min_vaule);
					}
				}else if((row_matrix->data[0] != FLT_MAX) && (row_matrix->data[1] == FLT_MAX)){
					max_vaule = row_matrix->data[0];
					max_vaule_is_row = true;
					max_vaule_index = index + 1;
					//user_nn_matrix_save_float(src_matrix,src_matrix->width - 2 + 1,index + 1,0);//设置为0
				}
			}
		}
	}
	for(index=0;index<row_matrix->width;index++){
		if(*user_nn_matrix_ext_value(src_matrix,index + 1,0) != 0.0f){	//判断权重是否是0 如果是那么这一列无用
			if(user_nn_matrix_cpy_array(col_matrix->data,src_matrix,index + 1,1,1,col_matrix->height)){
				qsort(col_matrix->data,col_matrix->width*col_matrix->height,sizeof(col_matrix->data),user_emd_float_sort);//升序排列
				//两种情况，一种最小与次最小都有效 另外只有最小有效次最小无效
				if((col_matrix->data[0] != FLT_MAX) && (col_matrix->data[1] != FLT_MAX)){
					//user_nn_matrix_save_float(src_matrix,index + 1,src_matrix->height - 2 + 1,min_vaule);//设置值	
					//min_vaule = col_matrix->data[1] - col_matrix->data[0];//取得差值
					if(max_vaule < (col_matrix->data[1] - col_matrix->data[0])){
						max_vaule = col_matrix->data[1] - col_matrix->data[0];
						max_vaule_is_row = false;
						max_vaule_index = index + 1;
						//printf("\nmin_vaule %d,%f\n",index+1,min_vaule);
					}
				}else if((col_matrix->data[0] != FLT_MAX) && (col_matrix->data[1] == FLT_MAX)){
					max_vaule = col_matrix->data[0];
					max_vaule_is_row = false;
					max_vaule_index = index + 1;
					//user_nn_matrix_save_float(src_matrix,src_matrix->width - 2 + 1,index + 1,0);//设置为0
				}
			}
		}
	}
	//printf("\n%s,index=%d,max value=%f\n",max_vaule_is_row==true?"row":"col",max_vaule_index,max_vaule);
	//第二阶段 处理数据对象
	if(max_vaule_is_row){
		user_nn_matrix_cpy_array(row_matrix->data,src_matrix,1,0,row_matrix->width,1);//提取行需求量数据
		constant_value = *user_nn_matrix_ext_value(src_matrix,0,max_vaule_index);//取得输入权重的唯一不变值
		variable_value = user_nn_matrix_ext_value(src_matrix,1,0);//取得输出权重的起始位置
		distance_vaule = user_nn_matrix_ext_value(src_matrix,1,max_vaule_index);
		for(index=0;index < (row_matrix->height * row_matrix->width);index++){
			//printf("\nrow constant_value=%f,*variable_value=%f，*distance_vaule=%f\n",constant_value,*variable_value,*distance_vaule);
			if((*distance_vaule !=FLT_MAX) && (constant_value != 0) && (*variable_value != 0)){			
				if(*variable_value > constant_value){
					if(result > (*distance_vaule * constant_value)){
						result = *distance_vaule * constant_value;
						min_vaule_index = index;
					}	
				}else{
					if(result > (*distance_vaule * (*variable_value))){
						result = *distance_vaule * (*variable_value);
						min_vaule_index = index;
					}	
				}
			}
			distance_vaule++;
			variable_value++;
		}
		
		if ((constant_value != 0) && (row_matrix->data[min_vaule_index] != 0)){//行列不是零才能进行计算
			if (constant_value > row_matrix->data[min_vaule_index]) {
				//如果起点值大于需求值 删除一列
				transfer_value = row_matrix->data[min_vaule_index];////获取转移值
				user_nn_matrix_save_float(src_matrix, 0, max_vaule_index, constant_value - row_matrix->data[min_vaule_index]);//修改输入值
				user_nn_matrix_save_float(src_matrix, min_vaule_index + 1, 0, 0);//设置值

				user_nn_matrix_memset(col_matrix, FLT_MAX);//设置行矩阵为最大值
				user_nn_matrix_save_matrix(src_matrix, col_matrix, min_vaule_index + 1, 1);//设置最大值清除行不在处理
			}
			else {
				//如果起点值小于需求值 删除行
				transfer_value = constant_value;////获取转移值
				user_nn_matrix_save_float(src_matrix, 0, max_vaule_index, 0);//设置值
				user_nn_matrix_save_float(src_matrix, min_vaule_index + 1, 0, row_matrix->data[min_vaule_index] - constant_value);//设置值

				user_nn_matrix_memset(row_matrix, FLT_MAX);//设置行矩阵为最大值
				user_nn_matrix_save_matrix(src_matrix, row_matrix, 1, max_vaule_index);//设置最大值清除行不在处理
			}
			*user_nn_matrix_ext_value(tra_matrix, min_vaule_index + 1, max_vaule_index) = transfer_value;
		}
		//printf("change: x:%d y：%d v:%f\n",  min_vaule_index + 1, max_vaule_index, transfer_value);
		//user_nn_matrix_printf(NULL, tra_matrix);//
		//result = result;
	}else{
		user_nn_matrix_cpy_array(col_matrix->data,src_matrix,0,1,1,col_matrix->height);//提取列需求量数据
		constant_value = *user_nn_matrix_ext_value(src_matrix,max_vaule_index,0);//取得输入权重的唯一不变值
		variable_value = user_nn_matrix_ext_value(src_matrix,0,1);//取得输出权重的起始位置
		distance_vaule = user_nn_matrix_ext_value(src_matrix,max_vaule_index,1);
		for(index=0;index < (col_matrix->height * col_matrix->width);index++){
			//printf("\ncol constant_value=%f,*variable_value=%f，*distance_vaule=%f\n",constant_value,*variable_value,*distance_vaule);
			if((*distance_vaule != FLT_MAX) && (constant_value != 0) && (*variable_value != 0)){		
				if(*variable_value > constant_value){
					if(result > (*distance_vaule * constant_value)){
						result = *distance_vaule * constant_value;
						min_vaule_index = index;
					}
				}else{
					if(result > (*distance_vaule * (*variable_value))){
						result = *distance_vaule * (*variable_value);
						min_vaule_index = index;
					}
				}	
			}
			distance_vaule += src_matrix->width;//列需要移动width的距离
			variable_value += src_matrix->width;//列权重同样需要移动width的距离
		}
		if ((constant_value != 0) && (row_matrix->data[min_vaule_index] != 0)){//行列不是零才能进行计算
			if (col_matrix->data[min_vaule_index] > constant_value) {
				//如果输入值 大于需求值 清除 列
				transfer_value = constant_value;//获取转移值
				user_nn_matrix_save_float(src_matrix, 0, min_vaule_index + 1, col_matrix->data[min_vaule_index] - constant_value);//设置值
				user_nn_matrix_save_float(src_matrix, max_vaule_index, 0, 0);//设置值

				user_nn_matrix_memset(col_matrix, FLT_MAX);//设置行矩阵为最大值
				user_nn_matrix_save_matrix(src_matrix, col_matrix, max_vaule_index, 1);//设置最大值清除行不在处理
			}
			else {
				//如果输入值 小于需求值 清除 行
				transfer_value = col_matrix->data[min_vaule_index];//获取转移值
				user_nn_matrix_save_float(src_matrix, 0, min_vaule_index + 1, 0);//设置值
				user_nn_matrix_save_float(src_matrix, max_vaule_index, 0, constant_value - col_matrix->data[min_vaule_index]);//设置值

				user_nn_matrix_memset(row_matrix, FLT_MAX);//设置行矩阵为最大值
				user_nn_matrix_save_matrix(src_matrix, row_matrix, 1, min_vaule_index + 1);//设置最大值清除行不在处理
			}
			*user_nn_matrix_ext_value(tra_matrix, max_vaule_index, min_vaule_index + 1) = transfer_value;
		}
		//printf("change: x:%d y：%d v:%f\n", max_vaule_index, min_vaule_index + 1, transfer_value);
		//user_nn_matrix_printf(NULL, tra_matrix);//
		//result = result;
	}
	//
	user_nn_matrix_delete(row_matrix);//删除矩阵
	user_nn_matrix_delete(col_matrix);//删除矩阵

	//user_nn_matrix_printf(NULL, src_matrix);
	
	return  result;
}

//采用威格尔算法获取初始矩阵 
user_nn_matrix *user_emd_get_vogel_init_martix(user_nn_matrix *src_matrix){
	user_nn_matrix *temp_matrix = NULL,*result_matirx = NULL;

	temp_matrix = user_nn_matrix_cpy_create(src_matrix);//复制矩阵
	result_matirx = user_nn_matrix_create(src_matrix->width, src_matrix->height);//转移量的矩阵
	//float socre = 0;
	if (temp_matrix != NULL) {
		while (user_emd_vogel_plan_init_martix(temp_matrix, result_matirx) < FLT_MAX) {
		}
	}
//	printf("\n%f\n", socre);
	//user_nn_matrix_printf(NULL, result_matirx);//

	user_nn_matrix_delete(temp_matrix);//删除矩阵
	return result_matirx;
}
//采用位势差（对偶变量方法）方法解决对偶变量
user_nn_matrix *user_emd_potential_plan_matrix(user_nn_matrix *src_matrix) {
	user_nn_matrix *result_matirx = NULL;
	int post_x = 0, post_y = 0;
	float *variable_value = NULL, *variable_row = NULL, *variable_col = NULL;

	result_matirx = user_nn_matrix_cpy_create(src_matrix);//复制矩阵
	//printf("\n被求检验数矩阵\n");
	//user_nn_matrix_printf(NULL, src_matrix);//
	while (*user_nn_matrix_ext_value(result_matirx, 0, 0) == 0) {
		*user_nn_matrix_ext_value(result_matirx, 0, 0) = 1;//设置循环一次
//		user_nn_matrix_printf(NULL, result_matirx);//
		for (post_y = 1; post_y < result_matirx->height; post_y++) {
			for (post_x = 1; post_x < result_matirx->width; post_x++) {//循环宽度值
				variable_value = user_nn_matrix_ext_value(result_matirx, post_x, post_y);
				variable_row = user_nn_matrix_ext_value(result_matirx, post_x, 0);
				variable_col = user_nn_matrix_ext_value(result_matirx, 0, post_y);
				if (*variable_value != 0) {
					//如果不是第一行，且行或列值不是0，那么进行计算
					if (post_y == 1) {//设置列首个值为0
						if (*variable_row == 0) {
							*variable_row = *variable_value;
						}
					}
					else {
						//printf("\n%f\n", fabs(*variable_value - *variable_row - *variable_col));
						//if ((*variable_row + *variable_col) != *variable_value) {
						if (fabs(*variable_value - *variable_row - *variable_col) > emd_potential_accuracy) {
							if ((*variable_row == 0) && (*variable_col != 0)) {
								*variable_row = *variable_value - *variable_col;
							}
							else if ((*variable_row != 0) && (*variable_col == 0)) {
								*variable_col = *variable_value - *variable_row;
							}
							else {
								//printf("\n%f %f\n", *variable_row + *variable_col, *variable_value);
								*user_nn_matrix_ext_value(result_matirx, 0, 0) = 0;//设置为数据未完成
							}
						}
					}
				}				
			}
		}
	}
	result_matirx->data[0] = 0;//设置为0
	return result_matirx;
}
//检验对偶数据是否正确
user_nn_matrix *user_emd_censor_potential_matrix(user_nn_matrix *src_matrix) {
	user_nn_matrix *result_matirx = NULL;
	int post_x = 0, post_y = 0;
	float *variable_value = NULL, *variable_row = NULL, *variable_col = NULL;

	result_matirx = user_nn_matrix_cpy_create(src_matrix);//复制矩阵
	while (*user_nn_matrix_ext_value(result_matirx, 0, 0) == 0) {
		*user_nn_matrix_ext_value(result_matirx, 0, 0) = 1;//设置循环一次
		for (post_y = 1; post_y < result_matirx->height; post_y++) {
			for (post_x = 1; post_x < result_matirx->width; post_x++) {//循环宽度值
				variable_value = user_nn_matrix_ext_value(result_matirx, post_x, post_y);
				variable_row = user_nn_matrix_ext_value(result_matirx, post_x, 0);
				variable_col = user_nn_matrix_ext_value(result_matirx, 0, post_y);
				if (*variable_value != 0) {
					*variable_value = *variable_value - *variable_row - *variable_col;
				}
			}
		}
	}
	result_matirx->data[0] = 0;//设置为0
	return result_matirx;
}
//调整值
void user_emd_adjust_matrix_value(user_nn_matrix *src_matrix, user_nn_matrix *path_matix) {
	float *path_matix_data = path_matix->data;
	int count = 0;
	float min_value = FLT_MAX;

	path_matix_data = path_matix->data;
	int start_point = 1;//设置起点
	float dest_point = 0.0f;
	bool is_row = false;
	if (fabs(path_matix_data[0] - path_matix_data[1]) < path_matix->width) {//初始化行列
		is_row = true;
	}
	//user_nn_matrix_printf(NULL, path_matix);//
	//删除路径上非顶点的点
	for (; start_point < (path_matix->height * path_matix->width - 1); start_point++) {
		if (path_matix_data[start_point + 1] != 0) {
			dest_point = fabs(path_matix_data[start_point] - path_matix_data[start_point + 1]);//计算差值
		}else {
			dest_point = fabs(path_matix_data[start_point] - path_matix_data[0]);//计算差值
		}
		if (dest_point < path_matix->width) {//属于行
			if (is_row == true) {
				*user_nn_matrix_ext_value_index(path_matix, start_point) = 0;//
			}
			else {
				is_row = true;
			}
		}else {
			if (is_row == false) {
				*user_nn_matrix_ext_value_index(path_matix, start_point) = 0;//
			}
			else {
				is_row = false;
			}
		}
	}
	//获取所有点里面的最小值
	path_matix_data = path_matix->data;
	for (start_point = 0; start_point < path_matix->height * path_matix->width; start_point++) {
		//printf("\path_matix_data：%f\n", *path_matix_data);
		if (*path_matix_data != 0) {
			if ((count++ % 2) == 1) {
				//*user_nn_matrix_ext_value_index(src_matrix, (int)*path_matix_data) -= value;
				if ((*user_nn_matrix_ext_value_index(src_matrix, (int)*path_matix_data) < min_value)&&
					*user_nn_matrix_ext_value_index(src_matrix, (int)*path_matix_data) != 0) {
					min_value = *user_nn_matrix_ext_value_index(src_matrix, (int)*path_matix_data);//
				}
			}
		}
		path_matix_data++;
	}
	//printf("\nmin_value：%f\n", min_value);
	//user_nn_matrix_printf(NULL, path_matix);//
	//修改目标值
	count = 0;
	path_matix_data = path_matix->data;
	for (start_point = 0; start_point < path_matix->height * path_matix->width; start_point++) {
		if (*path_matix_data != 0) {
			if ((count++ % 2) == 1) {//奇数
				*user_nn_matrix_ext_value_index(src_matrix, (int)*path_matix_data) -= min_value;
			}
			else {//偶数
				*user_nn_matrix_ext_value_index(src_matrix, (int)*path_matix_data) += min_value;
			}
		}
		path_matix_data++;
	}
	//******* 重要 有必要添加 
	//如果回路的偶数项顶点中同时存在两个格子以上变化值相同，那么在调整后将其中一个变未空格，其余填0。
	//******* 重要 有必要添加
}
//返回一个矩阵路径
user_nn_matrix *user_emd_get_loop_path_list(user_nn_matrix *maps_matrix, int centor_point) {
	//创建一个数组设置路径表
	user_nn_matrix *path_matrix = user_nn_matrix_create(maps_matrix->width, maps_matrix->height);//创建路径矩阵
	user_nn_matrix *result_matrix = user_nn_matrix_create(maps_matrix->width, maps_matrix->height);
	int start_end_point_array[4], next_point_array[4], path_index = 1, new_point = 0;

	*user_nn_matrix_ext_value_index(maps_matrix, centor_point) = 1;//删除maps_matrix中心点
	*user_nn_matrix_ext_value_index(path_matrix, centor_point) = 1;//删除path_matrix中心点
	result_matrix->data[0] = (float)centor_point;
	result_matrix->data[1] = (float)user_emd_get_center_around_point(maps_matrix, centor_point, centor_point, start_end_point_array);//按左上右下的方式查找相邻的周围四个点
	*user_nn_matrix_ext_value_index(path_matrix, (int)result_matrix->data[1]) = 1.0f;//设置点为路径点

	//printf("\ncentor_point:%d\n", centor_point);
	//user_nn_matrix_printf(NULL, maps_matrix);//
	for (;;) {
		path_index++;
		user_emd_get_center_around_point(maps_matrix, centor_point, (int)result_matrix->data[path_index - 1], next_point_array);//获取周围的点
		new_point = user_emd_check_vaild_point(path_matrix, next_point_array);//校验哪些点有效 同时删除无效的点
		if (new_point == -1) {//所有点都无效
			*user_nn_matrix_ext_value_index(maps_matrix, (int)result_matrix->data[path_index - 1]) = 1.0f;//删除已经走死的路径点
			path_index = 1;
			user_nn_matrix_memset(result_matrix, 0);//清空路径
			user_nn_matrix_memset(path_matrix, 0);//清空路径
			result_matrix->data[0] = (float)centor_point;
			result_matrix->data[1] = (float)user_emd_get_center_around_point(maps_matrix, centor_point, centor_point, start_end_point_array);//按左上右下的方式查找相邻的周围四个点
			if ((-1 == start_end_point_array[0]) && (-1 == start_end_point_array[1]) &&
				(-1 == start_end_point_array[2]) && (-1 == start_end_point_array[3])) {
				//找不到任何完整路径点 进行退出
				//user_nn_matrix_printf(NULL, path_matrix);//
				user_nn_matrix_delete(path_matrix);//删除矩阵
				user_nn_matrix_delete(result_matrix);//删除矩阵
				return NULL;//返回空
			}
			if (result_matrix->data[1] != -1) {
				*user_nn_matrix_ext_value_index(path_matrix, (int)result_matrix->data[1]) = 1.0f;//设置为路径点
			}
		}
		else {
			result_matrix->data[path_index] = (float)new_point;//设置新的有效点
			if ((new_point == start_end_point_array[0]) || (new_point == start_end_point_array[1]) ||
				(new_point == start_end_point_array[2]) || (new_point == start_end_point_array[3])) {
				//找到终点 返回路径
				user_nn_matrix_delete(path_matrix);//删除矩阵
				return result_matrix;
			}
		}
		//user_nn_matrix_printf(NULL, maps_matrix);//
		//user_nn_matrix_printf(NULL, path_matrix);//
	}
	return NULL;
}

//检查路径点是否为已存在
int user_emd_check_vaild_point(user_nn_matrix *path_matrix, int *around_array) {
	for (int index = 0; index < 4; index++) {
		if (around_array[index] != -1) {//检查是否存在周围点
			if (*user_nn_matrix_ext_value_index(path_matrix, around_array[index]) == 0) {//如果此点不是路径直接返回
				*user_nn_matrix_ext_value_index(path_matrix, around_array[index]) = 1;//设置为路径点
				return around_array[index];
			}
			else {
				around_array[index] = -1;//删除目标点
			}
		}
	}
	return -1;
}
//查找开始或者结束点
int user_emd_get_center_around_point(user_nn_matrix *matrix, int obstacle_point, int center_point, int *around_array) {
	int point_x = center_point % matrix->width;
	int point_y = center_point / matrix->width;
	int next_point = 0;
	int result = -1;
	around_array[0] = -1; around_array[1] = -1; around_array[2] = -1; around_array[3] = -1;	//向左查找点

																							//向下查找点
	for (int point_h = point_y + 1; point_h < matrix->height; point_h++) {
		//printf("\n v:%f\n", *user_nn_matrix_ext_value(matrix, point_w, point_y));
		next_point = point_h * matrix->width + point_x;
		if (obstacle_point == next_point) {//跳过障碍点
			break;
		}
		if (*user_nn_matrix_ext_value(matrix, point_x, point_h) == 0)
		{
			around_array[3] = next_point;
			result = next_point;
			break;
		}
	}
	//向右查找点
	for (int point_w = point_x + 1; point_w < matrix->width; point_w++) {
		//printf("\n v:%f\n", *user_nn_matrix_ext_value(matrix, point_w, point_y));
		next_point = point_y * matrix->width + point_w;
		if (obstacle_point == next_point) {//跳过障碍点
			break;
		}
		if (*user_nn_matrix_ext_value(matrix, point_w, point_y) == 0)
		{
			around_array[2] = next_point;
			result = next_point;
			break;
		}
	}
	//向上查找点
	for (int point_h = point_y - 1; point_h >= 0; point_h--) {
		//printf("\n v:%f\n", *user_nn_matrix_ext_value(matrix, point_w, point_y));
		next_point = point_h * matrix->width + point_x;
		if (obstacle_point == next_point) {//跳过障碍点
			break;
		}
		if (*user_nn_matrix_ext_value(matrix, point_x, point_h) == 0)
		{
			around_array[1] = next_point;//
			result = next_point;
			break;
		}
	}
	//向左查找点
	for (int point_w = point_x - 1; point_w >= 0; point_w--) {
		//printf("\n v:%f\n", *user_nn_matrix_ext_value(matrix, point_w, point_y));
		next_point = point_y * matrix->width + point_w;
		if (obstacle_point == next_point) {//跳过障碍点
			break;
		}
		if (*user_nn_matrix_ext_value(matrix, point_w, point_y) == 0)
		{
			around_array[0] = next_point;//
			result = next_point;
			break;
		}
	}
	return result;
}

//计算运输算法的最小值计算
float user_emd_earth_movers_distance(float *dist_array, float *height_array, int height, float *width_array, int width) {
	user_nn_matrix *src_matrix = NULL;//原始矩阵
	user_nn_matrix *vogel_matirx = NULL;//第一阶段vogel矩阵
	user_nn_matrix *vogel_matirx_t = NULL;//第一阶段vogel矩阵的复制
	user_nn_matrix *potential_matrix_t = NULL;//计算位势差的矩阵
	user_nn_matrix *censor_matrix_t = NULL;//校验位势差的矩阵
	user_nn_matrix *path_matrix_t = NULL;//回路规划矩阵
	float result = 0.0f;

	src_matrix = user_emd_object_create(dist_array, height_array, height, width_array, width);
	vogel_matirx = user_emd_get_vogel_init_martix(src_matrix);//采用vogel方式进行首次求解
	for (;;) {
		vogel_matirx_t = user_nn_matrix_cpy_create(vogel_matirx);//复制矩阵
		user_emd_matrix_unzero_mapping_cpy(vogel_matirx_t, src_matrix);//vogel_matirx_t不是0的位置被src_matrix相同位置数据覆盖
		potential_matrix_t = user_emd_potential_plan_matrix(vogel_matirx_t);//计算位势差
		user_emd_matrix_zero_mapping_cpy(potential_matrix_t, src_matrix);//potential_matrix_t是0的位置被src_matrix相同位置数据覆盖 不是0的设置为0
		censor_matrix_t = user_emd_censor_potential_matrix(potential_matrix_t);//进行对偶数检验
		user_emd_matrix_set_row_col(censor_matrix_t, *user_nn_matrix_return_max_addr(censor_matrix_t));//第一行与第一列设置为最大值，后面处理主要是找最小值
		if (*user_nn_matrix_return_min_addr(censor_matrix_t) >= 0) {//找不到最小值直接退出
			break;
		}
		path_matrix_t = user_emd_get_loop_path_list(censor_matrix_t, user_nn_matrix_return_min_index(censor_matrix_t));//获取路径
		user_emd_adjust_matrix_value(vogel_matirx, path_matrix_t);//更新值

		user_nn_matrix_delete(vogel_matirx_t);
		user_nn_matrix_delete(potential_matrix_t);
		user_nn_matrix_delete(censor_matrix_t);
		user_nn_matrix_delete(path_matrix_t);
	}
	result =  user_emd_matrix_get_cost_value(src_matrix, vogel_matirx);//获取结果

	user_nn_matrix_delete(src_matrix); 
	user_nn_matrix_delete(vogel_matirx);
	/*user_nn_matrix_delete(vogel_matirx_t);
	user_nn_matrix_delete(potential_matrix_t);
	user_nn_matrix_delete(censor_matrix_t);
	user_nn_matrix_delete(path_matrix_t);*/

	return result;
}
/*
	计算emd距离
	//float content[]={0.001,0.359,0.224,0.394,1.024,0.1,0.05,0.99,0.7,0.07,0.07,0.07};//矩阵数据
	//float height_weight[]={0.2,0.7,0.1};//输入数据的权重
	//float width_weight[]={0.45,0.05,0.05,0.45};//输出数据的权重
	
	//float content[]={0.5,11, 3, 6,5, 9,10, 2,18, 7, 4, 1};//矩阵数据
	//float height_weight[]={5,10,15};//输入数据的权重
	//float width_weight[]={3,3,12,12};//输出数据的权重

	float content[]={0.1,1,1,1,1,1,1,1,1,1,1,1};//矩阵数据
	float height_weight[]={0.2,0.6,0.2};//输入数据的权重
	float width_weight[]={0.3,0.3,0.3,0.1};//输出数据的权重
	
	float total_distence=0;

	user_nn_matrix *data_object = NULL;
	data_object = user_emd_object_create(content,height_weight,sizeof(height_weight)/sizeof(float)
												,width_weight,sizeof(width_weight)/sizeof(float));
	if(data_object != NULL){
		while(1){
			float dis = user_emd_calculate_once(data_object);
			//user_nn_matrix_printf(NULL,data_object);
			printf("\n%f\n",dis);
			if(dis < FLT_MAX){
				total_distence += dis;
			}else{
				break;
			}
			
		}
		printf("\ntotal distence : %f\n",total_distence);
	}else{
		printf("\ninput data error\n");
	}

*/

