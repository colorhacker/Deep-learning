

#include "../cnn/user_cnn_rw_model.h"

//保存层信息
//file 文件对象
//offset 偏移地址
//layers 保存的对象
//返回 文件指针位置
static long user_cnn_model_save_layer(FILE *file, long offset, user_cnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fwrite(layers, sizeof(user_cnn_layers), 1, file);//写入层
	return ftell(file);
}
//读取层信息
//file 文件对象
//offset 偏移地址
//layers 保存的对象
//返回 文件指针位置
static long user_cnn_model_read_layer(FILE *file, long offset, user_cnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fread(layers, sizeof(user_cnn_layers), 1, file);//写入层
	return ftell(file);
}
//保存输入层
//file 文件对象
//offset 偏移地址
//input 保存的对象
//返回 文件指针位置
static long user_cnn_model_save_input(FILE *file, long offset, user_cnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fwrite(input, sizeof(user_cnn_input_layers), 1, file);//写入层
	return ftell(file);
}
//读取输入层信息
//file 文件对象
//offset 偏移地址
//input 保存的对象
//返回 文件指针位置
static long user_cnn_model_read_input(FILE *file, long offset, user_cnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fread(input, sizeof(user_cnn_input_layers), 1, file);//写入层
	return ftell(file);
}
//保存卷积层
//file 文件对象
//offset 偏移地址
//conv 保存的对象
//返回 文件指针位置
static long user_cnn_model_save_conv(FILE *file, long offset, user_cnn_conv_layers *conv){
	fseek(file, offset, SEEK_SET);
	fwrite(conv, sizeof(user_cnn_conv_layers), 1, file);//写入层
	return ftell(file);
}
//读取卷积层
//file 文件对象
//offset 偏移地址
//conv 保存的对象
//返回 文件指针位置
static long user_cnn_model_read_conv(FILE *file, long offset, user_cnn_conv_layers *conv){
	fseek(file, offset, SEEK_SET);
	fread(conv, sizeof(user_cnn_conv_layers), 1, file);//写入层
	return ftell(file);
}
//保存池化层
//file 文件对象
//offset 偏移地址
//pool 保存的对象
//返回 文件指针位置
static long user_cnn_model_save_pool(FILE *file, long offset, user_cnn_pool_layers *pool){
	fseek(file, offset, SEEK_SET);
	fwrite(pool, sizeof(user_cnn_pool_layers), 1, file);//写入层
	return ftell(file);
}
//读取池化层
//file 文件对象
//offset 偏移地址
//pool 保存的对象
//返回 文件指针位置
static long user_cnn_model_read_pool(FILE *file, long offset, user_cnn_pool_layers *pool){
	fseek(file, offset, SEEK_SET);
	fread(pool, sizeof(user_cnn_pool_layers), 1, file);//写入层
	return ftell(file);
}
//保存全连接层
//file 文件对象
//offset 偏移地址
//output 保存的对象
//返回 文件指针位置
static long user_cnn_model_save_full(FILE *file, long offset, user_cnn_full_layers *full){
	fseek(file, offset, SEEK_SET);
	fwrite(full, sizeof(user_cnn_full_layers), 1, file);//写入层
	return ftell(file);
}
//读取全连接层
//file 文件对象
//offset 偏移地址
//output 保存的对象
//返回 文件指针位置
static long user_cnn_model_read_full(FILE *file, long offset, user_cnn_full_layers *full){
	fseek(file, offset, SEEK_SET);
	fread(full, sizeof(user_cnn_full_layers), 1, file);//写入层
	return ftell(file);
}
//保存输出层
//file 文件对象
//offset 偏移地址
//output 保存的对象
//返回 文件指针位置
static long user_cnn_model_save_output(FILE *file, long offset, user_cnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fwrite(output, sizeof(user_cnn_output_layers), 1, file);//写入层
	return ftell(file);
}
//读取输出层
//file 文件对象
//offset 偏移地址
//output 保存的对象
//返回 文件指针位置
static long user_cnn_model_read_output(FILE *file, long offset, user_cnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fread(output, sizeof(user_cnn_output_layers), 1, file);//写入层
	return ftell(file);
}
//保存模型
//path 保存路径
//layers 层对象
//返回 成功或者失败
bool user_cnn_model_save_model(const char *path,user_cnn_layers *layers){
	FILE *model_file = NULL;
	user_cnn_input_layers	*input_infor = NULL;
	user_cnn_conv_layers	*conv_infor = NULL;
	user_cnn_pool_layers	*pool_infor = NULL;
	user_cnn_output_layers  *output_infor = NULL;
	user_cnn_full_layers	*full_infor = NULL;
	long layers_offset = user_nn_model_cnn_layer_addr;//层保存位置
	long infor_offset = user_nn_model_cnn_content_addr;//信息描述位置
	long data_offset = user_nn_model_cnn_data_addr;//数据对象位置

	fopen_s(&model_file,path, "wb+");//打开模型文件
	if (model_file == NULL)return false;

	while (1){
		switch (layers->type){
		case u_cnn_layer_type_null:
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			break;
		case u_cnn_layer_type_input:
			input_infor = (user_cnn_input_layers *)layers->content;
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_cnn_model_save_input(model_file, infor_offset, input_infor);
			break;
		case u_cnn_layer_type_conv:
			conv_infor = (user_cnn_conv_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_cnn_model_save_conv(model_file, infor_offset, conv_infor);//保存卷积层数据
			data_offset = user_nn_model_save_matrix(model_file, data_offset, conv_infor->biases_matrix);//保存偏置参数
			data_offset = user_nn_model_save_matrices(model_file, data_offset, conv_infor->kernel_matrices);//保存偏置参数		
			break;
		case u_cnn_layer_type_pool:
			pool_infor = (user_cnn_pool_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_cnn_model_save_pool(model_file, infor_offset, pool_infor);//保存卷积层数据
			break;
		case u_cnn_layer_type_full:
			full_infor = (user_cnn_full_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_cnn_model_save_full(model_file, infor_offset, full_infor);//
			data_offset = user_nn_model_save_matrix(model_file, data_offset, full_infor->biases_matrix);//保存偏置参数
			data_offset = user_nn_model_save_matrix(model_file, data_offset, full_infor->kernel_matrix);//保存偏置参数
			break;
		case u_cnn_layer_type_output:
			output_infor = (user_cnn_output_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_cnn_model_save_output(model_file, infor_offset, output_infor);//
			data_offset = user_nn_model_save_matrix(model_file, data_offset, output_infor->biases_matrix);//保存偏置参数
			data_offset = user_nn_model_save_matrix(model_file, data_offset, output_infor->kernel_matrix);//保存偏置参数
			break;
		default:
			break;
		}
		if (layers->next == NULL){
			break;
		}
		else{
			layers = layers->next;
		}
	}
	fclose(model_file);
	return true;
}
//加载模型
//path 保存路径
//返回 null或者模型对象
user_cnn_layers	*user_cnn_model_load_model(const char *path){
	FILE *model_file = NULL;
	long layers_offset = user_nn_model_cnn_layer_addr;//层保存位置
	long infor_offset = user_nn_model_cnn_content_addr;//信息描述位置
	long data_offset = user_nn_model_cnn_data_addr;//数据对象位置
	user_cnn_layers			*cnn_layers = NULL, *temp_cnn_layers = NULL;
	user_cnn_input_layers	*input_infor = NULL, *temp_input_infor = NULL;
	user_cnn_conv_layers	*conv_infor = NULL, *temp_conv_infor = NULL;
	user_cnn_pool_layers	*pool_infor = NULL, *temp_pool_infor = NULL;
	user_cnn_output_layers  *output_infor = NULL, *temp_output_infor = NULL;
	user_cnn_full_layers	*full_infor = NULL, *temp_full_infor = NULL;

	fopen_s(&model_file,path, "rb");//打开模型文件
	if (model_file == NULL)return NULL;
	temp_cnn_layers = user_cnn_layers_create(u_cnn_layer_type_null, 0);
	while (1){
		layers_offset = user_cnn_model_read_layer(model_file, layers_offset, temp_cnn_layers);//获取层信息
		temp_cnn_layers->content = NULL;//避免内存重叠 清除内存同时清除了数据
		temp_cnn_layers->next = NULL;//避免内存重叠 清除内存同时清除了数据
		switch (temp_cnn_layers->type){
		case u_cnn_layer_type_null:
			cnn_layers = user_cnn_layers_create(u_cnn_layer_type_null, 0);//创建一个空层用于获取数据
			break;
		case u_cnn_layer_type_input:
			temp_input_infor = (user_cnn_input_layers *)malloc(sizeof(user_cnn_input_layers));//分配层空间
			infor_offset = user_cnn_model_read_input(model_file, infor_offset, temp_input_infor);//加载层信息
			input_infor = user_cnn_layers_input_create(cnn_layers, temp_input_infor->feature_width, temp_input_infor->feature_height, temp_input_infor->feature_number);//创建输入层
			free(temp_input_infor);//释放空间
			break;
		case u_cnn_layer_type_conv:
			temp_conv_infor = (user_cnn_conv_layers *)malloc(sizeof(user_cnn_conv_layers));//分配层空间
			infor_offset = user_cnn_model_read_conv(model_file, infor_offset, temp_conv_infor);//读取输入层信息
			conv_infor = user_cnn_layers_convolution_create(cnn_layers, temp_conv_infor->kernel_width, temp_conv_infor->kernel_height, temp_conv_infor->feature_number);//创建卷积层
			data_offset = user_nn_model_read_matrix(model_file, data_offset, conv_infor->biases_matrix);//载入偏置参数
			data_offset = user_nn_model_read_matrices(model_file, data_offset, conv_infor->kernel_matrices);//载入偏置参数		
			free(temp_conv_infor);//释放空间
			break;
		case u_cnn_layer_type_pool:
			temp_pool_infor = (user_cnn_pool_layers *)malloc(sizeof(user_cnn_pool_layers));//分配层空间
			infor_offset = user_cnn_model_read_pool(model_file, infor_offset, temp_pool_infor);//读取输入层信息
			pool_infor = user_cnn_layers_pooling_create(cnn_layers, temp_pool_infor->pool_width, temp_pool_infor->pool_height, temp_pool_infor->pool_type);//创建输入层
			free(temp_pool_infor);//释放空间
			break;
		case u_cnn_layer_type_full:
			temp_full_infor = (user_cnn_full_layers *)malloc(sizeof(user_cnn_full_layers));//分配层空间
			infor_offset = user_cnn_model_read_full(model_file, infor_offset, temp_full_infor);//读取输入层信息
			full_infor = user_cnn_layers_fullconnect_create(cnn_layers);//创建全连接层
			data_offset = user_nn_model_read_matrix(model_file, data_offset, full_infor->biases_matrix);//载入偏置参数
			data_offset = user_nn_model_read_matrix(model_file, data_offset, full_infor->kernel_matrix);//载入偏置参数
			free(temp_full_infor);//释放空间
			break;
		case u_cnn_layer_type_output:
			temp_output_infor = (user_cnn_output_layers *)malloc(sizeof(user_cnn_output_layers));//分配层空间
			infor_offset = user_cnn_model_read_output(model_file, infor_offset, temp_output_infor);//读取输入层信息
			output_infor = user_cnn_layers_output_create(cnn_layers, temp_output_infor->class_number);//创建输出层
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->biases_matrix);//载入偏置参数
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->kernel_matrix);//载入偏置参数
			free(temp_output_infor);//释放空间
			fclose(model_file);//关闭文件
			user_cnn_layers_delete(temp_cnn_layers);
			return cnn_layers;
			//break;
		default:
			printf("loading error\n");
			break;
		}
	}
	user_cnn_layers_delete(temp_cnn_layers);
	fclose(model_file);
	return NULL;
}
