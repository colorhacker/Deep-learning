

#include "../rnn/user_rnn_rw_model.h"


//保存层信息
//file 文件对象
//offset 偏移地址
//layers 保存的对象
//返回 文件指针位置
static long user_rnn_model_save_layer(FILE *file, long offset, user_rnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fwrite(layers, sizeof(user_rnn_layers), 1, file);//写入层
	return ftell(file);
}
//读取层信息
//file 文件对象
//offset 偏移地址
//layers 保存的对象
//返回 文件指针位置
static long user_rnn_model_read_layer(FILE *file, long offset, user_rnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fread(layers, sizeof(user_rnn_layers), 1, file);//写入层
	return ftell(file);
}
//保存输入层
//file 文件对象
//offset 偏移地址
//input 保存的对象
//返回 文件指针位置
static long user_rnn_model_save_input(FILE *file, long offset, user_rnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fwrite(input, sizeof(user_rnn_input_layers), 1, file);//写入层
	return ftell(file);
}
//读取输入层信息
//file 文件对象
//offset 偏移地址
//input 保存的对象
//返回 文件指针位置
static long user_rnn_model_read_input(FILE *file, long offset, user_rnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fread(input, sizeof(user_rnn_input_layers), 1, file);//写入层
	return ftell(file);
}
//保存卷积层
//file 文件对象
//offset 偏移地址
//conv 保存的对象
//返回 文件指针位置
static long user_rnn_model_save_hidden(FILE *file, long offset, user_rnn_hidden_layers *conv){
	fseek(file, offset, SEEK_SET);
	fwrite(conv, sizeof(user_rnn_hidden_layers), 1, file);//写入层
	return ftell(file);
}
//隐含层卷积层
//file 文件对象
//offset 偏移地址
//conv 保存的对象
//返回 文件指针位置
static long user_rnn_model_read_hidden(FILE *file, long offset, user_rnn_hidden_layers *conv){
	fseek(file, offset, SEEK_SET);
	fread(conv, sizeof(user_rnn_hidden_layers), 1, file);//写入层
	return ftell(file);
}
//保存输出层
//file 文件对象
//offset 偏移地址
//output 保存的对象
//返回 文件指针位置
static long user_rnn_model_save_output(FILE *file, long offset, user_rnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fwrite(output, sizeof(user_rnn_output_layers), 1, file);//写入层
	return ftell(file);
}
//读取输出层
//file 文件对象
//offset 偏移地址
//output 保存的对象
//返回 文件指针位置
static long user_rnn_model_read_output(FILE *file, long offset, user_rnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fread(output, sizeof(user_rnn_output_layers), 1, file);//写入层
	return ftell(file);
}
//保存模型
//path 保存路径
//layers 层对象
//返回 成功或者失败
bool user_rnn_model_save_model(user_rnn_layers *layers,int id){
	char full_path[MAX_PATH] = "";
	FILE *model_file = NULL;
	user_rnn_input_layers	*input_infor = NULL;
	user_rnn_hidden_layers	*hidden_infor = NULL;
	user_rnn_output_layers  *output_infor = NULL;
	long layers_offset = user_nn_model_rnn_layer_addr;//层保存位置
	long infor_offset = user_nn_model_rnn_content_addr;//信息描述位置
	long data_offset = user_nn_model_rnn_data_addr;//数据对象位置
	if (id == 0) {
		sprintf(full_path, "%s.bin", user_nn_model_rnn_file_name);
	}
	else {
		sprintf(full_path, "%s_%d.bin", user_nn_model_rnn_file_name, id);
	}
	fopen_s(&model_file, full_path, "wb+");//打开模型文件
	if (model_file == NULL)return false;

	while (1){
		switch (layers->type){
		case u_rnn_layer_type_null:
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			break;
		case u_rnn_layer_type_input:
			input_infor = (user_rnn_input_layers *)layers->content;
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_rnn_model_save_input(model_file, infor_offset, input_infor);
			break;
		case u_rnn_layer_type_hidden:
			hidden_infor = (user_rnn_hidden_layers *)layers->content;//
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_rnn_model_save_hidden(model_file, infor_offset, hidden_infor);//保存卷积层数据
			data_offset = user_nn_model_save_matrix(model_file, data_offset, hidden_infor->biases_matrix);//保存偏置参数
			data_offset = user_nn_model_save_matrix(model_file, data_offset, hidden_infor->kernel_matrix_t);//保存偏置参数
			data_offset = user_nn_model_save_matrix(model_file, data_offset, hidden_infor->kernel_matrix);//保存偏置参数
			break;
		case u_rnn_layer_type_output:
			output_infor = (user_rnn_output_layers *)layers->content;//
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//保存层信息
			infor_offset = user_rnn_model_save_output(model_file, infor_offset, output_infor);//
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
user_rnn_layers	*user_rnn_model_load_model(int id){
	char full_path[MAX_PATH] = "";
	FILE *model_file = NULL;
	long layers_offset = user_nn_model_rnn_layer_addr;//层保存位置
	long infor_offset = user_nn_model_rnn_content_addr;//信息描述位置
	long data_offset = user_nn_model_rnn_data_addr;//数据对象位置
	user_rnn_layers			*rnn_layers = NULL, *temp_cnn_layers = NULL;
	user_rnn_input_layers	*input_infor = NULL, *temp_input_infor = NULL;
	user_rnn_hidden_layers	*hidden_infor = NULL, *temp_hidden_infor = NULL;
	user_rnn_output_layers  *output_infor = NULL, *temp_output_infor = NULL;
	if (id == 0) {
		sprintf(full_path, "%s.bin", user_nn_model_rnn_file_name);
	}
	else {
		sprintf(full_path, "%s_%d.bin", user_nn_model_rnn_file_name, id);
	}
	fopen_s(&model_file, full_path, "rb");//打开模型文件
	if (model_file == NULL)return NULL;
	temp_cnn_layers = user_rnn_layers_create(u_rnn_layer_type_null, 0);
	while (1){
		layers_offset = user_rnn_model_read_layer(model_file, layers_offset, temp_cnn_layers);//获取层信息
		temp_cnn_layers->content = NULL;//避免内存重叠 清除内存同时清除了数据
		temp_cnn_layers->next = NULL;//避免内存重叠 清除内存同时清除了数据
		switch (temp_cnn_layers->type){
		case u_rnn_layer_type_null:
			rnn_layers = user_rnn_layers_create(u_rnn_layer_type_null, 0);//创建一个空层用于获取数据
			break;
		case u_rnn_layer_type_input:
			temp_input_infor = (user_rnn_input_layers *)malloc(sizeof(user_rnn_input_layers));//分配层空间
			infor_offset = user_rnn_model_read_input(model_file, infor_offset, temp_input_infor);//加载层信息
			input_infor = user_rnn_layers_input_create(rnn_layers, temp_input_infor->feature_width, temp_input_infor->feature_height, temp_input_infor->time_number);//创建输入层
			free(temp_input_infor);//释放空间
			break;
		case u_rnn_layer_type_hidden:
			temp_hidden_infor = (user_rnn_hidden_layers *)malloc(sizeof(user_rnn_hidden_layers));//分配层空间
			infor_offset = user_rnn_model_read_hidden(model_file, infor_offset, temp_hidden_infor);//读取输入层信息
			hidden_infor = user_rnn_layers_hidden_create(rnn_layers, temp_hidden_infor->feature_height, temp_hidden_infor->time_number);//创建卷积层
			data_offset = user_nn_model_read_matrix(model_file, data_offset, hidden_infor->biases_matrix);//载入偏置参数
			data_offset = user_nn_model_read_matrix(model_file, data_offset, hidden_infor->kernel_matrix_t);//载入偏置参数
			data_offset = user_nn_model_read_matrix(model_file, data_offset, hidden_infor->kernel_matrix);//载入偏置参数
			free(temp_hidden_infor);//释放空间
			break;
		case u_rnn_layer_type_output:
			temp_output_infor = (user_rnn_output_layers *)malloc(sizeof(user_rnn_output_layers));//分配层空间
			infor_offset = user_rnn_model_read_output(model_file, infor_offset, temp_output_infor);//读取输入层信息
			output_infor = user_rnn_layers_output_create(rnn_layers, temp_output_infor->feature_height, temp_output_infor->time_number);//创建输出层
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->biases_matrix);//载入偏置参数
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->kernel_matrix);//载入偏置参数
			free(temp_output_infor);//释放空间
			fclose(model_file);//关闭文件
			user_rnn_layers_delete(temp_cnn_layers);
			return rnn_layers;
			//break;
		default:
			printf("loading error\n");
			break;
		}
	}
	user_rnn_layers_delete(temp_cnn_layers);
	fclose(model_file);
	return NULL;
}
