

#include "../cnn/user_cnn_rw_model.h"

//�������Ϣ
//file �ļ�����
//offset ƫ�Ƶ�ַ
//layers ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_save_layer(FILE *file, long offset, user_cnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fwrite(layers, sizeof(user_cnn_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ����Ϣ
//file �ļ�����
//offset ƫ�Ƶ�ַ
//layers ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_read_layer(FILE *file, long offset, user_cnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fread(layers, sizeof(user_cnn_layers), 1, file);//д���
	return ftell(file);
}
//���������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//input ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_save_input(FILE *file, long offset, user_cnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fwrite(input, sizeof(user_cnn_input_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ�������Ϣ
//file �ļ�����
//offset ƫ�Ƶ�ַ
//input ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_read_input(FILE *file, long offset, user_cnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fread(input, sizeof(user_cnn_input_layers), 1, file);//д���
	return ftell(file);
}
//��������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//conv ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_save_conv(FILE *file, long offset, user_cnn_conv_layers *conv){
	fseek(file, offset, SEEK_SET);
	fwrite(conv, sizeof(user_cnn_conv_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ�����
//file �ļ�����
//offset ƫ�Ƶ�ַ
//conv ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_read_conv(FILE *file, long offset, user_cnn_conv_layers *conv){
	fseek(file, offset, SEEK_SET);
	fread(conv, sizeof(user_cnn_conv_layers), 1, file);//д���
	return ftell(file);
}
//����ػ���
//file �ļ�����
//offset ƫ�Ƶ�ַ
//pool ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_save_pool(FILE *file, long offset, user_cnn_pool_layers *pool){
	fseek(file, offset, SEEK_SET);
	fwrite(pool, sizeof(user_cnn_pool_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ�ػ���
//file �ļ�����
//offset ƫ�Ƶ�ַ
//pool ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_read_pool(FILE *file, long offset, user_cnn_pool_layers *pool){
	fseek(file, offset, SEEK_SET);
	fread(pool, sizeof(user_cnn_pool_layers), 1, file);//д���
	return ftell(file);
}
//����ȫ���Ӳ�
//file �ļ�����
//offset ƫ�Ƶ�ַ
//output ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_save_full(FILE *file, long offset, user_cnn_full_layers *full){
	fseek(file, offset, SEEK_SET);
	fwrite(full, sizeof(user_cnn_full_layers), 1, file);//д���
	return ftell(file);
}
//��ȡȫ���Ӳ�
//file �ļ�����
//offset ƫ�Ƶ�ַ
//output ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_read_full(FILE *file, long offset, user_cnn_full_layers *full){
	fseek(file, offset, SEEK_SET);
	fread(full, sizeof(user_cnn_full_layers), 1, file);//д���
	return ftell(file);
}
//���������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//output ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_save_output(FILE *file, long offset, user_cnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fwrite(output, sizeof(user_cnn_output_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ�����
//file �ļ�����
//offset ƫ�Ƶ�ַ
//output ����Ķ���
//���� �ļ�ָ��λ��
static long user_cnn_model_read_output(FILE *file, long offset, user_cnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fread(output, sizeof(user_cnn_output_layers), 1, file);//д���
	return ftell(file);
}
//����ģ��
//path ����·��
//layers �����
//���� �ɹ�����ʧ��
bool user_cnn_model_save_model(user_cnn_layers *layers,int id){
	char full_path[MAX_PATH] = "";
	FILE *model_file = NULL;
	user_cnn_input_layers	*input_infor = NULL;
	user_cnn_conv_layers	*conv_infor = NULL;
	user_cnn_pool_layers	*pool_infor = NULL;
	user_cnn_output_layers  *output_infor = NULL;
	user_cnn_full_layers	*full_infor = NULL;
	long layers_offset = user_nn_model_cnn_layer_addr;//�㱣��λ��
	long infor_offset = user_nn_model_cnn_content_addr;//��Ϣ����λ��
	long data_offset = user_nn_model_cnn_data_addr;//���ݶ���λ��
	if (id == 0) {
		sprintf(full_path, "%s.bin", user_nn_model_cnn_file_name);
	}
	else {
		sprintf(full_path, "%s_%d.bin", user_nn_model_cnn_file_name, id);
	}
	fopen_s(&model_file, full_path, "wb+");//��ģ���ļ�
	if (model_file == NULL)return false;

	while (1){
		switch (layers->type){
		case u_cnn_layer_type_null:
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			break;
		case u_cnn_layer_type_input:
			input_infor = (user_cnn_input_layers *)layers->content;
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_cnn_model_save_input(model_file, infor_offset, input_infor);
			break;
		case u_cnn_layer_type_conv:
			conv_infor = (user_cnn_conv_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_cnn_model_save_conv(model_file, infor_offset, conv_infor);//������������
			data_offset = user_nn_model_save_matrix(model_file, data_offset, conv_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_save_matrices(model_file, data_offset, conv_infor->kernel_matrices);//����ƫ�ò���		
			break;
		case u_cnn_layer_type_pool:
			pool_infor = (user_cnn_pool_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_cnn_model_save_pool(model_file, infor_offset, pool_infor);//������������
			break;
		case u_cnn_layer_type_full:
			full_infor = (user_cnn_full_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_cnn_model_save_full(model_file, infor_offset, full_infor);//
			data_offset = user_nn_model_save_matrix(model_file, data_offset, full_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_save_matrix(model_file, data_offset, full_infor->kernel_matrix);//����ƫ�ò���
			break;
		case u_cnn_layer_type_output:
			output_infor = (user_cnn_output_layers *)layers->content;//
			layers_offset = user_cnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_cnn_model_save_output(model_file, infor_offset, output_infor);//
			data_offset = user_nn_model_save_matrix(model_file, data_offset, output_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_save_matrix(model_file, data_offset, output_infor->kernel_matrix);//����ƫ�ò���
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
//����ģ��
//path ����·��
//���� null����ģ�Ͷ���
user_cnn_layers	*user_cnn_model_load_model(int id){
	char full_path[MAX_PATH] = "";
	FILE *model_file = NULL;
	long layers_offset = user_nn_model_cnn_layer_addr;//�㱣��λ��
	long infor_offset = user_nn_model_cnn_content_addr;//��Ϣ����λ��
	long data_offset = user_nn_model_cnn_data_addr;//���ݶ���λ��
	user_cnn_layers			*cnn_layers = NULL, *temp_cnn_layers = NULL;
	user_cnn_input_layers	*input_infor = NULL, *temp_input_infor = NULL;
	user_cnn_conv_layers	*conv_infor = NULL, *temp_conv_infor = NULL;
	user_cnn_pool_layers	*pool_infor = NULL, *temp_pool_infor = NULL;
	user_cnn_output_layers  *output_infor = NULL, *temp_output_infor = NULL;
	user_cnn_full_layers	*full_infor = NULL, *temp_full_infor = NULL;
	if (id == 0) {
		sprintf(full_path, "%s.bin", user_nn_model_cnn_file_name);
	}
	else {
		sprintf(full_path, "%s_%d.bin", user_nn_model_cnn_file_name, id);
	}
	fopen_s(&model_file, full_path, "rb");//��ģ���ļ�
	if (model_file == NULL)return NULL;
	temp_cnn_layers = user_cnn_layers_create(u_cnn_layer_type_null, 0);
	while (1){
		layers_offset = user_cnn_model_read_layer(model_file, layers_offset, temp_cnn_layers);//��ȡ����Ϣ
		temp_cnn_layers->content = NULL;//�����ڴ��ص� ����ڴ�ͬʱ���������
		temp_cnn_layers->next = NULL;//�����ڴ��ص� ����ڴ�ͬʱ���������
		switch (temp_cnn_layers->type){
		case u_cnn_layer_type_null:
			cnn_layers = user_cnn_layers_create(u_cnn_layer_type_null, 0);//����һ���ղ����ڻ�ȡ����
			break;
		case u_cnn_layer_type_input:
			temp_input_infor = (user_cnn_input_layers *)malloc(sizeof(user_cnn_input_layers));//�����ռ�
			infor_offset = user_cnn_model_read_input(model_file, infor_offset, temp_input_infor);//���ز���Ϣ
			input_infor = user_cnn_layers_input_create(cnn_layers, temp_input_infor->feature_width, temp_input_infor->feature_height, temp_input_infor->feature_number);//���������
			free(temp_input_infor);//�ͷſռ�
			break;
		case u_cnn_layer_type_conv:
			temp_conv_infor = (user_cnn_conv_layers *)malloc(sizeof(user_cnn_conv_layers));//�����ռ�
			infor_offset = user_cnn_model_read_conv(model_file, infor_offset, temp_conv_infor);//��ȡ�������Ϣ
			conv_infor = user_cnn_layers_convolution_create(cnn_layers, temp_conv_infor->kernel_width, temp_conv_infor->kernel_height, temp_conv_infor->feature_number);//���������
			data_offset = user_nn_model_read_matrix(model_file, data_offset, conv_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_read_matrices(model_file, data_offset, conv_infor->kernel_matrices);//����ƫ�ò���		
			free(temp_conv_infor);//�ͷſռ�
			break;
		case u_cnn_layer_type_pool:
			temp_pool_infor = (user_cnn_pool_layers *)malloc(sizeof(user_cnn_pool_layers));//�����ռ�
			infor_offset = user_cnn_model_read_pool(model_file, infor_offset, temp_pool_infor);//��ȡ�������Ϣ
			pool_infor = user_cnn_layers_pooling_create(cnn_layers, temp_pool_infor->pool_width, temp_pool_infor->pool_height, temp_pool_infor->pool_type);//���������
			free(temp_pool_infor);//�ͷſռ�
			break;
		case u_cnn_layer_type_full:
			temp_full_infor = (user_cnn_full_layers *)malloc(sizeof(user_cnn_full_layers));//�����ռ�
			infor_offset = user_cnn_model_read_full(model_file, infor_offset, temp_full_infor);//��ȡ�������Ϣ
			full_infor = user_cnn_layers_fullconnect_create(cnn_layers);//����ȫ���Ӳ�
			data_offset = user_nn_model_read_matrix(model_file, data_offset, full_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_read_matrix(model_file, data_offset, full_infor->kernel_matrix);//����ƫ�ò���
			free(temp_full_infor);//�ͷſռ�
			break;
		case u_cnn_layer_type_output:
			temp_output_infor = (user_cnn_output_layers *)malloc(sizeof(user_cnn_output_layers));//�����ռ�
			infor_offset = user_cnn_model_read_output(model_file, infor_offset, temp_output_infor);//��ȡ�������Ϣ
			output_infor = user_cnn_layers_output_create(cnn_layers, temp_output_infor->class_number);//���������
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->kernel_matrix);//����ƫ�ò���
			free(temp_output_infor);//�ͷſռ�
			fclose(model_file);//�ر��ļ�
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
