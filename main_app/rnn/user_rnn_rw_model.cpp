

#include "../rnn/user_rnn_rw_model.h"


//�������Ϣ
//file �ļ�����
//offset ƫ�Ƶ�ַ
//layers ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_save_layer(FILE *file, long offset, user_rnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fwrite(layers, sizeof(user_rnn_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ����Ϣ
//file �ļ�����
//offset ƫ�Ƶ�ַ
//layers ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_read_layer(FILE *file, long offset, user_rnn_layers *layers){
	fseek(file, offset, SEEK_SET);
	fread(layers, sizeof(user_rnn_layers), 1, file);//д���
	return ftell(file);
}
//���������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//input ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_save_input(FILE *file, long offset, user_rnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fwrite(input, sizeof(user_rnn_input_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ�������Ϣ
//file �ļ�����
//offset ƫ�Ƶ�ַ
//input ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_read_input(FILE *file, long offset, user_rnn_input_layers *input){
	fseek(file, offset, SEEK_SET);
	fread(input, sizeof(user_rnn_input_layers), 1, file);//д���
	return ftell(file);
}
//��������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//conv ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_save_hidden(FILE *file, long offset, user_rnn_hidden_layers *conv){
	fseek(file, offset, SEEK_SET);
	fwrite(conv, sizeof(user_rnn_hidden_layers), 1, file);//д���
	return ftell(file);
}
//����������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//conv ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_read_hidden(FILE *file, long offset, user_rnn_hidden_layers *conv){
	fseek(file, offset, SEEK_SET);
	fread(conv, sizeof(user_rnn_hidden_layers), 1, file);//д���
	return ftell(file);
}
//���������
//file �ļ�����
//offset ƫ�Ƶ�ַ
//output ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_save_output(FILE *file, long offset, user_rnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fwrite(output, sizeof(user_rnn_output_layers), 1, file);//д���
	return ftell(file);
}
//��ȡ�����
//file �ļ�����
//offset ƫ�Ƶ�ַ
//output ����Ķ���
//���� �ļ�ָ��λ��
static long user_rnn_model_read_output(FILE *file, long offset, user_rnn_output_layers *output){
	fseek(file, offset, SEEK_SET);
	fread(output, sizeof(user_rnn_output_layers), 1, file);//д���
	return ftell(file);
}
//����ģ��
//path ����·��
//layers �����
//���� �ɹ�����ʧ��
bool user_rnn_model_save_model(user_rnn_layers *layers,int id){
	char full_path[MAX_PATH] = "";
	FILE *model_file = NULL;
	user_rnn_input_layers	*input_infor = NULL;
	user_rnn_hidden_layers	*hidden_infor = NULL;
	user_rnn_output_layers  *output_infor = NULL;
	long layers_offset = user_nn_model_rnn_layer_addr;//�㱣��λ��
	long infor_offset = user_nn_model_rnn_content_addr;//��Ϣ����λ��
	long data_offset = user_nn_model_rnn_data_addr;//���ݶ���λ��
	if (id == 0) {
		sprintf(full_path, "%s.bin", user_nn_model_rnn_file_name);
	}
	else {
		sprintf(full_path, "%s_%d.bin", user_nn_model_rnn_file_name, id);
	}
	fopen_s(&model_file, full_path, "wb+");//��ģ���ļ�
	if (model_file == NULL)return false;

	while (1){
		switch (layers->type){
		case u_rnn_layer_type_null:
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			break;
		case u_rnn_layer_type_input:
			input_infor = (user_rnn_input_layers *)layers->content;
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_rnn_model_save_input(model_file, infor_offset, input_infor);
			break;
		case u_rnn_layer_type_hidden:
			hidden_infor = (user_rnn_hidden_layers *)layers->content;//
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_rnn_model_save_hidden(model_file, infor_offset, hidden_infor);//������������
			data_offset = user_nn_model_save_matrix(model_file, data_offset, hidden_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_save_matrix(model_file, data_offset, hidden_infor->kernel_matrix_t);//����ƫ�ò���
			data_offset = user_nn_model_save_matrix(model_file, data_offset, hidden_infor->kernel_matrix);//����ƫ�ò���
			break;
		case u_rnn_layer_type_output:
			output_infor = (user_rnn_output_layers *)layers->content;//
			layers_offset = user_rnn_model_save_layer(model_file, layers_offset, layers);//�������Ϣ
			infor_offset = user_rnn_model_save_output(model_file, infor_offset, output_infor);//
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
user_rnn_layers	*user_rnn_model_load_model(int id){
	char full_path[MAX_PATH] = "";
	FILE *model_file = NULL;
	long layers_offset = user_nn_model_rnn_layer_addr;//�㱣��λ��
	long infor_offset = user_nn_model_rnn_content_addr;//��Ϣ����λ��
	long data_offset = user_nn_model_rnn_data_addr;//���ݶ���λ��
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
	fopen_s(&model_file, full_path, "rb");//��ģ���ļ�
	if (model_file == NULL)return NULL;
	temp_cnn_layers = user_rnn_layers_create(u_rnn_layer_type_null, 0);
	while (1){
		layers_offset = user_rnn_model_read_layer(model_file, layers_offset, temp_cnn_layers);//��ȡ����Ϣ
		temp_cnn_layers->content = NULL;//�����ڴ��ص� ����ڴ�ͬʱ���������
		temp_cnn_layers->next = NULL;//�����ڴ��ص� ����ڴ�ͬʱ���������
		switch (temp_cnn_layers->type){
		case u_rnn_layer_type_null:
			rnn_layers = user_rnn_layers_create(u_rnn_layer_type_null, 0);//����һ���ղ����ڻ�ȡ����
			break;
		case u_rnn_layer_type_input:
			temp_input_infor = (user_rnn_input_layers *)malloc(sizeof(user_rnn_input_layers));//�����ռ�
			infor_offset = user_rnn_model_read_input(model_file, infor_offset, temp_input_infor);//���ز���Ϣ
			input_infor = user_rnn_layers_input_create(rnn_layers, temp_input_infor->feature_width, temp_input_infor->feature_height, temp_input_infor->time_number);//���������
			free(temp_input_infor);//�ͷſռ�
			break;
		case u_rnn_layer_type_hidden:
			temp_hidden_infor = (user_rnn_hidden_layers *)malloc(sizeof(user_rnn_hidden_layers));//�����ռ�
			infor_offset = user_rnn_model_read_hidden(model_file, infor_offset, temp_hidden_infor);//��ȡ�������Ϣ
			hidden_infor = user_rnn_layers_hidden_create(rnn_layers, temp_hidden_infor->feature_height, temp_hidden_infor->time_number);//���������
			data_offset = user_nn_model_read_matrix(model_file, data_offset, hidden_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_read_matrix(model_file, data_offset, hidden_infor->kernel_matrix_t);//����ƫ�ò���
			data_offset = user_nn_model_read_matrix(model_file, data_offset, hidden_infor->kernel_matrix);//����ƫ�ò���
			free(temp_hidden_infor);//�ͷſռ�
			break;
		case u_rnn_layer_type_output:
			temp_output_infor = (user_rnn_output_layers *)malloc(sizeof(user_rnn_output_layers));//�����ռ�
			infor_offset = user_rnn_model_read_output(model_file, infor_offset, temp_output_infor);//��ȡ�������Ϣ
			output_infor = user_rnn_layers_output_create(rnn_layers, temp_output_infor->feature_height, temp_output_infor->time_number);//���������
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->biases_matrix);//����ƫ�ò���
			data_offset = user_nn_model_read_matrix(model_file, data_offset, output_infor->kernel_matrix);//����ƫ�ò���
			free(temp_output_infor);//�ͷſռ�
			fclose(model_file);//�ر��ļ�
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
