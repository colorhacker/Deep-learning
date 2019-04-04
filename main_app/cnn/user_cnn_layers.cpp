
#include "user_cnn_layers.h"

//����ָ����
user_cnn_layers *user_cnn_layers_get(user_cnn_layers *dest, int index){
	while (index--){
		if (dest->next != NULL){
			dest = dest->next;
		}else{
		}
	}
	return dest;
}

//����һ����
//������
//type��������
//index��ָ��
//���� ������Ĳ�
user_cnn_layers *user_cnn_layers_create(user_cnn_layer_type type,int index){
	user_cnn_layers *cnn_layers = NULL;

	cnn_layers = (user_cnn_layers *)malloc(sizeof(user_cnn_layers));//�����ڴ�
	cnn_layers->prior	= NULL;//ָ����һ��
	cnn_layers->type	= type;//�������
	cnn_layers->index	= index;//ָ��
	cnn_layers->content = NULL;//ָ������
	cnn_layers->next	= NULL;//ָ����һ��

	return cnn_layers;
}
//ɾ����
void user_cnn_layers_delete(user_cnn_layers *layers){
	if (layers != NULL){
		if (layers->content != NULL){
			free(layers->content);
		}
		free(layers);
	}
}
//���������
//����
//width���������ݵĿ��
//height���������ݵĸ߶�
//maps���������ݵ�����
//���أ��ɹ���ʧ��
user_cnn_input_layers *user_cnn_layers_input_create(user_cnn_layers *cnn_layers, int feature_width, int feature_height, int feature_number){
	user_cnn_layers			*last_layers = cnn_layers;
	user_cnn_input_layers	*input_layers = NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//��ѯ����cnn_layers�ն���
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_input, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->content = malloc(sizeof(user_cnn_input_layers));//�����ڴ������Ķ���ռ�
	input_layers = (user_cnn_input_layers *)last_layers->next->content;//ת����ǰ���ֵ �������ò���

	input_layers->feature_width		= feature_width;//�����������ݵĿ��
	input_layers->feature_height	= feature_height;//�����������ݵĸ߶�
	input_layers->feature_number	= feature_number;//�����������ݵĸ���
	input_layers->feature_matrices	= user_nn_matrices_create(1, input_layers->feature_number, input_layers->feature_width, input_layers->feature_height);//����������������ݾ��� 

	return input_layers;
}

//���������
//����
//outputmaps�����ͼ������
//kernelsize������˴�С
//���� 
user_cnn_conv_layers *user_cnn_layers_convolution_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height ,int feature_number){
	user_cnn_layers			*last_layers	= cnn_layers;
	user_cnn_conv_layers	*conv_layers	= NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//��ѯ����cnn_layers�ն���
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_conv, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_cnn_conv_layers));//���������ڴ��������ռ�
	conv_layers = (user_cnn_conv_layers *)last_layers->next->content;//�����������ȡ

	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers	= (user_cnn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		conv_layers->feature_width				= temp_layers->feature_width - kernel_width + 1;//���㵱ǰ������ݵĶ����� ��ʽ������߶�=����߶�-�����+1
		conv_layers->feature_height				= temp_layers->feature_height - kernel_height + 1;//���㵱ǰ������ݶ���߶�   ��ʽ������߶�=����߶�-�����+1
		conv_layers->input_feature_number		= temp_layers->feature_number;//��һ���������ݸ���Ϊ������������ݸ���
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//���ǰ��һ���ǳػ���
		user_cnn_pool_layers	*temp_layers	= (user_cnn_pool_layers *)last_layers->content;//ת������
		conv_layers->feature_width				= temp_layers->feature_width - kernel_width + 1;//���㵱ǰ������ݵĶ����� ��ʽ������߶�=����߶�-�����+1
		conv_layers->feature_height				= temp_layers->feature_height - kernel_height + 1;//���㵱ǰ������ݶ���߶�   ��ʽ������߶�=����߶�-�����+1
		conv_layers->input_feature_number		= temp_layers->feature_number;//��һ���������ݸ���Ϊ������������ݸ���
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers	= (user_cnn_conv_layers *)last_layers->content;
		conv_layers->feature_width				= temp_layers->feature_width - kernel_width + 1;//���㵱ǰ������ݵĶ����� ��ʽ������߶�=����߶�-�����+1
		conv_layers->feature_height				= temp_layers->feature_height - kernel_height + 1;//���㵱ǰ������ݶ���߶�   ��ʽ������߶�=����߶�-�����+1
		conv_layers->input_feature_number		= temp_layers->feature_number;//��һ���������ݸ���Ϊ������������ݸ���
	}
	else{
		return NULL;
	}

	conv_layers->feature_number			= feature_number;	//����������ݸ���
	conv_layers->kernel_width			= kernel_width;	//����Ϊ�����  ���þ���˵Ŀ��
	conv_layers->kernel_height			= kernel_height;//����Ϊ�����  ���þ���˵ĸ߶�
	conv_layers->biases_matrix			= user_nn_matrix_create(1, conv_layers->feature_number);//��ӱ����ƫ�ò��� ����������������ݸ���һ��
	conv_layers->feature_matrices		= user_nn_matrices_create(1, conv_layers->feature_number, conv_layers->feature_width, conv_layers->feature_height);//�������汾����������ݾ��� ������������������ݵĸ���
	conv_layers->kernel_matrices		= user_nn_matrices_create(conv_layers->input_feature_number, conv_layers->feature_number, conv_layers->kernel_width, conv_layers->kernel_height);//��������ˣ�����ÿ�����������������һ����������ݶ���һ����Ӧ����ˣ�����������Ǳ���������*�ϲ�������������������ѱ���������Ϊ����������У��ϲ���Ϊ�н��д�����
	conv_layers->deltas_matrices		= user_nn_matrices_create(1, conv_layers->feature_number, conv_layers->feature_width, conv_layers->feature_height);//��������Ĳв� ��С�뱾�������һ��
	conv_layers->deltas_kernel_matrices = user_nn_matrices_create(conv_layers->input_feature_number, conv_layers->feature_number, conv_layers->kernel_width, conv_layers->kernel_height);//�в�Ծ���˵ĵ������в��ǰһ���������ݵľ�����
	conv_layers->deltas_biases_matrix	= user_nn_matrix_create(1, conv_layers->feature_number);//��Ӳв���� 

	//������������������ = (float)conv->outputmaps * conv->kernel_width * conv->kernel_height;//���������ܲ�����С  --- ���ڳ�ʼ�������
	//������������������ = (float)conv->inputmaps  * conv->kernel_width * conv->kernel_height;//����������ܲ�����С  --- ���ڳ�ʼ�������
	//�򻯺�(conv->outputmaps + conv->inputmaps) * conv->kernel_width * conv->kernel_height
	user_nn_matrices_init_vaule(conv_layers->kernel_matrices, conv_layers->feature_number, conv_layers->input_feature_number*conv_layers->kernel_width*conv_layers->kernel_height);//�Ծ���˽��г�ʼ��

	return conv_layers;
}

//�����ػ���
//����
//scale���ػ���С
//���� 
user_cnn_pool_layers *user_cnn_layers_pooling_create(user_cnn_layers *cnn_layers, int kernel_width, int kernel_height, user_nn_pooling_type pool_type){
	user_cnn_layers		 *last_layers = cnn_layers;
	user_cnn_pool_layers *pool_layers = NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//��ѯ����cnn_layers�ն���
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_pool, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_cnn_pool_layers));//���������ڴ��������ռ�
	pool_layers = (user_cnn_pool_layers *)last_layers->next->content;//�����������ȡ

	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers	= (user_cnn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		pool_layers->feature_width				= temp_layers->feature_width / kernel_width;//
		pool_layers->feature_height				= temp_layers->feature_height / kernel_height;//
		pool_layers->input_feature_number		= temp_layers->feature_number;//��һ���������ݸ���Ϊ������������ݸ���
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//���ǰ��һ���ǳػ���
		user_cnn_pool_layers	*temp_layers	= (user_cnn_pool_layers *)last_layers->content;//ת������
		pool_layers->feature_width				= temp_layers->feature_width / kernel_width;//
		pool_layers->feature_height				= temp_layers->feature_height / kernel_height;//
		pool_layers->input_feature_number		= temp_layers->feature_number;//��һ���������ݸ���Ϊ������������ݸ���
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers	= (user_cnn_conv_layers *)last_layers->content;
		pool_layers->feature_width				= temp_layers->feature_width / kernel_width;//
		pool_layers->feature_height				= temp_layers->feature_height / kernel_height;//
		pool_layers->input_feature_number		= temp_layers->feature_number;//��һ���������ݸ���Ϊ������������ݸ���
	}
	else{
		return NULL;
	}

	pool_layers->feature_number		= pool_layers->input_feature_number;//���������һ��
	pool_layers->pool_width			= kernel_width;//����˴�С
	pool_layers->pool_height		= kernel_height;//����˴�С
	pool_layers->pool_type			= pool_type;//�ػ���ʽ ƽ��ֵ ���ֵ
	pool_layers->kernel_matrix		= user_nn_matrix_create(pool_layers->pool_width, pool_layers->pool_height);//�����ػ���ľ���
	pool_layers->feature_matrices	= user_nn_matrices_create(1, pool_layers->feature_number, pool_layers->feature_width, pool_layers->feature_height);//����������������ݾ��� �ػ���������������Ϊ�����������ݴ�С
	pool_layers->deltas_matrices	= user_nn_matrices_create(1, pool_layers->feature_number, pool_layers->feature_width, pool_layers->feature_height);;//�в���Ҫ���򴫲����и�ֵ

	user_nn_matrix_memset(pool_layers->kernel_matrix, (float)1 / (pool_layers->pool_width * pool_layers->pool_height));//��ʼ��ֵ�� �ػ���������

	return pool_layers;
}
//����ȫ���Ӳ�
//����
//count���������
//���� �ɹ���ʧ��
user_cnn_full_layers *user_cnn_layers_fullconnect_create(user_cnn_layers *cnn_layers){
	user_cnn_layers			*last_layers = cnn_layers;
	user_cnn_full_layers	*full_layers = NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//��ѯ����cnn_layers�ն���
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_full, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_cnn_full_layers));//���������ڴ��������ռ�
	full_layers = (user_cnn_full_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ

	//ȫ����Ӧ������һ������������ܺ�
	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers = (user_cnn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		full_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		full_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//�ϲ������������浽����
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//���ǰ��һ���ǳػ���
		user_cnn_pool_layers	*temp_layers = (user_cnn_pool_layers *)last_layers->content;//ת������
		full_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		full_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//�ϲ������������浽����
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers = (user_cnn_conv_layers *)last_layers->content;
		full_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		full_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//�ϲ������������浽����
	}
	else{
		return NULL;
	}
	//����ȫ���Ӳ��ƫ�ò���
	full_layers->biases_matrix			= user_nn_matrix_create(1, full_layers->feature_number);//���N��ƫ�ò��� ����ʹ��softmat�ع��ƫ�ò���
	full_layers->feature_matrix			= user_nn_matrix_create(1, full_layers->feature_number);//�������ֵ �������
	full_layers->kernel_matrix			= user_nn_matrix_create(full_layers->feature_number, full_layers->feature_number);//ȫ���Ӳ��Ȩ��ֵ
	full_layers->deltas_matrix			= user_nn_matrix_create(1, full_layers->feature_number);//����в�
	full_layers->deltas_kernel_matrix	= user_nn_matrix_create(full_layers->feature_number, full_layers->feature_number);//����в���ϲ�ľ�������W

	user_nn_matrix_init_vaule(full_layers->kernel_matrix, full_layers->feature_number, full_layers->feature_number);//��ʼ��ȫ���ӵ�Ȩ��ֵ

	return full_layers;
}

//���������
//����
//count���������
//���� �ɹ���ʧ��
user_cnn_output_layers *user_cnn_layers_output_create(user_cnn_layers *cnn_layers, int class_number){
	user_cnn_layers			*last_layers = cnn_layers;
	user_cnn_output_layers	*output_layers	= NULL;

	while (last_layers->next != NULL){
		last_layers = last_layers->next;//��ѯ����cnn_layers�ն���
	}
	last_layers->next = user_cnn_layers_create(u_cnn_layer_type_output, last_layers->index + 1);//��������� ������ָ��Ϊǰһ��+1
	last_layers->next->prior = last_layers;//ָ��ǰһ��
	last_layers->next->content = malloc(sizeof(user_cnn_output_layers));//���������ڴ��������ռ�
	output_layers = (user_cnn_output_layers *)last_layers->next->content;//��ȫ���Ӳ�����ȡ

	//�����Ϊȫ���Ӳ���ô��������������ܸ���Ϊǰһ�����������Ԫ���ܺ�
	if (last_layers->type == u_cnn_layer_type_input){
		user_cnn_input_layers	*temp_layers = (user_cnn_input_layers *)last_layers->content;//��ȡ��һ�� ������ֵ
		output_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//�ϲ������������浽����
	}
	else if (last_layers->type == u_cnn_layer_type_pool){//���ǰ��һ���ǳػ���
		user_cnn_pool_layers	*temp_layers = (user_cnn_pool_layers *)last_layers->content;//ת������
		output_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//�ϲ������������浽����
	}
	else if (last_layers->type == u_cnn_layer_type_conv){
		user_cnn_conv_layers	*temp_layers = (user_cnn_conv_layers *)last_layers->content;
		output_layers->feature_number = temp_layers->feature_width * temp_layers->feature_height * temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number * temp_layers->feature_width * temp_layers->feature_height);//�ϲ������������浽����
	}
	else if (last_layers->type == u_cnn_layer_type_full){
		user_cnn_full_layers	*temp_layers = (user_cnn_full_layers *)last_layers->content;
		output_layers->feature_number = temp_layers->feature_number;
		output_layers->input_feature_matrix = user_nn_matrix_create(1, temp_layers->feature_number);//�ϲ������������浽����
	}
	else{
		return NULL;
	}

	output_layers->class_number			= class_number;//�������
	output_layers->loss_function		= 0.0f;//���ۺ���
	//����ȫ���Ӳ��ƫ�ò���
	output_layers->biases_matrix		= user_nn_matrix_create(1, output_layers->class_number);//���N��ƫ�ò��� ����ʹ��softmat�ع��ƫ�ò���
	output_layers->feature_matrix		= user_nn_matrix_create(1, output_layers->class_number);//�������ֵ �������
	output_layers->kernel_matrix		= user_nn_matrix_create(output_layers->feature_number, output_layers->class_number);//����������kenerlģ��
	output_layers->error_matrix			= user_nn_matrix_create(1, output_layers->class_number);//����ֵ�������
	output_layers->target_matrix		= user_nn_matrix_create(1, output_layers->class_number);//����ֵ�������
	output_layers->deltas_matrix		= user_nn_matrix_create(1, output_layers->class_number);//����в�
	output_layers->deltas_kernel_matrix = user_nn_matrix_create(output_layers->feature_number, output_layers->class_number);//����в���ϲ�ľ�������W

	user_nn_matrix_init_vaule(output_layers->kernel_matrix, output_layers->feature_number, output_layers->class_number);//��ʼ�������kernelģ��
	
	return output_layers;
}

