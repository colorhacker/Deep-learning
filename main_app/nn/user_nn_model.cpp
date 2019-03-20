
#include "user_nn_model.h"
#include "user_nn_ffp.h"
#include "user_nn_bp.h"
#include "user_nn_grads.h"

//ͨ����������Ϣ����ģ��
//layer_infor ����ģ�Ͳ���
//���� ������������ģ�Ͷ���
user_nn_layers *user_nn_model_create(int *layer_infor) {
	user_nn_layers			*nn_layers = NULL;
	nn_layers = user_nn_layers_create(u_nn_layer_type_null, 0);//����һ���ղ�

	while (1) {
		switch (*layer_infor) {
		case 'i':
			user_nn_layers_input_create(nn_layers, *(layer_infor + 1), *(layer_infor + 2));	//�����
			layer_infor += 3;
			break;
		case 'h':
			user_nn_layers_hidden_create(nn_layers, *(layer_infor + 1));//������
			layer_infor += 2;
			break;
		case 'o':
			user_nn_layers_output_create(nn_layers, *(layer_infor + 1));//�����
			layer_infor += 2;
			return nn_layers;
		default:
			printf("set error\n"); while (1);
			break;
		}
	}
	return NULL;
}
//�����������ݵ�ָ������������������
//layers ���ض����
//src_matrix Ŀ������
//���� ��
void user_nn_model_load_input_feature(user_nn_layers *layers, user_nn_matrix *src_matrix) {
	user_nn_layers *nn_input_layer = user_nn_layers_get(layers, 1);//��ȡ�����
	user_nn_matrix_cpy_matrix(((user_nn_input_layers *)nn_input_layer->content)->feature_matrix, src_matrix);
}
//�����������ݵ�ָ������������������
//layers ���ض����
//src_matrix Ŀ������
//���� ��
void user_nn_model_load_target_feature(user_nn_layers *layers, user_nn_matrix *src_matrix) {
	user_nn_layers *nn_output_layer = user_nn_model_return_layer(layers, u_nn_layer_type_output);//��ȡ�����
	user_nn_matrix_cpy_matrix(((user_nn_output_layers *)nn_output_layer->content)->target_matrix, src_matrix);
}
//����ִ��һ�ε���
//layers �������Ĳ�
//����ֵ ��
void user_nn_model_ffp(user_nn_layers *layers) {
	while (1) {
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			break;
		case u_nn_layer_type_hidden:
			user_nn_ffp_hidden(layers->prior, layers);//�Ӳ�������
			break;
		case u_nn_layer_type_output:
			user_nn_ffp_output(layers->prior, layers);//��������
			break;
		default:
			break;
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
}

//���򴫲�һ��
//layers������ʼλ��
//index����ǰ��ǩλ��
//alpha������ϵ��
//����ֵ����
void user_nn_model_bp(user_nn_layers *layers, float alpha) {
	//ȡ��ָ�����һ������ָ��
	while (layers->next != NULL) {
		layers = layers->next;
	}
	//�������в�
	while (1) {
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			break;
		case u_nn_layer_type_hidden:
			user_nn_bp_hidden_back_prior(layers->prior, layers);
			break;
		case u_nn_layer_type_output:
			user_nn_bp_output_back_prior(layers->prior, layers);
			break;
		default:
			break;
		}
		if (layers->prior == NULL) {
			break;
		}
		else {
			layers = layers->prior;
		}
	}
	//���Ȩ�زв�ֵ
	while (1) {
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			break;
		case u_nn_layer_type_hidden:
			user_nn_grads_hidden(layers, alpha);//����Ȩ��
			break;
		case u_nn_layer_type_output:
			user_nn_grads_output(layers, alpha);//����Ȩ��
			break;
		default:
			break;
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
}
//��ȡloss��ʧֵ
//layers ��ȡ�����
//���� ��ʧֵ�Ĵ�С
float user_nn_model_return_loss(user_nn_layers *layers) {
	static float loss_function = 0;//ȫ�ֱ�����lossֵ
	while (1) {
		if (layers->type == u_nn_layer_type_output) {
			if (loss_function == 0) {
				loss_function = ((user_nn_output_layers *)layers->content)->loss_function;
			}
			else {
				loss_function = (float)0.99f * loss_function + 0.01f * ((user_nn_output_layers *)layers->content)->loss_function;
			}
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return loss_function;
}

//�����������л�ȡһ��ָ���� ��˳�����
//layers ���ҵĶ����
//type Ŀ�������
//���� ��������
user_nn_layers *user_nn_model_return_layer(user_nn_layers *layers, user_nn_layer_type type) {
	while (1) {
		if (layers->type == type) {
			return layers;//�������ֵ
		}
		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
	return NULL;
}
//��ȡ�������
//layers ��ȡ�����
//���� ����ֵ
user_nn_matrix *user_nn_model_return_result(user_nn_layers *layers) {
	return ((user_nn_output_layers *)user_nn_model_return_layer(layers, u_nn_layer_type_output)->content)->feature_matrix;
}

//��ʾһ��������
//window_name ��������
//src_matrices ��������Ķ���
//gain �Ŵ���
//���� ��
void user_nn_model_display_matrix(char *window_name, user_nn_matrix  *src_matrix, int gain) {
	user_nn_matrix *dest_matrix = user_nn_matrix_expand_mult_constant(src_matrix, gain, gain, (float)255);//���зŴ���
	CvSize cvsize = { dest_matrix->width, dest_matrix->height };
	IplImage *dest_image = cvCreateImage(cvsize, IPL_DEPTH_8U, 1);
	user_nn_matrix_uchar_memcpy((unsigned char *)dest_image->imageData, dest_matrix);//����ͼ������
	cvShowImage(window_name, dest_image);//��ʾͼ��
	cvWaitKey(1);
	cvReleaseImage(&dest_image);//�ͷ��ڴ�
	user_nn_matrix_delete(dest_matrix);//ɾ������
}
void user_nn_model_display_feature(user_nn_layers *layers) {
	static int create_flags = 0;
	char windows_name[128];

	if (create_flags == 0) {
		create_flags = 1;
	}
	while (1) {
		memset(windows_name, 0, sizeof(windows_name));
		switch (layers->type) {
		case u_nn_layer_type_null:
			break;
		case u_nn_layer_type_input:
			sprintf(windows_name, "input%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_input_layers  *)layers->content)->feature_matrix, 2);//��ʾ��ָ������
			break;
		case u_nn_layer_type_hidden:
			sprintf(windows_name, "hidden%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_hidden_layers  *)layers->content)->feature_matrix, 2);//��ʾ��ָ������
			break;
		case u_nn_layer_type_output:
			sprintf(windows_name, "output%d", layers->index);
			user_nn_model_display_matrix(windows_name, ((user_nn_output_layers  *)layers->content)->feature_matrix, 2);//��ʾ��ָ������
			break;
		default:
			break;
		}

		if (layers->next == NULL) {
			break;
		}
		else {
			layers = layers->next;
		}
	}
}