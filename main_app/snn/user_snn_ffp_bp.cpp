
#include "user_snn_ffp_bp.h"

//intput->hidden->output
//hidden=tanh(input*w1+hidden_t*w2+bias1)
//
void user_snn_ffp_hidden(user_snn_layers *prior_layer, user_snn_layers *hidden_layer) {
	user_snn_output_layers *hidden_layers = (user_snn_output_layers  *)hidden_layer->content;//��ȡ����Ĳ���
	user_nn_matrix *input_feature_matrix = NULL;
	//��ȡǰһ�������
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_output_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else {
		return;
	}
	user_nn_matrix_memset(hidden_layers->feature_matrix, 0.0f);
	user_nn_matrix_thred_acc(input_feature_matrix, hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix, hidden_layers->feature_matrix);//����ֵ
	user_snn_data_softmax(hidden_layers->feature_matrix);//���ݹ�һ������
}

void user_snn_ffp_output(user_snn_layers *prior_layer, user_snn_layers *output_layer) {
	user_snn_output_layers *output_layers = (user_snn_output_layers  *)output_layer->content;//��ȡ����Ĳ���
	user_nn_matrix *input_feature_matrix = NULL;
	//��ȡǰһ�������
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_output_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else {
		return;
	}
	user_nn_matrix_memset(output_layers->feature_matrix,0.0f);//�����������
	user_nn_matrix_thred_acc(input_feature_matrix, output_layers->min_kernel_matrix, output_layers->max_kernel_matrix, output_layers->feature_matrix);//����ֵ
	user_snn_data_softmax(output_layers->feature_matrix);//���ݹ�һ������
}

//���򴫲�������ȡ�����ݶ�ֵ
//
void user_snn_bp_output_back_prior(user_snn_layers *prior_layer, user_snn_layers *output_layer) {
	user_nn_matrix   *input_feature_matrix = NULL;//
	user_nn_matrix   *input_thred_matrix = NULL;//
	user_snn_output_layers  *output_layers = (user_snn_output_layers  *)output_layer->content;//��ȡ����ػ�������
	//��ȡǰһ�������
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//ת����������
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_thred_matrix = ((user_snn_hidden_layers *)prior_layer->content)->thred_matrix;//ת����������
	}
	else {
		return ;
	}

	user_nn_matrix_cpy_matrix(input_thred_matrix, input_feature_matrix);//����ֵ
	//user_nn_matrix_memset(input_thred_matrix,0.0f);
	//
	//**** ʹ�ù�һ��������δ��һ����������ݽ�����ֵ����
	user_nn_matrix_thred_process(output_layers->thred_matrix,output_layers->feature_matrix, output_layers->target_matrix);//�������ֵ�仯����
	user_nn_matrix_update_thred(input_feature_matrix, input_thred_matrix, output_layers->min_kernel_matrix, output_layers->max_kernel_matrix, output_layers->thred_matrix, snn_avg_vaule, snn_step_vaule);//������ֵ
	user_snn_data_softmax(input_thred_matrix);
	//user_nn_matrix_printf(NULL, input_feature_matrix);
	//user_nn_matrix_printf(NULL, input_thred_matrix);
	user_nn_matrix_thred_process(input_thred_matrix, input_feature_matrix, input_thred_matrix);//�����ǰһ����ֵ�仯����
	//user_nn_matrix_printf(NULL, input_thred_matrix);
}

//���򴫲�������ȡ�����ݶ�ֵ
void user_snn_bp_hidden_back_prior(user_snn_layers *prior_layer, user_snn_layers *hidden_layer) {
	user_snn_hidden_layers  *hidden_layers = (user_snn_hidden_layers  *)hidden_layer->content;//��ȡ����ػ�������
	user_nn_matrix	 *input_feature_matrix = NULL;
	user_nn_matrix   *input_thred_matrix = NULL;//
	//��ȡǰһ�������
	if (prior_layer->type == u_snn_layer_type_input) {
		input_feature_matrix = ((user_snn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//ת����������
	}
	else if (prior_layer->type == u_snn_layer_type_hidden) {
		input_feature_matrix = ((user_snn_hidden_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_thred_matrix = ((user_snn_input_layers *)prior_layer->content)->thred_matrix;//ת����������
	}
	else {
		return;
	}
	user_nn_matrix_cpy_matrix(input_thred_matrix, input_feature_matrix);//����ֵ
	//user_nn_matrix_memset(input_thred_matrix, 0.0f);
	user_nn_matrix_update_thred(input_feature_matrix, input_thred_matrix, hidden_layers->min_kernel_matrix, hidden_layers->max_kernel_matrix, hidden_layers->thred_matrix, snn_avg_vaule, snn_step_vaule);//������ֵ
	user_snn_data_softmax(input_thred_matrix);
	user_nn_matrix_thred_process(input_thred_matrix, input_feature_matrix, input_thred_matrix);//�������ֵ�仯����
}

