
#include "user_nn_ffp.h"

//intput->hidden->output
//hidden=tanh(input*w1+hidden_t*w2+bias1)
//
void user_nn_ffp_hidden(user_nn_layers *prior_layer, user_nn_layers *hidden_layer) {
	user_nn_hidden_layers *hidden_layers = (user_nn_hidden_layers  *)hidden_layer->content;//��ȡ����Ĳ���
	user_nn_matrix *hidden_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *hidden_bias_matrix = NULL;
	user_nn_matrix *hidden_feature_matrix = NULL;

	user_nn_matrix *intput_to_hidden_feature = NULL;

	//��ȡǰһ�������
	if (prior_layer->type == u_nn_layer_type_input) {
		input_feature_matrix = ((user_nn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else if (prior_layer->type == u_nn_layer_type_hidden) {
		input_feature_matrix = ((user_nn_hidden_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else {
		return;
	}
	
	hidden_feature_matrix = hidden_layers->feature_matrix;//��������	
	hidden_kernel_matrix = hidden_layers->kernel_matrix;//��ȡǰһ����赼�������ݵ�Ȩ��
	hidden_bias_matrix = hidden_layers->biases_matrix;//��ȡƫ�ò�������
	//Hi=act_function(np.dot(Wi,i)+bh)	
	intput_to_hidden_feature = user_nn_matrix_mult_matrix(hidden_kernel_matrix, input_feature_matrix);//np.dot(Wi,i)
#if user_nn_use_bias
	user_nn_matrix_cum_matrix(hidden_feature_matrix, intput_to_hidden_feature, hidden_bias_matrix);//Ȩ�����+bh����ƫ�ò���
	user_nn_activate_matrix(hidden_feature_matrix, user_nn_nn_softmax);//���ü�������м���
#else
	user_nn_matrix_cpy_matrix(hidden_feature_matrix, intput_to_hidden_feature);//��������
	user_nn_activate_matrix(hidden_feature_matrix, user_nn_nn_softmax);//���ü�������м���
#endif // user_nn_use_bias


	user_nn_matrix_delete(intput_to_hidden_feature);//ɾ������

}
//intput->hidden->output
//output=tanh(hidden*w2+bias2)
//
void user_nn_ffp_output(user_nn_layers *prior_layer, user_nn_layers *output_layer) {
	user_nn_output_layers *output_layers = (user_nn_output_layers  *)output_layer->content;//��ȡ����Ĳ���
	user_nn_matrix *output_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *output_bias_matrix = NULL;
	user_nn_matrix *output_feature_matrix = NULL;

	user_nn_matrix *intput_to_output_feature = NULL;
	int time_index = 0;

	//��ȡǰһ�������
	if (prior_layer->type == u_nn_layer_type_input) {
		input_feature_matrix = ((user_nn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else if (prior_layer->type == u_nn_layer_type_hidden) {
		input_feature_matrix = ((user_nn_hidden_layers *)prior_layer->content)->feature_matrix;//ת����������
	}
	else {
		return;
	}

	output_kernel_matrix = output_layers->kernel_matrix;//ȡ��Ȩ��
	output_bias_matrix = output_layers->biases_matrix;//ȡ��ƫ�ò���
	output_feature_matrix = output_layers->feature_matrix;//��ȡ�������
	//Oh=act_function(np.dot(Wo,Hi)+bo)
	intput_to_output_feature = user_nn_matrix_mult_matrix(output_kernel_matrix, input_feature_matrix);//np.dot(Wo,Hi)	
#if user_nn_use_bias	
	user_nn_matrix_cum_matrix(output_feature_matrix, intput_to_output_feature, output_bias_matrix);//+bo����ƫ�ò���
	user_nn_activate_matrix(output_feature_matrix, user_nn_nn_softmax);//���ݽ��м���,���汾��������������
#else
	user_nn_matrix_cpy_matrix(output_feature_matrix, intput_to_output_feature);//��������
	user_nn_activate_matrix(output_feature_matrix, user_nn_nn_softmax);//���ü�������м���
#endif // user_nn_use_bias

	user_nn_matrix_delete(intput_to_output_feature);//ɾ����ʱ����	

}