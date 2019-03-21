
#include "user_rnn_bptt.h"


//���򴫲�������ȡ�����ݶ�ֵ
//
float user_rnn_bp_output_back_prior(user_rnn_layers *prior_layer, user_rnn_layers *output_layer) {
	user_nn_list_matrix   *input_feature_matrices = NULL;//����������������
	user_nn_list_matrix	  *input_deltas_matrices = NULL;
	user_rnn_output_layers  *output_layers = (user_rnn_output_layers  *)output_layer->content;//��ȡ����ػ�������
	user_nn_matrix *target_feature_matrix = NULL;
	user_nn_matrix *output_feature_matrix = NULL;
	user_nn_matrix *error_matrix = NULL;

	user_nn_matrix *deltas_biase_matrix = NULL;
	user_nn_matrix *deltas_kernel_matrix = NULL;
	user_nn_matrix *deltas_matrix = NULL;//�в�����
	user_nn_matrix *input_feature_matrix = NULL;//��ȡ��������

	user_nn_matrix *deltas_kernel_matrix_temp = NULL;

	user_nn_matrix *output_kernel_matrix = NULL;
	user_nn_matrix *input_deltas_matrix = NULL;
	user_nn_matrix *input_deltas_matrix_temp = NULL;
	float *loss_vaule = NULL;
	int time_index = 0;
	//��ȡǰһ�������
	if (prior_layer->type == u_rnn_layer_type_input) {
		input_feature_matrices = ((user_rnn_input_layers *)prior_layer->content)->feature_matrices;//ת����������
		input_deltas_matrices = ((user_rnn_input_layers *)prior_layer->content)->deltas_matrices;
	}
	else if (prior_layer->type == u_rnn_layer_type_hidden) {
		input_feature_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->feature_matrices;//ת����������
		input_deltas_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->deltas_matrices;//��ȡ��һ��Ĳв�Ȩ��
	}
	else {
		return 0;
	}

	output_layers->loss_function = 0;//��ʧ��������
	for (time_index = output_layers->time_number - 1; time_index >= 0; time_index--) {
		error_matrix = output_layers->error_matrix;//��ȡ�������
		target_feature_matrix = user_nn_matrices_ext_matrix_index(output_layers->target_matrices, time_index);//��ȡĿ�����ֵ
		output_feature_matrix = user_nn_matrices_ext_matrix_index(output_layers->feature_matrices,time_index);//��ȡ��������
		loss_vaule = &output_layers->loss_function;//��ȡ��ʧ����
		//(Lo, Er) = output_loss_error(Oh, Ta)
		user_nn_matrix_cum_matrix_mult_alpha(error_matrix, output_feature_matrix, target_feature_matrix, -1.0f);//����
		*loss_vaule = *loss_vaule + user_nn_matrix_get_rmse(error_matrix);//��ȡ�������
		//dOh=np.multiply(Er,act_function_d(Oh))
		deltas_matrix = user_nn_matrices_ext_matrix_index(output_layers->deltas_matrices, time_index);//��ȡ�в����
		user_nn_activate_matrix_d(output_feature_matrix, user_nn_rnn_softmax);//���������
		user_nn_matrix_poit_mult_matrix(deltas_matrix, error_matrix, output_feature_matrix);//�в�=�������*����ֵ
		//��Wo=��Wo+np.dot(dOh,Hi.T)
		deltas_kernel_matrix = output_layers->deltas_kernel_matrix;//�в����
		input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, time_index);//��ȡ��������
		user_nn_matrix_transpose(input_feature_matrix);//����ת��
		deltas_kernel_matrix_temp = user_nn_matrix_mult_matrix(deltas_matrix, input_feature_matrix);//����˷�
		user_nn_matrix_transpose(input_feature_matrix);//����ת��
		user_nn_matrix_cum_matrix(deltas_kernel_matrix, deltas_kernel_matrix, deltas_kernel_matrix_temp);//�ۼ�Ȩ�ظ���ֵ ��Wo
		//��bo=��bo+dOh
		deltas_biase_matrix = output_layers->deltas_biases_matrix;
		user_nn_matrix_cum_matrix(deltas_biase_matrix, deltas_biase_matrix, deltas_matrix);//�ۼ�ƫ�ò�������ֵ��bo
		//dHh=np.dot(Wo.T,dOh)#�в��ǰһ��
		output_kernel_matrix = output_layers->kernel_matrix;//��ȡȨֵ����
		input_deltas_matrix = user_nn_matrices_ext_matrix_index(input_deltas_matrices, time_index);//��ȡ�в�ָ��
		user_nn_matrix_transpose(output_kernel_matrix);//����ת��
		input_deltas_matrix_temp = user_nn_matrix_mult_matrix(output_kernel_matrix, deltas_matrix);//����˷�
		user_nn_matrix_transpose(output_kernel_matrix);//����ת��
		user_nn_matrix_cpy_matrix(input_deltas_matrix, input_deltas_matrix_temp);//����в�

		user_nn_matrix_delete(deltas_kernel_matrix_temp);//ɾ������
		user_nn_matrix_delete(input_deltas_matrix_temp);//ɾ������
	}
	return *loss_vaule;
}

//���򴫲�������ȡ�����ݶ�ֵ
void user_rnn_bp_hidden_back_prior(user_rnn_layers *prior_layer, user_rnn_layers *hidden_layer) {
	user_nn_list_matrix   *input_feature_matrices = NULL;//����������������
	user_nn_list_matrix	  *input_deltas_matrices = NULL;
	user_rnn_hidden_layers  *hidden_layers = (user_rnn_hidden_layers  *)hidden_layer->content;//��ȡ����ػ�������
	user_nn_matrix *hidden_deltas_matrix = NULL;
	user_nn_matrix *hidden_feature_matrix = NULL;
	user_nn_matrix *hidden_deltas_matrix_t = NULL;

	user_nn_matrix *deltas_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *deltas_kernel_matrix_temp = NULL;

	user_nn_matrix *deltas_kernel_matrix_t = NULL;
	user_nn_matrix *input_feature_matrix_t = NULL;
	user_nn_matrix *deltas_kernel_matrix_t_temp = NULL;

	user_nn_matrix *deltas_biases_matrix = NULL;

	user_nn_matrix *kernel_matrix_t = NULL;
	user_nn_matrix *hidden_deltas_matrix_t_temp = NULL;

	user_nn_matrix *kernel_matrix = NULL;
	user_nn_matrix *input_deltas_matrix = NULL;
	user_nn_matrix *input_deltas_matrix_temp = NULL;
	int time_index = 0;
	//��ȡǰһ�������
	if (prior_layer->type == u_rnn_layer_type_input) {
		input_feature_matrices = ((user_rnn_input_layers *)prior_layer->content)->feature_matrices;//ת����������
		input_deltas_matrices = ((user_rnn_input_layers *)prior_layer->content)->deltas_matrices;
	}
	else if (prior_layer->type == u_rnn_layer_type_hidden) {
		input_feature_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->feature_matrices;//ת����������
		input_deltas_matrices = ((user_rnn_hidden_layers *)prior_layer->content)->deltas_matrices;//��ȡ��һ��Ĳв�Ȩ��
	}
	else {
		return;
	}

	for (time_index = hidden_layers->time_number - 1;time_index >= 0; time_index--) {
		hidden_deltas_matrix = user_nn_matrices_ext_matrix_index(hidden_layers->deltas_matrices, time_index);//��ȡ�в����
		hidden_feature_matrix= user_nn_matrices_ext_matrix_index(hidden_layers->feature_matrices, time_index);//��ȡ�������
		hidden_deltas_matrix_t = hidden_layers->deltas_matrix_t;
		//dHh=np.multiply(dOh,act_function_d(Hi))+dHt_1
		user_nn_activate_matrix_d(hidden_feature_matrix, user_nn_rnn_softmax);//���������
		user_nn_matrix_poit_mult_matrix(hidden_deltas_matrix, hidden_deltas_matrix, hidden_feature_matrix);//�в�=�������*����ֵ
		user_nn_matrix_cum_matrix(hidden_deltas_matrix, hidden_deltas_matrix, hidden_deltas_matrix_t);//�õ������ܲв�ֵ
		//��Wi=��Wi+np.dot(dHh,i.T)
		deltas_kernel_matrix = hidden_layers->deltas_kernel_matrix;//��ȡ�в����
		input_feature_matrix = user_nn_matrices_ext_matrix_index(input_feature_matrices, time_index);//��ȡ������������
		user_nn_matrix_transpose(input_feature_matrix);//����ת��
		deltas_kernel_matrix_temp = user_nn_matrix_mult_matrix(hidden_deltas_matrix, input_feature_matrix);//����˷�
		user_nn_matrix_transpose(input_feature_matrix);//����ת��
		user_nn_matrix_cum_matrix(deltas_kernel_matrix, deltas_kernel_matrix, deltas_kernel_matrix_temp);//�ۼ�����㵽������Ĳв�仯ֵ
		//��Wh = ��Wh + np.dot(dHh, dHt_1.T)
		deltas_kernel_matrix_t = hidden_layers->deltas_kernel_matrix_t;//��ȡ�в����
		user_nn_matrix_transpose(hidden_deltas_matrix_t);//����ת��
		deltas_kernel_matrix_t_temp = user_nn_matrix_mult_matrix(hidden_deltas_matrix, hidden_deltas_matrix_t);//����˷�
		user_nn_matrix_transpose(hidden_deltas_matrix_t);//����ת��
		user_nn_matrix_cum_matrix(deltas_kernel_matrix_t, deltas_kernel_matrix_t, deltas_kernel_matrix_t_temp);//�ۼ�����㵽������Ĳв�仯ֵ
		//��bh=��bh+dHh
		deltas_biases_matrix = hidden_layers->deltas_biases_matrix;
		user_nn_matrix_cum_matrix(deltas_biases_matrix, deltas_biases_matrix, hidden_deltas_matrix);//�ۼ�ƫ�ò����в�仯ֵ
		//dHt_1=np.dot(Wh.T,dHh)
		kernel_matrix_t = hidden_layers->kernel_matrix_t;
		user_nn_matrix_transpose(kernel_matrix_t);//����ת��
		hidden_deltas_matrix_t_temp = user_nn_matrix_mult_matrix(kernel_matrix_t, hidden_deltas_matrix);//����˷�
		user_nn_matrix_transpose(kernel_matrix_t);//����ת��
		user_nn_matrix_cpy_matrix(hidden_deltas_matrix_t, hidden_deltas_matrix_t_temp);//����в�
		//dIi = np.dot(Wi.T, dHh)#�в��ǰһ��
		kernel_matrix = hidden_layers->kernel_matrix;
		input_deltas_matrix = user_nn_matrices_ext_matrix_index(input_deltas_matrices, time_index);//��ȡ����в����
		user_nn_matrix_transpose(kernel_matrix);//����ת��
		input_deltas_matrix_temp = user_nn_matrix_mult_matrix(kernel_matrix, hidden_deltas_matrix);//����˷�
		user_nn_matrix_transpose(kernel_matrix);//����ת��
		user_nn_matrix_cpy_matrix(input_deltas_matrix, input_deltas_matrix_temp);//����в�

		user_nn_matrix_delete(deltas_kernel_matrix_temp);//ɾ������
		user_nn_matrix_delete(deltas_kernel_matrix_t_temp);//ɾ������
		user_nn_matrix_delete(hidden_deltas_matrix_t_temp);//ɾ������
		user_nn_matrix_delete(input_deltas_matrix_temp);//ɾ������
	}
}

