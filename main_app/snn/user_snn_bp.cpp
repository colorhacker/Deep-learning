
#include "user_nn_bp.h"


//���򴫲�������ȡ�����ݶ�ֵ
//
float user_nn_bp_output_back_prior(user_nn_layers *prior_layer, user_nn_layers *output_layer) {
	user_nn_matrix   *input_feature_matrix = NULL;//��ȡ��������
	user_nn_matrix	 *input_deltas_matrix = NULL;
	user_nn_output_layers  *output_layers = (user_nn_output_layers  *)output_layer->content;//��ȡ����ػ�������
	user_nn_matrix *target_feature_matrix = NULL;
	user_nn_matrix *output_feature_matrix = NULL;
	user_nn_matrix *error_matrix = NULL;

	user_nn_matrix *deltas_biase_matrix = NULL;
	user_nn_matrix *deltas_kernel_matrix = NULL;
	user_nn_matrix *deltas_matrix = NULL;//�в�����

	user_nn_matrix *deltas_kernel_matrix_temp = NULL;

	user_nn_matrix *output_kernel_matrix = NULL;
	user_nn_matrix *input_deltas_matrix_temp = NULL;

	//��ȡǰһ�������
	if (prior_layer->type == u_nn_layer_type_input) {
		input_feature_matrix = ((user_nn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_deltas_matrix = ((user_nn_input_layers *)prior_layer->content)->deltas_matrix;
	}
	else if (prior_layer->type == u_nn_layer_type_hidden) {
		input_feature_matrix = ((user_nn_hidden_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_deltas_matrix = ((user_nn_hidden_layers *)prior_layer->content)->deltas_matrix;//��ȡ��һ��Ĳв�Ȩ��
	}
	else {
		return 0;
	}

	error_matrix = output_layers->error_matrix;//��ȡ�������
	target_feature_matrix = output_layers->target_matrix;//��ȡĿ�����ֵ
	output_feature_matrix = output_layers->feature_matrix;//��ȡ��������

	//(Lo, Er) = output_loss_error(Oh, Ta)
	user_nn_matrix_cum_matrix_mult_alpha(error_matrix, output_feature_matrix, target_feature_matrix, -1.0f);//����
	output_layers->loss_function = user_nn_matrix_get_rmse(error_matrix);//��ȡ���������
	//dOh=np.multiply(Er,act_function_d(Oh))
	deltas_matrix = output_layers->deltas_matrix;//��ȡ�в����
	user_nn_activate_matrix_d(output_feature_matrix, user_nn_nn_softmax);//���������
	user_nn_matrix_poit_mult_matrix(deltas_matrix, error_matrix, output_feature_matrix);//�в�=�������*����ֵ
	//��Wo=np.dot(dOh,Hi.T)
	deltas_kernel_matrix = output_layers->deltas_kernel_matrix;//�в����
	user_nn_matrix_transpose(input_feature_matrix);//����ת��
	deltas_kernel_matrix_temp = user_nn_matrix_mult_matrix(deltas_matrix, input_feature_matrix);//����˷�
	user_nn_matrix_transpose(input_feature_matrix);//����ת��
	user_nn_matrix_cpy_matrix(deltas_kernel_matrix, deltas_kernel_matrix_temp);//�ۼ�Ȩ�ظ���ֵ ��Wo
#if user_nn_use_bias	
	//��bo=dOh
	deltas_biase_matrix = output_layers->deltas_biases_matrix;
	user_nn_matrix_cpy_matrix(deltas_biase_matrix, deltas_matrix);//�ۼ�ƫ�ò�������ֵ��bo
#endif
	//dHh=np.dot(Wo.T,dOh)#�в��ǰһ��
	output_kernel_matrix = output_layers->kernel_matrix;//��ȡȨֵ����
	user_nn_matrix_transpose(output_kernel_matrix);//����ת��
	input_deltas_matrix_temp = user_nn_matrix_mult_matrix(output_kernel_matrix, deltas_matrix);//����˷�
	user_nn_matrix_transpose(output_kernel_matrix);//����ת��
	user_nn_matrix_cpy_matrix(input_deltas_matrix, input_deltas_matrix_temp);//����в�

	user_nn_matrix_delete(deltas_kernel_matrix_temp);//ɾ������
	user_nn_matrix_delete(input_deltas_matrix_temp);//ɾ������
	
	return output_layers->loss_function;
}

//���򴫲�������ȡ�����ݶ�ֵ
void user_nn_bp_hidden_back_prior(user_nn_layers *prior_layer, user_nn_layers *hidden_layer) {
	user_nn_hidden_layers  *hidden_layers = (user_nn_hidden_layers  *)hidden_layer->content;//��ȡ����ػ�������
	user_nn_matrix *hidden_deltas_matrix = NULL;
	user_nn_matrix *hidden_feature_matrix = NULL;

	user_nn_matrix *deltas_kernel_matrix = NULL;
	user_nn_matrix *input_feature_matrix = NULL;
	user_nn_matrix *deltas_kernel_matrix_temp = NULL;
	
	user_nn_matrix *deltas_biases_matrix = NULL;

	user_nn_matrix *kernel_matrix = NULL;
	user_nn_matrix *input_deltas_matrix = NULL;
	user_nn_matrix *input_deltas_matrix_temp = NULL;

	//��ȡǰһ�������
	if (prior_layer->type == u_nn_layer_type_input) {
		input_feature_matrix = ((user_nn_input_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_deltas_matrix = ((user_nn_input_layers *)prior_layer->content)->deltas_matrix;
	}
	else if (prior_layer->type == u_nn_layer_type_hidden) {
		input_feature_matrix = ((user_nn_hidden_layers *)prior_layer->content)->feature_matrix;//ת����������
		input_deltas_matrix = ((user_nn_hidden_layers *)prior_layer->content)->deltas_matrix;//��ȡ��һ��Ĳв�Ȩ��
	}
	else {
		return;
	}

	hidden_deltas_matrix = hidden_layers->deltas_matrix;//��ȡ�в����
	hidden_feature_matrix= hidden_layers->feature_matrix;//��ȡ�������
	//dHh=np.multiply(dOh,act_function_d(Hi))+dHt_1
	user_nn_activate_matrix_d(hidden_feature_matrix, user_nn_nn_softmax);//���������
	user_nn_matrix_poit_mult_matrix(hidden_deltas_matrix, hidden_deltas_matrix, hidden_feature_matrix);//�в�=�������*����ֵ
	//��Wi=np.dot(dHh,i.T)
	deltas_kernel_matrix = hidden_layers->deltas_kernel_matrix;//��ȡ�в����
	user_nn_matrix_transpose(input_feature_matrix);//����ת��
	deltas_kernel_matrix_temp = user_nn_matrix_mult_matrix(hidden_deltas_matrix, input_feature_matrix);//����˷�
	user_nn_matrix_transpose(input_feature_matrix);//����ת��
	user_nn_matrix_cpy_matrix(deltas_kernel_matrix, deltas_kernel_matrix_temp);//�ۼ�����㵽������Ĳв�仯ֵ
#if user_nn_use_bias
	//��bh=dHh
	deltas_biases_matrix = hidden_layers->deltas_biases_matrix;
	user_nn_matrix_cpy_matrix(deltas_biases_matrix, hidden_deltas_matrix);//�ۼ�ƫ�ò����в�仯ֵ
#endif
	//dIi = np.dot(Wi.T, dHh)#�в��ǰһ��
	kernel_matrix = hidden_layers->kernel_matrix;
	user_nn_matrix_transpose(kernel_matrix);//����ת��
	input_deltas_matrix_temp = user_nn_matrix_mult_matrix(kernel_matrix, hidden_deltas_matrix);//����˷�
	user_nn_matrix_transpose(kernel_matrix);//����ת��
	user_nn_matrix_cpy_matrix(input_deltas_matrix, input_deltas_matrix_temp);//����в�

	user_nn_matrix_delete(deltas_kernel_matrix_temp);//ɾ������
	user_nn_matrix_delete(input_deltas_matrix_temp);//ɾ������
	
}

