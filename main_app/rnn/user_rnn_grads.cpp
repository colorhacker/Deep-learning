
#include "user_rnn_grads.h"


//�����������Ȩֵ��ƫ�ò���
//hidden_layer ����ȫ���Ӳ�Ķ���
//alpha ����ϵ��
//���� ��
void user_rnn_grads_hidden(user_rnn_layers *hidden_layer, float alpha) {
	user_rnn_hidden_layers *hidden_layers = (user_rnn_hidden_layers  *)hidden_layer->content;//��ȡ����Ĳ���

	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->kernel_matrix, hidden_layers->deltas_kernel_matrix, -1.0f*alpha);
	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->kernel_matrix_t, hidden_layers->deltas_kernel_matrix_t, -1.0f*alpha);
	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->biases_matrix, hidden_layers->deltas_biases_matrix, -1.0f*alpha);

	user_nn_matrix_memset(hidden_layers->deltas_kernel_matrix, 0);
	user_nn_matrix_memset(hidden_layers->deltas_kernel_matrix_t, 0);
	user_nn_matrix_memset(hidden_layers->deltas_biases_matrix, 0);
	user_nn_matrix_memset(hidden_layers->deltas_matrix_t, 0);
}
//����������Ȩֵ��ƫ�ò���
//output_layer ����ȫ�����Ķ���
//alpha ����ϵ��
//���� ��
void user_rnn_grads_output(user_rnn_layers *output_layer, float alpha) {
	user_rnn_output_layers *output_layers = (user_rnn_output_layers  *)output_layer->content;//��ȡ����Ĳ���
	user_nn_matrix_sum_matrix_mult_alpha(output_layers->kernel_matrix, output_layers->deltas_kernel_matrix, -1.0f*alpha);
	user_nn_matrix_sum_matrix_mult_alpha(output_layers->biases_matrix, output_layers->deltas_biases_matrix, -1.0f*alpha);

	user_nn_matrix_memset(output_layers->deltas_kernel_matrix, 0);
	user_nn_matrix_memset(output_layers->deltas_biases_matrix, 0);
	//user_nn_matrix_printf(NULL, output_layers->deltas_biases_matrix);
}