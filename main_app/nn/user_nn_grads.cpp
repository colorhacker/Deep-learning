
#include "user_nn_grads.h"


//�����������Ȩֵ��ƫ�ò���
//hidden_layer ����ȫ���Ӳ�Ķ���
//alpha ����ϵ��
//���� ��
void user_nn_grads_hidden(user_nn_layers *hidden_layer, float alpha) {
	user_nn_hidden_layers *hidden_layers = (user_nn_hidden_layers  *)hidden_layer->content;//��ȡ����Ĳ���

	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->kernel_matrix, hidden_layers->deltas_kernel_matrix, -1.0f*alpha);
#if user_nn_use_bias
	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->biases_matrix, hidden_layers->deltas_biases_matrix, -1.0f*alpha);
#endif
}
//����������Ȩֵ��ƫ�ò���
//output_layer ����ȫ�����Ķ���
//alpha ����ϵ��
//���� ��
void user_nn_grads_output(user_nn_layers *output_layer, float alpha) {
	user_nn_output_layers *output_layers = (user_nn_output_layers  *)output_layer->content;//��ȡ����Ĳ���
	user_nn_matrix_sum_matrix_mult_alpha(output_layers->kernel_matrix, output_layers->deltas_kernel_matrix, -1.0f*alpha);
#if user_nn_use_bias
	user_nn_matrix_sum_matrix_mult_alpha(output_layers->biases_matrix, output_layers->deltas_biases_matrix, -1.0f*alpha);
#endif
}