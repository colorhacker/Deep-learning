
#include "user_nn_grads.h"


//更新隐含层的权值与偏置参数
//hidden_layer 更新全连接层的对象
//alpha 更新系数
//返回 无
void user_nn_grads_hidden(user_nn_layers *hidden_layer, float alpha) {
	user_nn_hidden_layers *hidden_layers = (user_nn_hidden_layers  *)hidden_layer->content;//获取本层的参数

	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->kernel_matrix, hidden_layers->deltas_kernel_matrix, -1.0f*alpha);
#if user_nn_use_bias
	user_nn_matrix_sum_matrix_mult_alpha(hidden_layers->biases_matrix, hidden_layers->deltas_biases_matrix, -1.0f*alpha);
#endif
}
//更新输出层的权值与偏置参数
//output_layer 更新全输出层的对象
//alpha 更新系数
//返回 无
void user_nn_grads_output(user_nn_layers *output_layer, float alpha) {
	user_nn_output_layers *output_layers = (user_nn_output_layers  *)output_layer->content;//获取本层的参数
	user_nn_matrix_sum_matrix_mult_alpha(output_layers->kernel_matrix, output_layers->deltas_kernel_matrix, -1.0f*alpha);
#if user_nn_use_bias
	user_nn_matrix_sum_matrix_mult_alpha(output_layers->biases_matrix, output_layers->deltas_biases_matrix, -1.0f*alpha);
#endif
}