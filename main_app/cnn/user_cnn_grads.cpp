
#include "user_cnn_grads.h"

//����Ȩ����ƫ�ò���
//conv_layer ���¾����Ķ���
//alpha ����ϵ��
//���� ��
void user_cnn_grads_convolution(user_cnn_layers *conv_layer, float alpha){
	user_cnn_conv_layers  *conv_layers					= (user_cnn_conv_layers  *)conv_layer->content;//��ȡ����������
	user_nn_list_matrix	  *conv_kernel_matrices			= ((user_cnn_conv_layers  *)conv_layer->content)->kernel_matrices;
	user_nn_matrix		  *conv_kernel_matrix			= NULL;//�����ģ��
	float				  *conv_biases					= ((user_cnn_conv_layers  *)conv_layer->content)->biases_matrix->data;//����ƫ�ò���
	user_nn_list_matrix	  *conv_deltas_kernel_matrices	= ((user_cnn_conv_layers  *)conv_layer->content)->deltas_kernel_matrices;
	user_nn_matrix		  *conv_deltas_kernel_matrix	= NULL;//
	float				  *conv_deltas_biases			= ((user_cnn_conv_layers  *)conv_layer->content)->deltas_biases_matrix->data;//ָ��в�ĸ���ֵ

	int output_feature_index, input_feature_index;//������������

	for (output_feature_index = 0; output_feature_index < conv_layers->feature_number; output_feature_index++){
		for (input_feature_index = 0; input_feature_index < conv_layers->input_feature_number; input_feature_index++){
			conv_kernel_matrix		  = user_nn_matrices_ext_matrix(conv_kernel_matrices, input_feature_index, output_feature_index);//ȡ��ָ��λ����������
			conv_deltas_kernel_matrix = user_nn_matrices_ext_matrix(conv_deltas_kernel_matrices, input_feature_index, output_feature_index);//ȡ��ָ��λ����������
			//conv_kernel_maps = conv_kernel_maps - alpha * conv_deltas_kernel_maps
			user_nn_matrix_sum_matrix_mult_alpha(conv_kernel_matrix, conv_deltas_kernel_matrix, -1.0f * alpha);
		}
		//conv_bias = conv_bias - alpha * conv_deltas_bias
		*conv_biases++ = *conv_biases - *conv_deltas_biases++ *alpha;
	}
}
//����������Ȩֵ��ƫ�ò���
//full_layer ����ȫ���Ӳ�Ķ���
//alpha ����ϵ��
//���� ��
void user_cnn_grads_full(user_cnn_layers *full_layer, float alpha){
	user_nn_matrix			*full_bias_matrix			= ((user_cnn_full_layers *)full_layer->content)->biases_matrix;//�����ľ��ģ��
	user_nn_matrix          *full_kernel_matrix			= ((user_cnn_full_layers *)full_layer->content)->kernel_matrix;//�����ľ��ģ��
	user_nn_matrix          *full_deltas_matrix			= ((user_cnn_full_layers *)full_layer->content)->deltas_matrix;//�����������Ȼ��߲в�
	user_nn_matrix          *full_deltas_kernel_matrix	= ((user_cnn_full_layers *)full_layer->content)->deltas_kernel_matrix;//�����������Ȼ��߲в�
	//output_kernel_maps = output_kernel_maps - alpha * output_grads_maps
	user_nn_matrix_sum_matrix_mult_alpha(full_kernel_matrix, full_deltas_kernel_matrix, -1.0f * alpha);
	//output_bias_maps = output_bias_maps - alpha * output_deltas_maps;
	user_nn_matrix_sum_matrix_mult_alpha(full_bias_matrix, full_deltas_matrix, -1.0f * alpha);
}
//����������Ȩֵ��ƫ�ò���
//output_layer ����ȫ�����Ķ���
//alpha ����ϵ��
//���� ��
void user_cnn_grads_output(user_cnn_layers *output_layer, float alpha){
	user_nn_matrix			*output_bias_matrix = ((user_cnn_output_layers *)output_layer->content)->biases_matrix;//�����ľ��ģ��
	user_nn_matrix          *output_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->kernel_matrix;//�����ľ��ģ��
	user_nn_matrix          *output_deltas_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_matrix;//�����������Ȼ��߲в�
	user_nn_matrix          *output_deltas_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_kernel_matrix;//�����������Ȼ��߲в�
	//output_kernel_maps = output_kernel_maps - alpha * output_grads_maps
	user_nn_matrix_sum_matrix_mult_alpha(output_kernel_matrix, output_deltas_kernel_matrix, -1.0f * alpha);
	//output_bias_maps = output_bias_maps - alpha * output_deltas_maps;
	user_nn_matrix_sum_matrix_mult_alpha(output_bias_matrix, output_deltas_matrix, -1.0f * alpha);

}