#include "user_cnn_bp.h"

//�������д��ۺ������㡢����ֵ���㡢����в����
//1.����output_feture_maps-1�ķ�ʽ�õ�����ֵoutput_error_maps
//2.ͨ����ʽ 1/2*(output_error_maps.*output_error_maps)/output_error_maps������ �õ��������
//3.���� output_error_maps.*(output_feture_maps.*(1-output_feture_maps))�õ������� output_deltas_maps
//4.����output_deltas_maps.*output_kernel_maps �õ��в�
//����㷴����������һ��
//������
//prior_layer  ��һ��
//output_layer ����
//index ����ĳ��λ������ֵΪ1
//���� ��
void user_cnn_bp_output_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *output_layer){
	user_cnn_output_layers  *output_layers = (user_cnn_output_layers  *)output_layer->content;//��ȡ����ػ�������
	user_nn_matrix			*output_feature_matrix = ((user_cnn_output_layers *)output_layer->content)->feature_matrix;//�����������  ����
	user_nn_matrix          *output_deltas_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_matrix;//�����������Ȼ��߲в�
	user_nn_matrix			*output_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->kernel_matrix;//�������������
	user_nn_matrix          *output_error_matrix = ((user_cnn_output_layers *)output_layer->content)->error_matrix;//����ֵ����
	user_nn_matrix			*output_target_matrix = ((user_cnn_output_layers *)output_layer->content)->target_matrix;//Ŀ�����
	user_nn_matrix          *_feture_vector_deltas = NULL;//�в�򴫲���ǰһ��,net.fvd������ǲв�
	//��������� ����ֵ E = ʵ��ֵ - ����ֵ
	user_nn_matrix_cum_matrix_mult_alpha(output_error_matrix, output_feature_matrix, output_target_matrix, -1.0f);//�������ֵ
	user_nn_matrix_printf(NULL, output_target_matrix);
	user_nn_matrix_printf(NULL, output_feature_matrix);
	//�����������ۺ��� Y = (1/2)*E^2
	
	output_layers->loss_function = user_nn_matrix_get_rmse(output_error_matrix);//���ۺ��������þ���������Ϊ���ۺ���  
	//�����������в� matlab��ʽ��output_deltas_maps = output_error_maps.*output_feture_maps.*(1-output_feture_maps)  ����ֵ*���������sigmoid�ĵ���
	user_nn_activate_matrix_d_mult_matrix(output_deltas_matrix, output_error_matrix, output_feature_matrix, user_nn_cnn_softmax);//�Ա����󵼵õ������Ĳв� 
	//�����ǰѵõ����������з������������Ĳв��Ȩ��ֵ
	user_nn_matrix_transpose(output_kernel_matrix);//����output_kernel_maps �� width��height 
	_feture_vector_deltas = user_nn_matrix_mult_matrix(output_kernel_matrix, output_deltas_matrix);//����feature vector delta  ######������������Ҫ�仯���ݴ�С######
	user_nn_matrix_transpose(output_kernel_matrix);//����output_kernel_maps �� width��height ��������
	//������аѵõ��������в����ǰһ��
	if (prior_layer->type == u_cnn_layer_type_input){
		//return;//������޲в�
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		user_nn_list_matrix		*before_deltas_list = ((user_cnn_pool_layers *)prior_layer->content)->deltas_matrices;//ת����������
		user_nn_matrix_to_matrices(before_deltas_list, _feture_vector_deltas);//����вǰһ��
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		//feture_vector = feture_vector.*(output_feture_maps.*(1 - output_feture_maps))
		user_nn_list_matrix		*before_deltas_matrices = ((user_cnn_conv_layers *)prior_layer->content)->deltas_matrices;//ת����������
		user_nn_list_matrix		*before_feature_matrices = ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices;//ת����������	
		user_nn_matrix_to_matrices(before_deltas_matrices, _feture_vector_deltas);//����вǰһ��
		user_nn_activate_matrices_d_mult_matrices(before_deltas_matrices, before_deltas_matrices, before_feature_matrices, user_nn_cnn_softmax);//��
	} if (prior_layer->type == u_cnn_layer_type_full){
		user_nn_matrix		*before_deltas_matrix = ((user_cnn_full_layers *)prior_layer->content)->deltas_matrix;//ת����������
		user_nn_matrix		*before_feature_matrix = ((user_cnn_full_layers *)prior_layer->content)->feature_matrix;//ת����������	
		user_nn_matrix_cpy_matrix(before_deltas_matrix, _feture_vector_deltas);//ֱ�Ӹ��� ȫ���Ӳ�������ǣ�1,N���ľ���
		user_nn_activate_matrix_d_mult_matrix(before_deltas_matrix, before_deltas_matrix, before_feature_matrix, user_nn_cnn_softmax);//�󵼾���
	}
	else{
		//return;
	}
	user_nn_matrix_delete(_feture_vector_deltas);//ɾ������
}
//ȫ���Ӳ�в����
//prior_layer ȫ���Ӳ��ǰһ��
//full_layer ȫ���Ӳ�
//���� ��
void user_cnn_bp_fullconnect_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *full_layer){
	user_cnn_full_layers   *full_layers = (user_cnn_full_layers  *)full_layer->content;//��ȡ����ػ�������
	user_nn_matrix			*full_feature_matrix = ((user_cnn_full_layers *)full_layer->content)->feature_matrix;//�����������  ����
	user_nn_matrix          *full_deltas_matrix = ((user_cnn_full_layers *)full_layer->content)->deltas_matrix;//�����������Ȼ��߲в�
	user_nn_matrix			*full_kernel_matrix = ((user_cnn_full_layers *)full_layer->content)->kernel_matrix;//�������������
	user_nn_matrix          *_feture_vector_deltas = NULL;//�в�򴫲���ǰһ��,net.fvd������ǲв�
	//��ȡ����Ĳв�Ȩ��
	user_nn_matrix_transpose(full_kernel_matrix);//����output_kernel_maps �� width��height 
	_feture_vector_deltas = user_nn_matrix_mult_matrix(full_kernel_matrix, full_deltas_matrix);//����feature vector delta  ######������������Ҫ�仯���ݴ�С######
	user_nn_matrix_transpose(full_kernel_matrix);//����output_kernel_maps �� width��height ��������
	//������аѵõ��������в����ǰһ��
	if (prior_layer->type == u_cnn_layer_type_input){
		//return;//������޲в�
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		user_nn_list_matrix		*before_deltas_list = ((user_cnn_pool_layers *)prior_layer->content)->deltas_matrices;//ת����������
		user_nn_matrix_to_matrices(before_deltas_list, _feture_vector_deltas);//����вǰһ��
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		//feture_vector = feture_vector.*(output_feture_maps.*(1 - output_feture_maps))
		user_nn_list_matrix		*before_deltas_matrices = ((user_cnn_conv_layers *)prior_layer->content)->deltas_matrices;//ת����������
		user_nn_list_matrix		*before_feature_matrices = ((user_cnn_conv_layers *)prior_layer->content)->feature_matrices;//ת����������	
		user_nn_matrix_to_matrices(before_deltas_matrices, _feture_vector_deltas);//����вǰһ��
		user_nn_activate_matrices_d_mult_matrices(before_deltas_matrices, before_deltas_matrices, before_feature_matrices, user_nn_cnn_softmax);//��
	} if (prior_layer->type == u_cnn_layer_type_full){
		user_nn_matrix		*before_deltas_matrix = ((user_cnn_full_layers *)prior_layer->content)->deltas_matrix;//ת����������
		user_nn_matrix		*before_feature_matrix = ((user_cnn_full_layers *)prior_layer->content)->feature_matrix;//ת����������	
		user_nn_matrix_cpy_matrix(before_deltas_matrix, _feture_vector_deltas);//ֱ�Ӹ��� ȫ���Ӳ�������ǣ�1,N���ľ���
		user_nn_activate_matrix_d_mult_matrix(before_deltas_matrix, before_deltas_matrix, before_feature_matrix, user_nn_cnn_softmax);//�󵼾��� �õ�ǰһ��Ĳв�
	}
	else{
		//return;
	}
	user_nn_matrix_delete(_feture_vector_deltas);//ɾ������

}

//�ػ��㵽�����
//�в���ھ����
//1.��pool��Ĳв�������conv���С�����䷽ʽ���þ�ֵ���䡣
//2.ʹ�ù�ʽ��conv_feture_maps * (1 - conv_feture_maps) * pool_deltas_maps ������pool_deltas_maps�ǵ�һ��������ֵ pool_deltas_maps*��ǰһ����������
//����Ӧ���� pool�� ǰһ��Ӧ���Ǿ����
void user_cnn_bp_pooling_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *pool_layer){
	user_cnn_pool_layers	*pool_layers	  = (user_cnn_pool_layers  *)pool_layer->content;//����������������
	user_nn_list_matrix		*pooling_deltas_matrices = ((user_cnn_pool_layers  *)pool_layer->content)->deltas_matrices;//����������������
	user_nn_matrix          *pooling_deltas_matrix = NULL;//
	
	user_nn_list_matrix     *before_feature_matrices = NULL;
	user_nn_matrix          *before_feature_matrix = NULL;
	user_nn_list_matrix     *before_deltas_matrices = NULL;
	user_nn_matrix          *before_deltas_matrix = NULL;
	user_nn_matrix          *_deltas_matrix = NULL;
	int output_feature_index;

	if (prior_layer->type == u_cnn_layer_type_input){
		return;
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		before_feature_matrices = ((user_cnn_pool_layers  *)prior_layer->content)->feature_matrices;//
		before_deltas_matrices = ((user_cnn_pool_layers  *)prior_layer->content)->deltas_matrices;//
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		before_feature_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->feature_matrices;//
		before_deltas_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->deltas_matrices;//
	}
	else{
		return;
	}

	for (output_feature_index = 0; output_feature_index < pool_layers->feature_number; output_feature_index++){
		pooling_deltas_matrix = user_nn_matrices_ext_matrix_index(pooling_deltas_matrices, output_feature_index);//
		before_feature_matrix = user_nn_matrices_ext_matrix_index(before_feature_matrices, output_feature_index);//
		before_deltas_matrix = user_nn_matrices_ext_matrix_index(before_deltas_matrices, output_feature_index);//

		//����ÿ���в����Ȼ���ٽ�����
		_deltas_matrix = user_nn_matrix_expand_mult_constant(pooling_deltas_matrix, pool_layers->pool_width, pool_layers->pool_height, (float)1 / (pool_layers->pool_width * pool_layers->pool_height));//����ָ�������������
		user_nn_activate_matrix_d_mult_matrix(before_deltas_matrix, _deltas_matrix, before_feature_matrix, user_nn_cnn_softmax);//�Ա�����
		user_nn_matrix_delete(_deltas_matrix);//ɾ������
	}
}
//����㵽�ػ���
//�Ѿ����Ĳв� ͨ�������������㵽pool��
//
void user_cnn_bp_convolution_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer){
	user_cnn_conv_layers	*conv_layers		= (user_cnn_conv_layers  *)conv_layer->content;//����������������
	user_nn_list_matrix     *conv_deltas_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_matrices;//
	user_nn_matrix          *conv_deltas_matrix = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_matrices->matrix;//
	user_nn_list_matrix     *conv_kernel_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->kernel_matrices;//
	user_nn_matrix			*conv_kernel_matrix = NULL;//

	user_nn_matrix          *before_feature_matrix = NULL;//
	user_nn_list_matrix     *before_deltas_matrices = NULL;
	user_nn_matrix          *before_deltas_matrix = NULL;//

	user_nn_matrix          *_total_matrix = NULL;
	user_nn_matrix          *_conv_matrix = NULL;
	user_nn_matrix          *_kernel_matrix = NULL;

	int input_feature_index, output_feature_index;

	if (prior_layer->type == u_cnn_layer_type_input){
		return;
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		before_feature_matrix = ((user_cnn_pool_layers  *)prior_layer->content)->feature_matrices->matrix;//
		before_deltas_matrices = ((user_cnn_pool_layers  *)prior_layer->content)->deltas_matrices;
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		before_feature_matrix = ((user_cnn_conv_layers  *)prior_layer->content)->feature_matrices->matrix;//
		before_deltas_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->deltas_matrices;
	}
	else{
		return;
	}

	_total_matrix = user_nn_matrix_create(before_feature_matrix->width, before_feature_matrix->height);//����������ǰ��һ����ͬ��С����

	for (input_feature_index = 0; input_feature_index < conv_layers->input_feature_number; input_feature_index++){
		user_nn_matrix_memset(_total_matrix, 0);//��վ���
		for (output_feature_index = 0; output_feature_index < conv_layers->feature_number; output_feature_index++){
			conv_deltas_matrix = user_nn_matrices_ext_matrix_index(conv_deltas_matrices, output_feature_index);//��ȡ����в�
			conv_kernel_matrix = user_nn_matrices_ext_matrix(conv_kernel_matrices, input_feature_index, output_feature_index);//��ȡָ��λ�õľ����
			_kernel_matrix = user_nn_matrix_rotate180(conv_kernel_matrix);//�������ת180��
			_conv_matrix = user_nn_matrix_conv2(conv_deltas_matrix, _kernel_matrix, u_nn_conv2_type_full);//���о������
			user_nn_matrix_cum_matrix(_total_matrix, _total_matrix, _conv_matrix);//�����ۼ�
			user_nn_matrix_delete(_conv_matrix);//ɾ������
			user_nn_matrix_delete(_kernel_matrix);//ɾ������
		}
		before_deltas_matrix = user_nn_matrices_ext_matrix_index(before_deltas_matrices, input_feature_index);//��ȡ�в�ָ��
		user_nn_matrix_cpy_matrix(before_deltas_matrix, _total_matrix);//���²в�ֵ
	}
	user_nn_matrix_delete(_total_matrix);//ɾ������
}

//�����Ҫ���µ�Ȩ��ֵ
void user_cnn_bp_convolution_deltas_kernel(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer){
	user_cnn_conv_layers  *conv_layers				= (user_cnn_conv_layers  *)conv_layer->content;//��ȡ����������
	user_nn_list_matrix   *conv_deltas_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_matrices;
	user_nn_matrix        *conv_deltas_matrix = NULL;
	user_nn_list_matrix	  *conv_deltas_kernel_matrices = ((user_cnn_conv_layers  *)conv_layer->content)->deltas_kernel_matrices;//
	user_nn_matrix		  *conv_deltas_kernel_matrix = NULL;//
	float				  *conv_deltas_bias			= ((user_cnn_conv_layers  *)conv_layer->content)->deltas_biases_matrix->data;//ָ��в��ƫ�ò���
	user_nn_list_matrix   *before_feature_matrices = NULL;//������������  
	user_nn_matrix        *before_feature_matrix = NULL;//������������ 
	user_nn_matrix        *_result_matrix = NULL;//�������
	user_nn_matrix        *_deltas_matrix = NULL;//�������
	
	int count_conv_maps, count_input_maps;//��������

	//��ȡǰһ�����������
	if (prior_layer->type == u_cnn_layer_type_input){
		before_feature_matrices = ((user_cnn_input_layers *)prior_layer->content)->feature_matrices;
	}
	else if (prior_layer->type == u_cnn_layer_type_pool){
		before_feature_matrices = ((user_cnn_pool_layers *)prior_layer->content)->feature_matrices;
	}
	else if (prior_layer->type == u_cnn_layer_type_conv){
		before_feature_matrices = ((user_cnn_conv_layers  *)prior_layer->content)->feature_matrices;
	}
	else{

	}
	for (count_conv_maps = 0; count_conv_maps < conv_layers->feature_number; count_conv_maps++){
		conv_deltas_matrix = user_nn_matrices_ext_matrix_index(conv_deltas_matrices, count_conv_maps);//��ȡһ���в�
		for (count_input_maps = 0; count_input_maps < conv_layers->input_feature_number; count_input_maps++){
			before_feature_matrix = user_nn_matrices_ext_matrix_index(before_feature_matrices, count_input_maps);//��ȡǰһ�����������
			conv_deltas_kernel_matrix = user_nn_matrices_ext_matrix(conv_deltas_kernel_matrices, count_input_maps, count_conv_maps);//��ȡ�в�Ȩ�ض�Ӧ��ָ��
			_deltas_matrix = user_nn_matrix_rotate180(before_feature_matrix);//��ǰһ�������������ת180��
			_result_matrix = user_nn_matrix_conv2(_deltas_matrix, conv_deltas_matrix, u_nn_conv2_type_valid);//���о������
			user_nn_matrix_cpy_matrix(conv_deltas_kernel_matrix, _result_matrix);//���ﱣ����ǲв���Ȩ�س˻�
			user_nn_matrix_delete(_deltas_matrix);//ɾ������
			user_nn_matrix_delete(_result_matrix);//ɾ������
		}
		*conv_deltas_bias++ = user_nn_matrix_cum_element(conv_deltas_matrix);//��Ͳв���Ϊ�в�ƫ�ò���
	}
}
//���ȫ���Ӳ�ĸ���Ȩ��ֵ
void user_cnn_bp_full_deltas_kernel(user_cnn_layers *full_layer){
	user_nn_matrix			*full_input_feture_matrix	= ((user_cnn_full_layers *)full_layer->content)->input_feature_matrix;
	user_nn_matrix          *full_deltas_matrix			= ((user_cnn_full_layers *)full_layer->content)->deltas_matrix;//�����������Ȼ��߲в�
	user_nn_matrix          *full_deltas_kernel_matrix	= ((user_cnn_full_layers *)full_layer->content)->deltas_kernel_matrix;//�����������Ȼ��߲в�
	user_nn_matrix          *_grads_matrix = NULL;

	// output_deltas_maps*input_feture_maps
	user_nn_matrix_transpose(full_input_feture_matrix);//����output_kernel_maps �� width��height ��������
	_grads_matrix = user_nn_matrix_mult_matrix(full_deltas_matrix, full_input_feture_matrix);//��ȡ���������
	user_nn_matrix_transpose(full_input_feture_matrix);//����output_kernel_maps �� width��height
	user_nn_matrix_cpy_matrix(full_deltas_kernel_matrix, _grads_matrix);//
	user_nn_matrix_delete(_grads_matrix);//ɾ������
}
//��������ĸ���Ȩ��ֵ
void user_cnn_bp_output_deltas_kernel(user_cnn_layers *output_layer){
	user_nn_matrix			*output_input_feture_matrix = ((user_cnn_output_layers *)output_layer->content)->input_feature_matrix;
	user_nn_matrix          *output_deltas_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_matrix;//�����������Ȼ��߲в�
	user_nn_matrix          *output_deltas_kernel_matrix = ((user_cnn_output_layers *)output_layer->content)->deltas_kernel_matrix;//�����������Ȼ��߲в�
	user_nn_matrix          *_grads_matrix = NULL;

	// output_deltas_maps*input_feture_maps
	user_nn_matrix_transpose(output_input_feture_matrix);//����output_kernel_maps �� width��height ��������
	_grads_matrix = user_nn_matrix_mult_matrix(output_deltas_matrix, output_input_feture_matrix);//��ȡ���������
	user_nn_matrix_transpose(output_input_feture_matrix);//����output_kernel_maps �� width��height
	user_nn_matrix_cpy_matrix(output_deltas_kernel_matrix, _grads_matrix);//
	user_nn_matrix_delete(_grads_matrix);//ɾ������
}