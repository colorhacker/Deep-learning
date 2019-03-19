
#include "user_nn_app.h"

void user_nn_app_test(int argc, const char** argv) {

	//srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 2, //����� ��������ȡ��߶ȡ�ʱ�䳤�ȣ�
		'h', 2, //������ ���� ���߶ȡ�ʱ�䳤�ȣ�
		'o', 2 //����� ���� ���߶ȡ�ʱ�䳤�ȣ�
	};
	user_nn_input_layers	*nn_input_layers = NULL;
	user_nn_hidden_layers	*nn_hidden_layers = NULL;
	user_nn_output_layers	*nn_output_layers = NULL;

	float loss_function = 0.0f;
	bool model_is_exist = false;
	user_nn_matrix *input_data = user_nn_matrix_create(1, 2);//������������
	user_nn_matrix *target_data = user_nn_matrix_create(1, 2);//������������

	user_nn_layers *rnn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//����ģ��
	if (rnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		rnn_layers = user_nn_model_create(user_layers);//����ģ��
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	input_data->data[0] = 1;
	input_data->data[1] = 1;
	target_data->data[0] = 0.01f;
	target_data->data[1] = 0.01f;
	while (!model_is_exist) {
		user_nn_model_load_input_feature(rnn_layers, input_data);//������������
		user_nn_model_load_target_feature(rnn_layers, target_data);//����Ŀ������									   
		user_nn_model_ffp(rnn_layers);//�������һ��
		user_nn_model_bp(rnn_layers, 0.01f);//�������һ��
		loss_function = user_nn_model_return_loss(rnn_layers);
		if (loss_function <= 0.001f) {
			user_nn_model_save_model(user_nn_model_nn_file_name, rnn_layers);//����ģ��
			break;
		}
		printf("\nloss:%f", loss_function);
	}
	user_nn_model_load_input_feature(rnn_layers, input_data);//������������
	user_nn_model_ffp(rnn_layers);//���м���
	user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));

	getchar();
}
//���һ���� ��� start <=> end-1
float _nn_topic_random(int max)
{
	//float r = (float)(rand() / ((RAND_MAX / max) + 1));
	return (float)(rand() / ((RAND_MAX / max) + 1));
}

//������ ���ն����Ʒ�ʽ�������ظ�
int _nn_topic_increase_answer(char *topic_score) {
	int score = 0, offset = 0,select_count=0;//����ѡ�񡢵÷֡�����ƫ��
	if (topic_score[offset] == 't') {
		topic_score[offset + 2]++;//���м�������
	}
	while (topic_score[offset] == 't') {
		select_count = topic_score[offset + 3];//��¼�ж��ٸ�ѡ��
		switch (topic_score[offset + 1]) {//�ж�����
		case 'x'://x���ʹ�
			if (topic_score[offset + 2] >= select_count) {
				topic_score[offset + 2] = 0;
				if (topic_score[offset + select_count + 4] == 't') {
					topic_score[offset + select_count + 4 + 2]++;
				}else {
					score += topic_score[offset + topic_score[offset + 2] + 4];
					return score;
				}
			}
			//printf(" %d",select);
		default:break;
		}
		score += topic_score[offset + topic_score[offset + 2] + 4];//�ۼƻ���
		offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		//printf("offset:%d,select:%d,score:%d\n", offset,select, score);
	}
	if (topic_score[offset - select_count - 2] >= select_count) {
		offset = 0;
		while (topic_score[offset] == 't') {
			topic_score[offset + 2] = 0;//�������
			offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		}
	}
	return score;
}
//����� ����Ŀ���ֵ�������ô�
int _nn_topic_random_answer(char *topic_score) {
	int score = 0, offset = 0;//����ѡ�񡢵÷֡�����ƫ��
	while (topic_score[offset] == 't') {
		switch (topic_score[offset + 1]) {//�ж�����
			case 'x'://x���ʹ�
				topic_score[offset + 2] = (int)_nn_topic_random(topic_score[offset + 3]);//��ȡ�����ô�
				//printf(" %d",select);
			default:break;
		}
		score += topic_score[offset + topic_score[offset + 2] + 4];//�ۼƻ���
		offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		//printf("offset:%d,select:%d,score:%d\n", offset,select, score);
	}
	return score;
}
//�Ѵ����ɾ��� ���ն����Ʒ�ʽ������Ŀ��ʽ
user_nn_matrix *_nn_topic_abswer_to_matrix(char *topic_score, abswer_type type) {
	int offset = 0,offset_score = 0,total_size = 0;//����ѡ�񡢵÷֡�����ƫ��
	char *topic_ram = NULL;
	user_nn_matrix *result = NULL;
	float *result_data = NULL;
	if (type == abswer_hex) {
		offset = 0;
		while (topic_score[offset] == 't') {
			total_size ++;//��¼����
			offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		}
		offset = 0;
		result = user_nn_matrix_create(1, total_size);//���ɾ���
		result_data = result->data;
		while (topic_score[offset] == 't') {
			*result_data++ = topic_score[offset + 2] * 0.1f;
			offset_score += topic_score[offset + 3];//���
			offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		}
	}
	else if (type == abswer_bin) {
		offset = 0;
		while (topic_score[offset] == 't') {
			total_size += topic_score[offset + 3];//��¼����
			offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		}
		offset = 0;
		result = user_nn_matrix_create(1, total_size);//���ɾ���
		result_data = result->data;
		while (topic_score[offset] == 't') {
			result_data[offset_score + topic_score[offset + 2]] = 1.0f;//���ɴ�
			offset_score += topic_score[offset + 3];//���
			offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
		}
	}

	return result;
}
//�ѵ÷����ɾ���
user_nn_matrix *_nn_topic_score_to_matrix(char *topic_score) {
	int score = 0, offset = 0,target = 0;//����ѡ�񡢵÷֡�����ƫ��
	user_nn_matrix *result = NULL;
	float *result_data=NULL;
	while (topic_score[offset] == 't') {
		score += topic_score[offset + topic_score[offset + 2] + 4];//�ۼƻ���
		offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
	}
	if (topic_score[offset] == 's') {
		target = topic_score[offset + 3];
		result = user_nn_matrix_create(1, target);//��������
		result_data = result->data;
		offset += 4;
		while (target--) {
			if (score <= topic_score[offset++]) {
				*result_data = 1.0f;
				break;
			}
			result_data++;
		}
	}
	return result;
}
//����
/*int score = 0,count = 0;
user_nn_list_matrix *topic_matrices = user_nn_matrices_create_head(1, 1);//������������
user_nn_list_matrix *score_matrices = user_nn_matrices_create_head(1, 1);//������������
while (count++ < 100) {
score = _nn_topic_random_answer(topic_score);//�����
user_nn_matrix *topic_matrix = _nn_topic_abswer_to_matrix(topic_score);//�Ѵ���ת������
user_nn_matrix *score_matrix = _nn_topic_score_to_matrix(topic_score);//���ɽ�������ֶξ���
//user_nn_matrices_add_matrix(topic_matrices, topic_matrix);
//user_nn_matrices_add_matrix(score_matrices, score_matrix);
}*/
//��ȡ�ַ��������
int *user_nn_get_layer_array(int layer,int *buffer,int size) {
	while (size--) {
		if (*buffer == layer) {
			return buffer;
		}
		buffer++;
	}
	return NULL;
}
//����ѵ���Ʒ�
void user_nn_app_topic(int argc, const char** argv) {
	char topic_score[] = {
		//'t'��ʼ,'x'������,'n'��ѡ��,5�����𰸵ĸ���,...�𰸵÷�
		't','x',NULL,5,4,3,2,1,0,//��һ��������ֵ �ڶ��б���ѡ��Ĵ� ��������Ŀ���ѡ��,���������ٸ���,���汣��÷�ֵ
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		's','x',NULL,4,10,20,28,40
	};
	//srand((unsigned)time(NULL));//������� ----- ����������ôÿ��ѵ�����һ��
	int user_layers[] = {
		'i', 1, 0, //����� ��������ȡ��߶ȣ�
		'h', 40, //������ ���� �����ȣ�
		'h', 80, //������ ���� �����ȣ�
		'h', 40, //������ ���� �����ȣ�
		'o', 0 //����� ���� �����ȣ�
	};
	user_nn_input_layers	*nn_input_layers = NULL;
	user_nn_hidden_layers	*nn_hidden_layers = NULL;
	user_nn_output_layers	*nn_output_layers = NULL;
	user_nn_matrix *topic_matrix = NULL;
	user_nn_matrix *score_matrix = NULL;
	float loss_function = 0.0f;
	bool model_is_exist = false;
	abswer_type input_data_type = abswer_hex;//������������

	user_nn_layers *rnn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//����ģ��
	if (rnn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		topic_matrix = _nn_topic_abswer_to_matrix(topic_score, input_data_type);//�Ѵ���ת������
		*(user_nn_get_layer_array('i', user_layers, sizeof(user_layers)) + 1) = topic_matrix->width;//���ص�ǰ�߶�
		*(user_nn_get_layer_array('i', user_layers, sizeof(user_layers)) + 2) = topic_matrix->height;//���ص�ǰ�߶�
		score_matrix = _nn_topic_score_to_matrix(topic_score);//���ɽ�������ֶξ���
		*(user_nn_get_layer_array('o', user_layers, sizeof(user_layers)) + 1) = score_matrix->height;//���ص�ǰ�߶�
		rnn_layers = user_nn_model_create(user_layers);//����ģ��
		user_nn_matrix_delete(topic_matrix);//ɾ������
		user_nn_matrix_delete(score_matrix);//ɾ������
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	int train_max_count = 0;
	while (!model_is_exist) {
		_nn_topic_increase_answer(topic_score);//˳���
		//_nn_topic_random_answer(topic_score);
		topic_matrix = _nn_topic_abswer_to_matrix(topic_score, input_data_type);//�Ѵ���ת������
		score_matrix = _nn_topic_score_to_matrix(topic_score);//���ɽ�������ֶξ���
		user_nn_model_load_input_feature(rnn_layers, topic_matrix);//������������
		user_nn_model_load_target_feature(rnn_layers, score_matrix);//����Ŀ������
		//�������һ�� ��ʱ��Ƭ����N��
		user_nn_model_ffp(rnn_layers);
		//�������һ�� ��ʱ��Ƭ����N��
		user_nn_model_bp(rnn_layers, 0.001f);
		loss_function = user_nn_model_return_loss(rnn_layers);
		//user_nn_matrix_printf(NULL, score_matrix);//��ӡ
		user_nn_matrix_delete(topic_matrix);//ɾ������
		user_nn_matrix_delete(score_matrix);//ɾ������
		printf("train_max_count:%d loss:%f\n", train_max_count, loss_function);
		if ((++train_max_count >= 100000) && (loss_function <= 0.1f)) {
			user_nn_model_save_model(user_nn_model_nn_file_name, rnn_layers);//����ģ��
			break;
		}
	}
	int offset = 0;
	while (topic_score[offset] == 't') {
		topic_score[offset + 2] = 0;//���ɴ�
		offset += topic_score[offset + 3] + 4;//�����µ���Ŀ��ʼλ��
	}
	int test_count = 0, test_error = 0;
	int t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	while (1) {
		_nn_topic_increase_answer(topic_score);//˳���
		//_nn_topic_random_answer(topic_score);//�����
		topic_matrix = _nn_topic_abswer_to_matrix(topic_score, input_data_type);//�Ѵ���ת������
		score_matrix = _nn_topic_score_to_matrix(topic_score);//���ɽ�������ֶξ���
		user_nn_model_load_input_feature(rnn_layers, topic_matrix);//������������
		user_nn_model_ffp(rnn_layers);
		if (user_nn_matrix_return_max_index(user_nn_model_return_result(rnn_layers)) != user_nn_matrix_return_max_index(score_matrix)) {
//			user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));//��ӡ
//			user_nn_matrix_printf(NULL, score_matrix);//��ӡ
			test_error++;
			if (user_nn_matrix_return_max_index(score_matrix) == 0) {
				t1++;
			}
			else if (user_nn_matrix_return_max_index(score_matrix) == 1) {
				t2++;
			}
			else if (user_nn_matrix_return_max_index(score_matrix) == 2) {
				t3++;
			}
			else if (user_nn_matrix_return_max_index(score_matrix) == 3) {
				t4++;
			}
		}
		else {
			//user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));//��ӡ
			//user_nn_matrix_printf(NULL, score_matrix);//��ӡ		
		}

		user_nn_matrix_delete(topic_matrix);//ɾ������
		user_nn_matrix_delete(score_matrix);//ɾ������
		if (++test_count >= 390625) {
			break;
		}
	}
	printf("\nt1:%d t2:%d t3:%d t4:%d", t1,t2,t3,t4);
	printf("\ncount��%d error:%d total:%d > %f%%", train_max_count, test_error, test_count, ((float)1 - (float)test_error / test_count) * 100);

}

