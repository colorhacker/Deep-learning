
#include "user_nn_app.h"

void user_nn_app_test(int argc, const char** argv) {

	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 2, //输入层 特征（宽度、高度、时间长度）
		'h', 2, //隐含层 特征 （高度、时间长度）
		'o', 2 //输出层 特征 （高度、时间长度）
	};
	user_nn_input_layers	*nn_input_layers = NULL;
	user_nn_hidden_layers	*nn_hidden_layers = NULL;
	user_nn_output_layers	*nn_output_layers = NULL;

	float loss_function = 0.0f;
	bool model_is_exist = false;
	user_nn_matrix *input_data = user_nn_matrix_create(1, 2);//创建输入数据
	user_nn_matrix *target_data = user_nn_matrix_create(1, 2);//创建输入数据

	user_nn_layers *rnn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//载入模型
	if (rnn_layers == NULL) {
		printf("loading model failed\ncreate cnn new object \n");
		rnn_layers = user_nn_model_create(user_layers);//创建模型
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
		user_nn_model_load_input_feature(rnn_layers, input_data);//加载输入数据
		user_nn_model_load_target_feature(rnn_layers, target_data);//记载目标数据
																   //正向计算一次 按时间片迭代N此
		user_nn_model_ffp(rnn_layers);
		//反向计算一次 按时间片迭代N此
		user_nn_model_bp(rnn_layers, 0.01f);
		loss_function = user_nn_model_return_loss(rnn_layers);
		if (loss_function <= 0.001f) {
			user_nn_model_save_model(user_nn_model_nn_file_name, rnn_layers);//保存模型
			break;
		}
		printf("\nloss:%f", loss_function);
	}
	user_nn_model_load_input_feature(rnn_layers, input_data);//加载输入数据
	user_nn_model_ffp(rnn_layers);//进行计算
	user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));

	getchar();
}
//随机一个数 随机 start <=> end-1
float _nn_topic_random(int max)
{
	//float r = (float)(rand() / ((RAND_MAX / max) + 1));
	return (float)(rand() / ((RAND_MAX / max) + 1));
}

//递增答案 按照二进制方式递增无重复
int _nn_topic_increase_answer(char *topic_score) {
	int score = 0, offset = 0,select_count=0;//答题选择、得分、答题偏移
	if (topic_score[offset] == 't') {
		topic_score[offset + 2]++;//进行计数增加
	}
	while (topic_score[offset] == 't') {
		select_count = topic_score[offset + 3];//记录有多少个选项
		switch (topic_score[offset + 1]) {//判断类型
		case 'x'://x类型答案
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
		score += topic_score[offset + topic_score[offset + 2] + 4];//累计积分
		offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		//printf("offset:%d,select:%d,score:%d\n", offset,select, score);
	}
	if (topic_score[offset - select_count - 2] >= select_count) {
		offset = 0;
		while (topic_score[offset] == 't') {
			topic_score[offset + 2] = 0;//清空数据
			offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		}
	}
	return score;
}
//随机答案 按题目随机值进行设置答案
int _nn_topic_random_answer(char *topic_score) {
	int score = 0, offset = 0;//答题选择、得分、答题偏移
	while (topic_score[offset] == 't') {
		switch (topic_score[offset + 1]) {//判断类型
			case 'x'://x类型答案
				topic_score[offset + 2] = (int)_nn_topic_random(topic_score[offset + 3]);//获取所设置答案
				//printf(" %d",select);
			default:break;
		}
		score += topic_score[offset + topic_score[offset + 2] + 4];//累计积分
		offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		//printf("offset:%d,select:%d,score:%d\n", offset,select, score);
	}
	return score;
}
//把答案生成矩阵 按照二进制方式或者题目方式
user_nn_matrix *_nn_topic_abswer_to_matrix(char *topic_score, abswer_type type) {
	int offset = 0,offset_score = 0,total_size = 0;//答题选择、得分、答题偏移
	char *topic_ram = NULL;
	user_nn_matrix *result = NULL;
	float *result_data = NULL;
	if (type == abswer_hex) {
		offset = 0;
		while (topic_score[offset] == 't') {
			total_size ++;//记录个数
			offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		}
		offset = 0;
		result = user_nn_matrix_create(1, total_size);//生成矩阵
		result_data = result->data;
		while (topic_score[offset] == 't') {
			*result_data++ = topic_score[offset + 2] * 0.1f;
			offset_score += topic_score[offset + 3];//添加
			offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		}
	}
	else if (type == abswer_bin) {
		offset = 0;
		while (topic_score[offset] == 't') {
			total_size += topic_score[offset + 3];//记录个数
			offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		}
		offset = 0;
		result = user_nn_matrix_create(1, total_size);//生成矩阵
		result_data = result->data;
		while (topic_score[offset] == 't') {
			result_data[offset_score + topic_score[offset + 2]] = 1.0f;//生成答案
			offset_score += topic_score[offset + 3];//添加
			offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
		}
	}

	return result;
}
//把得分生成矩阵
user_nn_matrix *_nn_topic_score_to_matrix(char *topic_score) {
	int score = 0, offset = 0,target = 0;//答题选择、得分、答题偏移
	user_nn_matrix *result = NULL;
	float *result_data=NULL;
	while (topic_score[offset] == 't') {
		score += topic_score[offset + topic_score[offset + 2] + 4];//累计积分
		offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
	}
	if (topic_score[offset] == 's') {
		target = topic_score[offset + 3];
		result = user_nn_matrix_create(1, target);//创建矩阵
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
//测试
/*int score = 0,count = 0;
user_nn_list_matrix *topic_matrices = user_nn_matrices_create_head(1, 1);//创建输入数据
user_nn_list_matrix *score_matrices = user_nn_matrices_create_head(1, 1);//创建输入数据
while (count++ < 100) {
score = _nn_topic_random_answer(topic_score);//随机答案
user_nn_matrix *topic_matrix = _nn_topic_abswer_to_matrix(topic_score);//把答题转化矩阵
user_nn_matrix *score_matrix = _nn_topic_score_to_matrix(topic_score);//生成结果分数分段矩阵
//user_nn_matrices_add_matrix(topic_matrices, topic_matrix);
//user_nn_matrices_add_matrix(score_matrices, score_matrix);
}*/
//获取字符串网络层
int *user_nn_get_layer_array(int layer,int *buffer,int size) {
	while (size--) {
		if (*buffer == layer) {
			return buffer;
		}
		buffer++;
	}
	return NULL;
}
//答题训练计分
void user_nn_app_topic(int argc, const char** argv) {
	char topic_score[] = {
		//'t'起始,'x'答案类型,'n'所选答案,5后续答案的个数,...答案得分
		't','x',NULL,5,4,3,2,1,0,//第一列随机最大值 第二列保存选择的答案 第三列题目最大选项,第四项保存多少个答案,后面保存得分值
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		't','x',NULL,5,4,3,2,1,0,
		's','x',NULL,4,10,20,28,40
	};
	//srand((unsigned)time(NULL));//随机种子 ----- 若不设置那么每次训练结果一致
	int user_layers[] = {
		'i', 1, 0, //输入层 特征（宽度、高度）
		'h', 40, //隐含层 特征 （长度）
		'h', 80, //隐含层 特征 （长度）
		'h', 40, //隐含层 特征 （长度）
		'o', 0 //输出层 特征 （长度）
	};
	user_nn_input_layers	*nn_input_layers = NULL;
	user_nn_hidden_layers	*nn_hidden_layers = NULL;
	user_nn_output_layers	*nn_output_layers = NULL;
	user_nn_matrix *topic_matrix = NULL;
	user_nn_matrix *score_matrix = NULL;
	float loss_function = 0.0f;
	bool model_is_exist = false;
	abswer_type input_data_type = abswer_hex;//设置输入类型

	user_nn_layers *rnn_layers = user_nn_model_load_model(user_nn_model_nn_file_name);//载入模型
	if (rnn_layers == NULL) {
		printf("loading model failed\ncreate nn new object \n");
		topic_matrix = _nn_topic_abswer_to_matrix(topic_score, input_data_type);//把答题转化矩阵
		*(user_nn_get_layer_array('i', user_layers, sizeof(user_layers)) + 1) = topic_matrix->width;//加载当前高度
		*(user_nn_get_layer_array('i', user_layers, sizeof(user_layers)) + 2) = topic_matrix->height;//加载当前高度
		score_matrix = _nn_topic_score_to_matrix(topic_score);//生成结果分数分段矩阵
		*(user_nn_get_layer_array('o', user_layers, sizeof(user_layers)) + 1) = score_matrix->height;//加载当前高度
		rnn_layers = user_nn_model_create(user_layers);//创建模型
		user_nn_matrix_delete(topic_matrix);//删除矩阵
		user_nn_matrix_delete(score_matrix);//删除矩阵
		model_is_exist = false;
	}
	else {
		printf("loading model success\n");
		model_is_exist = true;
	}
	int train_max_count = 0;
	while (!model_is_exist) {
		_nn_topic_increase_answer(topic_score);//顺序答案
		//_nn_topic_random_answer(topic_score);
		topic_matrix = _nn_topic_abswer_to_matrix(topic_score, input_data_type);//把答题转化矩阵
		score_matrix = _nn_topic_score_to_matrix(topic_score);//生成结果分数分段矩阵
		user_nn_model_load_input_feature(rnn_layers, topic_matrix);//加载输入数据
		user_nn_model_load_target_feature(rnn_layers, score_matrix);//加载目标数据
		//正向计算一次 按时间片迭代N此
		user_nn_model_ffp(rnn_layers);
		//反向计算一次 按时间片迭代N此
		user_nn_model_bp(rnn_layers, 0.001f);
		loss_function = user_nn_model_return_loss(rnn_layers);
		//user_nn_matrix_printf(NULL, score_matrix);//打印
		user_nn_matrix_delete(topic_matrix);//删除矩阵
		user_nn_matrix_delete(score_matrix);//删除矩阵
		printf("train_max_count:%d loss:%f\n", train_max_count, loss_function);
		if ((++train_max_count >= 100000) && (loss_function <= 0.1f)) {
			user_nn_model_save_model(user_nn_model_nn_file_name, rnn_layers);//保存模型
			break;
		}
	}
	int offset = 0;
	while (topic_score[offset] == 't') {
		topic_score[offset + 2] = 0;//生成答案
		offset += topic_score[offset + 3] + 4;//设置新的题目起始位置
	}
	int test_count = 0, test_error = 0;
	int t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	while (1) {
		_nn_topic_increase_answer(topic_score);//顺序答案
		//_nn_topic_random_answer(topic_score);//随机答案
		topic_matrix = _nn_topic_abswer_to_matrix(topic_score, input_data_type);//把答题转化矩阵
		score_matrix = _nn_topic_score_to_matrix(topic_score);//生成结果分数分段矩阵
		user_nn_model_load_input_feature(rnn_layers, topic_matrix);//加载输入数据
		user_nn_model_ffp(rnn_layers);
		if (user_nn_matrix_return_max_index(user_nn_model_return_result(rnn_layers)) != user_nn_matrix_return_max_index(score_matrix)) {
//			user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));//打印
//			user_nn_matrix_printf(NULL, score_matrix);//打印
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
			//user_nn_matrix_printf(NULL, user_nn_model_return_result(rnn_layers));//打印
			//user_nn_matrix_printf(NULL, score_matrix);//打印		
		}

		user_nn_matrix_delete(topic_matrix);//删除矩阵
		user_nn_matrix_delete(score_matrix);//删除矩阵
		if (++test_count >= 390625) {
			break;
		}
	}
	printf("\nt1:%d t2:%d t3:%d t4:%d", t1,t2,t3,t4);
	printf("\ncount：%d error:%d total:%d > %f%%", train_max_count, test_error, test_count, ((float)1 - (float)test_error / test_count) * 100);

}

