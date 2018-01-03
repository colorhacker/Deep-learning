
#include "user_w2c_app.h"


void user_w2c_app_test(int argc, const char** argv) {
	user_w2v_words_vector *model = load_words_vector_model("word2vec_model.bin");//加载模型
	user_w2v_similar_list *result_list = user_w2v_words_most_similar(model, euclidean, "小龙女", 5.0f);//计算大于阈值的字符串对象并且提取出来
	printf("\n");
	user_w2v_similar_list_printf(false, result_list);//打印数据
	result_list = user_w2v_similar_sorting(result_list, sorting_down);//降序排列将会删除结果
	float dist = user_w2v_words_similar(model, cosine, "的", "了");//计算两个字符串之间的距离
	printf("\n dist=%f\n", dist);
	user_w2v_similar_list_printf(false, result_list);//打印数据

	user_w2v_words_vector_all_delete(model);//删除模型
	user_w2v_similar_list_all_delete(result_list);//删除结果链表

	getchar();
}