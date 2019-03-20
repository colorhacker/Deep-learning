#include "user_config.h"

#include "matrix/user_nn_matrix.h"
#include "matrix/user_nn_activate.h"
#include "matrix/user_nn_matrix_cuda.h"

#include "nn\user_nn_app.h"
#include "rnn\user_rnn_app.h"
#include "cnn\user_cnn_app.h"
#include "w2c\user_w2c_app.h"
#include "mnist\user_mnist.h"

int main(int argc, const char** argv){
	user_nn_app_train(NULL,NULL);
	getch();
	return 0;
}
