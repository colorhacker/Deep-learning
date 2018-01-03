#ifndef _user_cnn_matrix_cuda_H
#define _user_cnn_matrix_cuda_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"

#define TILE_DIM	16
#define BLOCK_ROWS	16

extern "C"{
	void user_nn_matrix_transpose_cuda(user_nn_matrix *src_matrix);
}
#ifdef WIN64
user_nn_matrix *user_nn_matrix_mult_matrix_cuda(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);
#endif
#endif