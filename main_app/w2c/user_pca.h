#ifndef _user_pca_H
#define _user_pca_H

#include <string.h>  
#include <math.h>  
#include <malloc.h>  
#include <stdio.h>  
#include "../matrix/user_nn_matrix.h"

#define USER_PCA_EIGS_EPSILON 0.0f

user_nn_list_matrix *user_pca_process(user_nn_matrix *src_matrix, float epsilon, eigs_type type);//PACΩµµÕŒ¨∂»À„∑®


#endif
