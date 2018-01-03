#ifndef _user_earth_movers_distance_H
#define _user_earth_movers_distance_H

#include "../user_config.h"
#include "../matrix/user_nn_matrix.h"

#define emd_potential_accuracy 0.000001

float user_emd_matrix_get_cost_value(user_nn_matrix *src_matrix, user_nn_matrix *sub_matrix);

void user_emd_matrix_set_row_col(user_nn_matrix *save_matrix, float value);
void user_emd_matrix_zero_mapping_cpy(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix);
void user_emd_matrix_unzero_mapping_cpy(user_nn_matrix *save_matrix, user_nn_matrix *sub_matrix);

user_nn_matrix *user_emd_object_create(float *dist_array,float *height_array,int height,float *output_array,int width);//创建一个emd结构矩阵
float user_emd_vogel_plan_init_martix(user_nn_matrix *src_matrix, user_nn_matrix *tra_matrix);//通过矩阵计算一次emd距离
user_nn_matrix *user_emd_get_vogel_init_martix(user_nn_matrix *src_matrix);
user_nn_matrix *user_emd_potential_plan_matrix(user_nn_matrix *src_matrix);
user_nn_matrix *user_emd_censor_potential_matrix(user_nn_matrix *src_matrix);

void user_emd_adjust_matrix_value(user_nn_matrix *src_matrix, user_nn_matrix *path_matix);
user_nn_matrix *user_emd_get_loop_path_list(user_nn_matrix *maps_matrix, int centor_point);
int user_emd_check_vaild_point(user_nn_matrix *path_matrix, int *around_array);
int user_emd_get_center_around_point(user_nn_matrix *matrix, int obstacle_point, int center_point, int *around_array);

float user_emd_earth_movers_distance(float *dist_array, float *height_array, int height, float *width_array, int width);
#endif