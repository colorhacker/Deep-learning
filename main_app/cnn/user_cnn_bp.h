#ifndef _user_cnn_bp_H
#define _user_cnn_bp_H

#include "../user_config.h"
#include "../cnn/user_cnn_layers.h"


void user_cnn_bp_output_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *output_layer);
void user_cnn_bp_fullconnect_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *full_layer);
void user_cnn_bp_pooling_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *pool_layer);
void user_cnn_bp_convolution_back_prior(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer);

void user_cnn_bp_convolution_deltas_kernel(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer);
void user_cnn_bp_full_deltas_kernel(user_cnn_layers *full_layer);
void user_cnn_bp_output_deltas_kernel(user_cnn_layers *output_layer);


#endif