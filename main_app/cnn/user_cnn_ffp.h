#ifndef _user_cnn_ffp_H
#define _user_cnn_ffp_H

#include "../user_config.h"
#include "../cnn/user_cnn_layers.h"

void user_cnn_ffp_convolution(user_cnn_layers *prior_layer, user_cnn_layers *conv_layer);
void user_cnn_ffp_pooling(user_cnn_layers *prior_layer, user_cnn_layers *pool_layer);
void user_cnn_ffp_fullconnect(user_cnn_layers *prior_layer, user_cnn_layers *full_layer);
void user_cnn_ffp_output(user_cnn_layers *prior_layer, user_cnn_layers *output_layer);

#endif