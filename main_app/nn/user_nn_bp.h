#ifndef _user_nn_bp_H
#define _user_nn_bp_H

#include "../user_config.h"
#include "../nn/user_nn_layers.h"


float user_nn_bp_output_back_prior(user_nn_layers *prior_layer, user_nn_layers *output_layer);
void user_nn_bp_hidden_back_prior(user_nn_layers *prior_layer, user_nn_layers *hidden_layer);

#endif