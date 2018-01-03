#ifndef _user_nn_ffp_H
#define _user_nn_ffp_H

#include "../user_config.h"
#include "../nn/user_nn_layers.h"

void user_nn_ffp_hidden(user_nn_layers *prior_layer, user_nn_layers *hidden_layer);
void user_nn_ffp_output(user_nn_layers *prior_layer, user_nn_layers *output_layer);

#endif