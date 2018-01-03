#ifndef _user_rnn_bp_H
#define _user_rnn_bp_H

#include "../user_config.h"
#include "../rnn/user_rnn_layers.h"


float user_rnn_bp_output_back_prior(user_rnn_layers *prior_layer, user_rnn_layers *output_layer);
void user_rnn_bp_hidden_back_prior(user_rnn_layers *prior_layer, user_rnn_layers *hidden_layer);

#endif