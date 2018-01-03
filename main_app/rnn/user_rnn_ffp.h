#ifndef _user_rnn_ffp_H
#define _user_rnn_ffp_H

#include "../user_config.h"
#include "../rnn/user_rnn_layers.h"

void user_rnn_ffp_hidden(user_rnn_layers *prior_layer, user_rnn_layers *hidden_layer);
void user_rnn_ffp_output(user_rnn_layers *prior_layer, user_rnn_layers *output_layer);

#endif