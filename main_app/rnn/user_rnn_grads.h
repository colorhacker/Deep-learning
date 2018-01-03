#ifndef _user_rnn_grads_H
#define _user_rnn_grads_H

#include "../user_config.h"
#include "../rnn/user_rnn_layers.h"

void user_rnn_grads_hidden(user_rnn_layers *hidden_layer, float alpha);
void user_rnn_grads_output(user_rnn_layers *output_layer, float alpha);

#endif