#ifndef _user_nn_grads_H
#define _user_nn_grads_H

#include "../user_config.h"
#include "../nn/user_nn_layers.h"

void user_nn_grads_hidden(user_nn_layers *hidden_layer, float alpha);
void user_nn_grads_output(user_nn_layers *output_layer, float alpha);

#endif