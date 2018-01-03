#ifndef _user_cnn_grads_H
#define _user_cnn_grads_H

#include "../user_config.h"
#include "../cnn/user_cnn_layers.h"

void user_cnn_grads_convolution(user_cnn_layers *conv_layer, float alpha);
void user_cnn_grads_full(user_cnn_layers *full_layer, float alpha);
void user_cnn_grads_output(user_cnn_layers *output_layer, float alpha);

#endif