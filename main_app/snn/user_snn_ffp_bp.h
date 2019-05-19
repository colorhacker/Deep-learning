#ifndef _user_snn_ffp_bp_H
#define _user_snn_ffp_bp_H

#include "../user_config.h"
#include "user_snn_layers.h"

void user_snn_ffp_hidden(user_snn_layers *prior_layer, user_snn_layers *hidden_layer);
void user_snn_ffp_output(user_snn_layers *prior_layer, user_snn_layers *output_layer);

void user_snn_bp_output_back_prior(user_snn_layers *prior_layer, user_snn_layers *output_layer);
void user_snn_bp_hidden_back_prior(user_snn_layers *prior_layer, user_snn_layers *hidden_layer);

#endif