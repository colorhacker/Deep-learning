#ifndef _user_nn_app_H
#define _user_nn_app_H

#include "user_nn_layers.h"
#include "user_nn_ffp.h"
#include "user_nn_bp.h"
#include "user_nn_grads.h"
#include "user_nn_model.h"
#include "user_nn_rw_model.h"

typedef enum _abswer_type {
	abswer_bin = 0,//
	abswer_hex = 1//
}abswer_type;

void user_nn_app_test(int argc, const char** argv);
void user_nn_app_topic(int argc, const char** argv);

#endif