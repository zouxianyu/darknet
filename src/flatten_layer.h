#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_flatten_layer(int batch, int inputs);

void forward_flatten_layer(layer l, network_state state);
void backward_flatten_layer(layer l, network_state state);

#ifdef GPU
void forward_flatten_layer_gpu(layer l, network_state state);
void backward_flatten_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif

#endif
