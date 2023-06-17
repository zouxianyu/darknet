#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "layer.h"
#include "network.h"

layer make_flatten_layer(int batch, int inputs);

void forward_flatten_layer(layer l, network net);
void backward_flatten_layer(layer l, network net);

#ifdef GPU
void forward_flatten_layer_gpu(layer l, network net);
void backward_flatten_layer_gpu(layer l, network net);
#endif

#endif

