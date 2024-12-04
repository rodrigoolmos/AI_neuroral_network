#include "NN.h"

void execute_NN(struct NN neurons, float features[MAX_FEATURES], 
                uint32_t use_features, uint32_t n_layers, float *prediction){

    float nodes_values[MAX_FEATURES];
    uint32_t feature_i;
    uint32_t nodes_i;
    uint32_t layers_i;

    for (feature_i = 0; feature_i < use_features; feature_i++){
        nodes_values[feature_i] = features[feature_i];
    }

    for (layers_i = 0; layers_i < n_layers; layers_i++){
        for (nodes_i = 0; nodes_i < use_features; nodes_i++){
            for (feature_i = 1; feature_i < use_features; feature_i++){
                nodes_values[nodes_i] *= (nodes_values[feature_i] * 
                            neurons.weights[layers_i * use_features * use_features + nodes_i * use_features + feature_i] + 
                            neurons.offsets[layers_i * use_features * use_features + nodes_i * use_features + feature_i]);
            }
        }
    }

    *prediction = 1;
    for (feature_i = 0; feature_i < use_features; feature_i++){
        *prediction *= (nodes_values[feature_i] * 
                    neurons.weights[layers_i * use_features * use_features + nodes_i * use_features + feature_i] + 
                    neurons.offsets[layers_i * use_features * use_features + nodes_i * use_features + feature_i]);
    }

}