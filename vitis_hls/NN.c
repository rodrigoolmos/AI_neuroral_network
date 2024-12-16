#include "NN.h"

void execute_NN(struct NN neurons, float features[MAX_FEATURES], 
                uint32_t use_features, uint32_t n_layers, float *prediction){

    float nodes_values[MAX_FEATURES];
    uint32_t weight_i;
    uint32_t nodes_i;
    uint32_t layers_i;

    for (nodes_i = 0; nodes_i < use_features; nodes_i++){
        
        nodes_values[nodes_i] = neurons.offsets[0][nodes_i];

        for (weight_i = 0; weight_i < use_features; weight_i++){
            nodes_values[nodes_i] += neurons.weights[0][nodes_i][weight_i] * 
                                        features[weight_i];
        }
        nodes_values[nodes_i] = nodes_values[nodes_i] >= -1 && nodes_values[nodes_i] <= 1 ? nodes_values[nodes_i] : 0;

    }    
    
    for ( layers_i = 1; layers_i < N_LAYERS ; layers_i++){
        for (nodes_i = 0; nodes_i < use_features; nodes_i++){

            nodes_values[nodes_i] += neurons.offsets[layers_i][nodes_i];

            for (weight_i = 0; weight_i < use_features; weight_i++){
                nodes_values[nodes_i] += neurons.weights[layers_i][nodes_i][weight_i] * 
                                            nodes_values[weight_i];
            }

            nodes_values[nodes_i] = nodes_values[nodes_i] >= -1 && nodes_values[nodes_i] <= 1 ? nodes_values[nodes_i] : 0;

        }  
    }

    *prediction = 0;
    for (weight_i = 0; weight_i < use_features; weight_i++){
        (*prediction) += nodes_values[weight_i];
    }

}