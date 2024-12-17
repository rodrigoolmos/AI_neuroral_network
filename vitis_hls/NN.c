#include "NN.h"

void activation_fun(float *value) {
    (*value) = ((*value) > -1 && (*value) < 2) ? (*value) : 0;
}

void execute_NN(struct NN neurons, float features[MAX_FEATURES], 
                uint32_t use_features, uint32_t n_layers, float *prediction) {
    
    float current_values[MAX_FEATURES];
    float next_values[MAX_FEATURES];
    uint32_t weight_i;
    uint32_t nodes_i;
    uint32_t layers_i;

    // Calcular la salida de la primera capa (capa 0)
    for (nodes_i = 0; nodes_i < use_features; nodes_i++) {
        float sum = neurons.offsets[0][nodes_i];
        for (weight_i = 0; weight_i < use_features; weight_i++) {
            sum += neurons.weights[0][nodes_i][weight_i] * features[weight_i];
        }
        activation_fun(&sum);
        current_values[nodes_i] = sum;
    }

    // Calcular la salida de las capas ocultas y/o finales, desde la 1 hasta n_layers-1
    for (layers_i = 1; layers_i < n_layers; layers_i++) {
        for (nodes_i = 0; nodes_i < use_features; nodes_i++) {
            float sum = neurons.offsets[layers_i][nodes_i];
            for (weight_i = 0; weight_i < use_features; weight_i++) {
                sum += neurons.weights[layers_i][nodes_i][weight_i] * current_values[weight_i];
            }
            activation_fun(&sum);
            next_values[nodes_i] = sum;
        }
        // Copiamos next_values a current_values para la siguiente capa
        for (nodes_i = 0; nodes_i < use_features; nodes_i++) {
            current_values[nodes_i] = next_values[nodes_i];
        }
    }

    // La predicción se calcula sumando las salidas de la última capa
    *prediction = 0.0f;
    for (weight_i = 0; weight_i < use_features; weight_i++) {
        (*prediction) += current_values[weight_i];
    }
}
