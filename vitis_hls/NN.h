#include <stdint.h>

#define N_NEURONS 1000
#define MAX_FEATURES 32

struct NN {
    float weights[N_NEURONS];
    float offsets[N_NEURONS];
};


void execute_NN(struct NN neurons, float features[MAX_FEATURES], 
                uint32_t use_features, uint32_t n_layers, float *prediction);