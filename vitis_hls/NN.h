#include <stdint.h>

#define MAX_FEATURES 32
#define N_LAYERS 3
#define N_WEIGHTS MAX_FEATURES
#define N_NEURONS MAX_FEATURES

struct NN {
    float weights[N_LAYERS][N_NEURONS][N_WEIGHTS];
    float offsets[N_LAYERS][N_NEURONS];
};


void execute_NN(struct NN neurons, float features[MAX_FEATURES], 
                uint32_t use_features, uint32_t n_layers, float *prediction);