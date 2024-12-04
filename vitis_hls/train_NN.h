#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <omp.h>
#include "NN.h"
#include "common.h"

#define POPULATION 1024
#define N_FEATURE 32

void generate_rando_NN(struct NN neurons[POPULATION],
                    uint8_t n_features, float max_features[N_FEATURE], 
                    float min_features[N_FEATURE]);

void mutate_population(struct NN neurons[POPULATION], float population_accuracy[POPULATION], 
                        float max_features[N_FEATURE], float min_features[N_FEATURE], 
                        uint8_t n_features, float mutation_factor);

void evaluate_model(struct NN neurons, 
                    struct feature *features, int read_samples, 
                    float *accuracy, uint8_t sow_log, uint32_t used_features);

void show_logs(float population_accuracy[POPULATION]);

void reorganize_population(float population_accuracy[POPULATION], struct NN neurons[POPULATION]);

void find_max_min_features(struct feature features[MAX_TEST_SAMPLES],
                                float max_features[N_FEATURE], float min_features[N_FEATURE]);