#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 1024
#define MAX_TEST_SAMPLES 30000
#define N_FEATURE 32

/**
 * @brief Represents a set of features along with the prediction result.
 * 
 * This structure holds an array of features and a prediction result. It is used for storing the feature
 * data of a sample and the corresponding prediction made by a model.
 * 
 * @var feature::features
 * Array containing the feature values for a given sample. The size of the array is N_FEATURE.
 * 
 * @var feature::prediction
 * The prediction result for the sample. It is represented as an 8-bit unsigned integer.
 */
struct feature {
    float features[N_FEATURE];
    uint8_t prediction;
};


int read_n_features(const char *csv_file, int n, struct feature *features, uint32_t *features_length);