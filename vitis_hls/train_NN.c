#include "train_NN.h"

float generate_random_float(float min, float max, int* seed) {
    float random = (float)rand_r(seed) / RAND_MAX;

    float distance = max - min;
    float step = distance * 0.01; // 1% de la distancia

    return min + round(random * (distance / step)) * step; // Multiplicamos por step para ajustar el paso
}

void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void swap_NN(struct NN neurons1, struct NN neurons2) {

    struct NN tmp_neurons;

    memcpy(&tmp_neurons, &neurons1, sizeof(struct NN ));
    memcpy(&neurons1, &neurons2, sizeof(struct NN ));
    memcpy(&neurons2, &tmp_neurons, sizeof(struct NN ));
}

int partition(float population_accuracy[POPULATION], struct NN neurons[POPULATION], 
              int low, int high) {

    float pivot = population_accuracy[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (population_accuracy[j] > pivot) {
            i++;
            swapf(&population_accuracy[i], &population_accuracy[j]);
            swap_NN(neurons[i], neurons[j]);
        }
    }
    swapf(&population_accuracy[i + 1], &population_accuracy[high]);
    swap_NN(neurons[i + 1], neurons[high]);
    return i + 1;
}


void quicksort(float population_accuracy[POPULATION], struct NN neurons[POPULATION],
               int low, int high){

    if (low < high) {
        int pi = partition(population_accuracy, neurons, low, high);

        quicksort(population_accuracy, neurons, low, pi - 1);
        quicksort(population_accuracy, neurons, pi + 1, high);
    }
}

void reorganize_population(float population_accuracy[POPULATION], struct NN neurons[POPULATION]) {

    quicksort(population_accuracy, neurons, 0, POPULATION - 1);
}

void generate_rando_NN(struct NN neurons[POPULATION],
                    uint8_t n_features, float max_features[N_FEATURE], float min_features[N_FEATURE]) {

    srand(clock());
    uint8_t n_feature;
    float max_value, min_value;
    int seed;

    #pragma omp parallel for schedule(static)
    for (int population_i = 0; population_i < POPULATION; population_i++){
        for (int neuron_i = 0; neuron_i < N_NEURONS; neuron_i++){
            seed = seed + omp_get_thread_num() + time(NULL) + population_i + neuron_i;
            if (neuron_i < n_features * n_features){
                max_value = max_features[neuron_i % n_features] == 0 ? 0 : 1/max_features[neuron_i % n_features];
                min_value = min_features[neuron_i % n_features] == 0 ? 0 : 1/min_features[neuron_i % n_features];
                neurons[population_i].weights[neuron_i] = generate_random_float(min_value, max_value, &seed);
                neurons[population_i].offsets[neuron_i] = generate_random_float(-1, 1, &seed+1);
            }else{
                neurons[population_i].weights[neuron_i] = generate_random_float(-1, 1, &seed);
                neurons[population_i].offsets[neuron_i] = generate_random_float(-1, 1, &seed);
            }
        }
    }
}

void mutate_NN(struct NN *neurons_input, struct NN *neurons_output,
                 uint8_t n_features, float mutation_rate, 
                 uint32_t n_neurons, float max_features[N_FEATURE], 
                 float min_features[N_FEATURE], int *seed) {

    uint32_t mutation_threshold = mutation_rate * RAND_MAX;
    memcpy(neurons_output, neurons_input, sizeof(struct NN));

    for (uint32_t neuron_i = 0; neuron_i < n_neurons; neuron_i++){
        uint32_t mutation_value = rand_r(seed);
        if (mutation_value < mutation_threshold){
            neurons_output->offsets[neuron_i] = generate_random_float(-1000, 1000, seed);
            neurons_output->weights[neuron_i] = generate_random_float(-1000, 1000, seed);
        }
    }

}

void mutate_population(struct NN neurons[POPULATION], float population_accuracy[POPULATION], 
                        float max_features[N_FEATURE], float min_features[N_FEATURE], 
                        uint8_t n_features, float mutation_factor){

    printf("Número de hilos: %d\n", omp_get_max_threads());

    #pragma omp parallel for schedule(static)
    for (uint32_t p = POPULATION/4; p < POPULATION; p++) {
        unsigned int seed = omp_get_thread_num() + time(NULL) + p;
        int index_elite = rand_r(&seed) % (POPULATION/4);

        struct NN local_neurons;
        memcpy(&local_neurons, &neurons[p], sizeof(struct NN));

    }
}

void evaluate_model(struct NN neurons, 
                    struct feature *features, int read_samples, 
                    float *accuracy, uint8_t sow_log, uint32_t used_features){

    int correct = 0;
    float prediction;


    for (int i = 0; i < read_samples; i++){
        execute_NN(neurons, features[i].features, used_features, 3, &prediction);
        if (features[i].prediction == (prediction > 0)){
            correct++;
        }
    }

    *accuracy = (float) correct / read_samples;
    if (sow_log)
        printf("Accuracy %f evaluates samples %i correct ones %i\n",
                        1.0 * (*accuracy), read_samples, correct);
    
}

void show_logs(float population_accuracy[POPULATION]){

        for (int32_t p = POPULATION/100; p >= 0; p--){
            printf("RANKING %i -> %f \t| RANKING %i -> %f \t| RANKING %i -> %f \t| RANKING %i -> %f| RANKING %i -> %f\n"
                            , p, population_accuracy[p]
                            , p + POPULATION/20, population_accuracy[p + POPULATION/20]
                            , p + POPULATION/10, population_accuracy[p + POPULATION/10]
                            , p + POPULATION/4 , population_accuracy[p + POPULATION/4]
                            , p + POPULATION/2 , population_accuracy[p + POPULATION/2]);
        }
}

void find_max_min_features(struct feature features[MAX_TEST_SAMPLES],
                                float max_features[N_FEATURE], float min_features[N_FEATURE]) {

    for (int j = 0; j < N_FEATURE; j++) {
        max_features[j] = features[0].features[j];
        min_features[j] = features[0].features[j];
    }

    for (int i = 1; i < MAX_TEST_SAMPLES; i++) {
        for (int j = 0; j < N_FEATURE; j++) {
            if (features[i].features[j] > max_features[j]) {
                max_features[j] = features[i].features[j];
            }
            if (features[i].features[j] < min_features[j]) {
                min_features[j] = features[i].features[j];
            }
        }
    }
}