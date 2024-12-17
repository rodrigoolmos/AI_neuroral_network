#include "train_NN.h"

float generate_random_float(float min, float max) {
    float random = (float)rand() / (float)RAND_MAX; // Valor entre 0 y 1
    float distance = max - min;
    return min + random * distance; 
}


void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void swap_NN(struct NN *neurons1, struct NN *neurons2){

    struct NN neurons_tmp;
    memcpy(&neurons_tmp, neurons1, sizeof(struct NN));
    memcpy(neurons1, neurons2, sizeof(struct NN));
    memcpy(neurons2, &neurons_tmp, sizeof(struct NN));
}

int partition(float population_accuracy[POPULATION], 
              struct NN neurons[POPULATION], int low, int high) {
                
    float pivot = population_accuracy[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (population_accuracy[j] > pivot) {
            i++;
            swapf(&population_accuracy[i], &population_accuracy[j]);
            swap_NN(&neurons[i], &neurons[j]);
        }
    }
    swapf(&population_accuracy[i + 1], &population_accuracy[high]);
    swap_NN(&neurons[i + 1], &neurons[high]);
    return i + 1;
}

void quicksort(float population_accuracy[POPULATION], 
               struct NN neurons[POPULATION], int low, int high) {

    if (low < high) {
        int pi = partition(population_accuracy, neurons, low, high);

        quicksort(population_accuracy, neurons, low, pi - 1);
        quicksort(population_accuracy, neurons, pi + 1, high);
    }
}

void reorganize_population(float population_accuracy[POPULATION], 
                    struct NN neurons[POPULATION]) {
    quicksort(population_accuracy, neurons, 0, POPULATION - 1);
}

void generate_rando_NN(struct NN neurons[POPULATION],
                    uint8_t n_features, float max_features[N_FEATURE], float min_features[N_FEATURE]) {

    srand(clock());
    uint32_t weight_i;
    uint32_t nodes_i;
    uint32_t layers_i;
    float range, div, abs_max, abs_min;

    for (int population_i = 0; population_i < POPULATION; population_i++){
        for (layers_i = 0; layers_i < N_LAYERS; layers_i++){
            for (nodes_i = 0; nodes_i < n_features; nodes_i++){
                for (weight_i = 0; weight_i < n_features; weight_i++){
                    if (layers_i == 0 ){
                        abs_max = fabs(max_features[weight_i]);
                        abs_min = fabs(min_features[weight_i]);
                        div = abs_max > abs_min ? abs_max : abs_min;
                        range = max_features[weight_i] == 0 ? 0 : 1/div;
                        neurons[population_i].weights[layers_i][nodes_i][weight_i] = 
                                                                    generate_random_float(-range, range)+
                                                                    generate_random_float(-range, range);
                    }else{
                        neurons[population_i].weights[layers_i][nodes_i][weight_i] = generate_random_float(-0.1, 0.1);
                    }
                }
                neurons[population_i].offsets[layers_i][nodes_i] = generate_random_float(-10, 10);
            }
        }
    }
}

 void mutate_NN(struct NN *neurons_input, struct NN *neurons_output,
                  uint8_t n_features, float mutation_rate, float max_features[N_FEATURE], 
                  float min_features[N_FEATURE]) {

    uint32_t mutation_threshold = mutation_rate * RAND_MAX;
    uint32_t mutation_value;
    uint32_t weight_i;
    uint32_t nodes_i;
    uint32_t layers_i;
    float range, div, abs_max, abs_min;
    memcpy(neurons_output, neurons_input, sizeof(struct NN));

    for (layers_i = 0; layers_i < N_LAYERS; layers_i++){
        for (nodes_i = 0; nodes_i < n_features; nodes_i++){
            mutation_value = rand();
            if (mutation_value < mutation_threshold){
                for (weight_i = 0; weight_i < n_features; weight_i++){
                        if (layers_i == 0 ){
                            abs_max = fabs(max_features[weight_i]);
                            abs_min = fabs(min_features[weight_i]);
                            div = abs_max > abs_min ? abs_max : abs_min;
                            range = max_features[weight_i] == 0 ? 0 : 1/div;
                            neurons_output->weights[layers_i][nodes_i][weight_i] = 
                                                                        generate_random_float(-range, range)+
                                                                        generate_random_float(-range, range);

                        }else{
                            neurons_output->weights[layers_i][nodes_i][weight_i] = generate_random_float(-0.1, 0.1);
                        }
                }
               
                neurons_output->offsets[layers_i][nodes_i] = generate_random_float(-10, 10);
            }
        }
    }
 }

 void tune_NN(struct NN *neurons_input, struct NN *neurons_output,
                  uint8_t n_features, float mutation_rate, float max_features[N_FEATURE], 
                  float min_features[N_FEATURE]) {

    uint32_t mutation_threshold = mutation_rate * RAND_MAX;
    uint32_t mutation_value;
    uint32_t weight_i;
    uint32_t nodes_i;
    uint32_t layers_i;
    float actual_value;
    memcpy(neurons_output, neurons_input, sizeof(struct NN));

    for (layers_i = 1; layers_i < N_LAYERS; layers_i++){
        for (nodes_i = 0; nodes_i < n_features; nodes_i++){
            mutation_value = rand();
            if (mutation_value < mutation_threshold){
                for (weight_i = 0; weight_i < n_features; weight_i++){
                    
                    actual_value = neurons_output->weights[layers_i][nodes_i][weight_i];

                    neurons_output->weights[layers_i][nodes_i][weight_i] += 
                            generate_random_float(actual_value, actual_value);
                }
               
                actual_value = neurons_output->offsets[layers_i][nodes_i];
                neurons_output->offsets[layers_i][nodes_i] += 
                            generate_random_float(actual_value, actual_value);
            }
        }
    }
 }

void mutate_population(struct NN neurons[POPULATION], float population_accuracy[POPULATION], 
                        float max_features[N_FEATURE], float min_features[N_FEATURE], 
                        uint8_t n_features, float mutation_factor){

    uint32_t mutation_part = POPULATION/4;


    for (uint32_t p = mutation_part; p < POPULATION; p++) {
        int index_elite = rand() % (mutation_part);
        int threshold = (int)((mutation_part/2)* population_accuracy[index_elite]);

        if (index_elite < threshold){
            tune_NN(&neurons[index_elite], &neurons[p], n_features, 
                        1 - population_accuracy[p] - 0.6, max_features, min_features);            
        }else{
            mutate_NN(&neurons[index_elite], &neurons[p], n_features, 
                        1 - population_accuracy[p], max_features, min_features);
        }
    }

}

void swap_features(struct feature* a, struct feature* b) {
    struct feature temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle(struct feature* array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap_features(&array[i], &array[j]);
    }
}

void evaluate_model(struct NN neurons, 
                    struct feature *features, int read_samples, uint32_t n_layers, 
                    float *accuracy, uint8_t sow_log, uint32_t used_features){

    int correct = 0;
    float prediction;

    for (int i = 0; i < read_samples; i++){
        execute_NN(neurons, features[i].features, used_features, n_layers, &prediction);
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

        for (int32_t p = POPULATION/400; p >= 0; p--){
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

void crossover_NN(struct NN *parent1, struct NN *parent2, struct NN *child, uint8_t n_features) {

    for (uint32_t layer = 0; layer < N_LAYERS; layer++) {
        for (uint32_t node = 0; node < n_features; node++) {

            if (rand() % 2) {  
                child->offsets[layer][node] = parent1->offsets[layer][node];
            } else {
                child->offsets[layer][node] = parent2->offsets[layer][node];
            }

            for (uint32_t weight = 0; weight < n_features; weight++) {
                if (rand() % 2) {
                    child->weights[layer][node][weight] = parent1->weights[layer][node][weight];
                } else {
                    child->weights[layer][node][weight] = parent2->weights[layer][node][weight];
                }
            }
        }
    }
}

void crossover_population(struct NN neurons[POPULATION], uint8_t n_features){

    for (uint32_t p = POPULATION - POPULATION/10; p < POPULATION; p++){
        int index_mother = rand() % (POPULATION/80);
        int index_father = rand() % (POPULATION/80) + POPULATION/80;

        crossover_NN(&neurons[index_mother], &neurons[index_father],
                                &neurons[p], n_features);
    }

}
