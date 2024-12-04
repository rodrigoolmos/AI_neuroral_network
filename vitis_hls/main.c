#include <stdio.h>
#include "train_NN.h"


int main(){

    struct NN neurons[POPULATION];
    struct feature features[MAX_TEST_SAMPLES];
    float population_accuracy[POPULATION] = {0};
    float max_features[N_FEATURE];
    float min_features[N_FEATURE];
    int n_features;
    uint32_t use_features;
    uint32_t n_layers;
    uint32_t n_read_features;
    float prediction;
    char *path = "/home/rodrigo/Documents/AI_neuroral_network/datasets/kaggle/dataset_laura.csv";
    n_read_features = read_n_features(path, MAX_TEST_SAMPLES, features, &n_features);
    n_features--;
    
    find_max_min_features(features, max_features, min_features);

    generate_rando_NN(neurons, n_features, max_features, min_features);


    //while (1){

        for (int i = 0; i < POPULATION; i++){
            evaluate_model(neurons[i], features, n_read_features, 
                    &population_accuracy[i], 0, n_features);
        }

        reorganize_population(population_accuracy, neurons);
        show_logs(population_accuracy);

        //mutate_population(neurons, population_accuracy, );

    //}
    

    printf("¡Hola, Mundo!\n");
    return 0;

}
