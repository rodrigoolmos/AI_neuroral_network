#include <stdio.h>
#include "train_NN.h"


int main(){

    struct timeval init_predictions = {0};
    struct timeval end_predictions = {0};
    struct timeval init_train = {0};
    struct timeval end_train = {0};

    struct NN neurons[POPULATION];
    struct feature features[MAX_TEST_SAMPLES];
    float population_accuracy[POPULATION] = {0};
    float population_accuracy_test = 0;
    float max_features[N_FEATURE];
    float min_features[N_FEATURE];
    int n_features;
    uint32_t n_read_features;
    char *path = "/home/rodrigo/Documents/AI_neuroral_network/datasets/kaggle/Lung_Cancer_processed_dataset.csv";
    n_read_features = read_n_features(path, MAX_TEST_SAMPLES, features, &n_features);
    n_features--;// prediction not used
    
    find_max_min_features(features, max_features, min_features);

    shuffle(features, n_read_features);

    generate_rando_NN(neurons, n_features, max_features, min_features);

    //n_read_features = n_read_features/10;

    while (1){
        gettimeofday(&init_predictions, NULL);
        #pragma omp parallel for
        for (int i = 0; i < POPULATION; i++){
            evaluate_model(neurons[i], features, 8*n_read_features/10, N_LAYERS,
                    &population_accuracy[i], 0, n_features);
        }
        gettimeofday(&end_predictions, NULL);
        reorganize_population(population_accuracy, neurons);

        evaluate_model(neurons[0], &features[8*n_read_features/10], 2*n_read_features/10, N_LAYERS,
                &population_accuracy_test, 1, n_features);

        mutate_population(neurons, population_accuracy, 
                            max_features, min_features, n_features, 234578);

        crossover_population(neurons, n_features);

        gettimeofday(&end_train, NULL);
        show_logs(population_accuracy);
        printf("Execution trainig %fs\n", (end_predictions.tv_sec - init_predictions.tv_sec) + 
                                    (end_predictions.tv_usec - init_predictions.tv_usec) / 1000000.0);
        printf("Execution all %fs\n", (end_train.tv_sec - init_predictions.tv_sec) + 
                                    (end_train.tv_usec - init_predictions.tv_usec) / 1000000.0);
        printf("\n\n");
        printf("\n\n");

    }


    return 0;

}
