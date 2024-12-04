cmake_minimum_required(VERSION 3.10)

# Nombre del proyecto
project(Neuronal_network)

# Encontrar OpenCL
find_package(OpenMP REQUIRED)

# Agregar el ejecutable
add_executable(Neuronal_network ./vitis_hls/main.c ./vitis_hls/NN.c ./vitis_hls/train_NN.c ./vitis_hls/common.c)
target_link_libraries(Neuronal_network PRIVATE m OpenMP::OpenMP_C)
