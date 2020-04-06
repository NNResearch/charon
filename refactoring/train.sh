#!/bin/bash
cd build
cmake ..
make
# ./learn ../../example/training_properties.txt
time mpirun -n 2 ./learn ../../example/training_properties.txt
# time mpirun -n 4 ./learn ../../example/training_properties.txt
# time mpirun -n 8 ./learn ../example/training_properties.txt
# time mpirun -n 12 ./learn ../example/training_properties.txt
# time ./learn ../example/training_properties.txt
