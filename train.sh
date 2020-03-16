#!/bin/bash
cd build
# mpirun -n N ./learn ../example/training_properties.txt
make
# time mpirun -n 2 ./learn ../example/training_properties.txt
# time mpirun -n 8 ./learn ../example/training_properties.txt
time mpirun -n 12 ./learn ../example/training_properties.txt
# time ./learn ../example/training_properties.txt
