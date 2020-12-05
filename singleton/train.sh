#!/bin/bash
cd build
cmake ..
make
# time mpirun -n 2 ./learn ../example/training_properties.txt
time ./learn ../../example/training_properties.txt
# time ./learn ../example/training_properties.txt
