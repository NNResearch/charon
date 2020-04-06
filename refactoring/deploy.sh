#!/bin/bash
cd build
make
time ./run ../../example/acas_robustness.bmk ../../example/acas_xu_1_1.txt ../../example/basic_strategy.txt ../../example/counterexample.txt
