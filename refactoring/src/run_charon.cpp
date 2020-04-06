#include <zonotope.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <random>
#include <unistd.h>
#include <memory>
#include <set>
#include "network.hpp"
#include "strategy.hpp"

#define TIMEOUT 1000

using namespace std;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: ./run <property-file> <network-file> <strategy-file>"
            << " <counterexample-file>" << std::endl;
        std::abort();
    }
    std::string property_file = argv[1];
    std::string network_filename = argv[2];
    std::string strategy_filename = argv[3];
    std::string counterexample_filename = argv[4];
    Network net = read_network(network_filename);
    Interval property = Interval(property_file);
    auto si = std::unique_ptr<StrategyInterpretation>(new BayesianStrategy());
    int dos = si->domain_output_size();
    int dis = si->domain_input_size();
    int sos = si->split_output_size();
    int sis = si->split_input_size();
    Vec strategyMat(dis * dos + sis * sos);
    std::ifstream in(strategy_filename);
    Mat domain_strat(dos, dis);
    Mat split_strat(sos, sis);
    for (int i = 0; i < dos * dis; i++) {
        domain_strat(i / dis, i % dis) = strategyMat(i);
    }
    for (int i = 0; i < sos * sis; i++) {
        split_strat(i / sis, i % sis) = strategyMat(dos * dis + i);
    }

    Attack_Initialize();
    PyObject* pgdAttack = Attack_For_Net(net);

    Vec out = net.evaluate(property.lower);
    int max_ind = 0;
    double max = out(0); 
    for (int i = 1; i < out.size(); i++) {
        if (out(i) > max) {
            max_ind = i;
            max = out(i);
        }
    }

    bool verified = false;
    bool timeout = false;
    Vec counterexample(net.get_input_size());
    try {
        int num_calls = 0;
        verified = verify_with_strategy(
                property.lower, property, max_ind, net,
                counterexample, num_calls, domain_strat, split_strat, *si,
                TIMEOUT, pgdAttack, pFunc);
    } catch (const timeout_exception& e) {
        timeout = true;
    }

    if (verified) {
        std::cout << "Property verified" << std::endl;
    } else if (timeout) {
        std::cout << "Timed out after " << TIMEOUT << " seconds" << std::endl;
    } else {
        std::ofstream out(counterexample_filename);
        for (int i = 0; i < counterexample.size(); i++) {
            out << counterexample(i) << std::endl;
        }
        std::cout << "Property falsified, counterexample written to " <<
            counterexample_filename << std::endl;
    }

    Attack_Finalize();
    Py_Finalize();

    return 0;
}
