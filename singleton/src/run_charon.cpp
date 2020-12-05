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

static PyObject* pModule;
static PyObject* pAttackInit;
static PyObject* pFunc;
PyGILState_STATE gstate;
std::vector<PyObject*> liveObjects;
// static PyGILState_STATE gstate = PyGILState_Ensure();
void Cleanup_PyObjects() {
    for (PyObject* o: liveObjects)
        Py_DECREF(o);
    liveObjects.clear();
}

void Attack_Initialize() {
    std::string cwd = SINGLETON_HOME;
    Py_Initialize();
    PyEval_InitThreads();
    gstate = PyGILState_Ensure();
    char s[5] = "path";
    PyObject* sysPath = PySys_GetObject(s);
    PyObject* newElem = PyUnicode_FromString((cwd + "src").c_str());
    PyList_Append(sysPath, newElem);
    PySys_SetObject(s, sysPath);
    PyObject* pName = PyUnicode_FromString("interface");
    PyObject* pModule = PyImport_Import(pName);
    liveObjects.insert(liveObjects.end(), {sysPath, newElem, pName, pModule});
    // initialize_pgd_class
    pAttackInit = PyObject_GetAttrString(pModule, "initialize_pgd_class");
    if (!pAttackInit || !PyCallable_Check(pAttackInit)) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        Py_XDECREF(pAttackInit);
        Cleanup_PyObjects();
        throw std::runtime_error("Python error: Finding constructor");
    }
    liveObjects.push_back(pAttackInit);
    // prepare the pointer to function run_attack
    pFunc = PyObject_GetAttrString(pModule, "run_attack");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        Py_XDECREF(pFunc);
        Cleanup_PyObjects();
        throw std::runtime_error("Python error: loading attack function");
    }
    liveObjects.push_back(pFunc);
}

PyObject* Attack_Net(Network& net) {
    PyObject* pgdAttack;
    try {
        pgdAttack = create_attack_from_network(net, pAttackInit); // this return an object to IntervalPGDAttack
    } catch (const std::runtime_error& e) {
        Py_XDECREF(pgdAttack);
        Cleanup_PyObjects();
        throw e;
    }
    liveObjects.push_back(pgdAttack);
    return pgdAttack;
}

void Attack_Finalize() {
    gstate = PyGILState_Ensure();
    Cleanup_PyObjects();
    Py_Finalize();
}


int main(int argc, char** argv) {
#ifndef SINGLETON_HOME
    std::cout << "SINGLETON_HOME is undefined. If you compiled with the provided "
        << "CMake file this shouldn't happen, otherwise set SINGLETON_HOME" << std::endl;
    std::abort();
#endif
    std::string cwd = SINGLETON_HOME;
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
    Interval interval = Interval(property_file);
    auto si = std::unique_ptr<StrategyInterpretation>(new BayesianStrategy());
    Vec strategyVec(si->size());
    std::ifstream in(strategy_filename);
    for(int i=0; !in.eof(); i++) {
        in >> strategyVec(i);
    }
    std::cout << "strategy vector: " << vec_to_str(strategyVec) << "\n";

    Mat domain_strat, split_strat;
    si->vec_to_strategy_mats<Vec>(strategyVec, domain_strat, split_strat);
    std::cout << " Strategy: \n";
    std::cout << "   |- domain: "<< mat_to_str(domain_strat) << "\n";
    std::cout << "   |- split : "<< mat_to_str(split_strat) << "\n";

    Py_Initialize();
    PyEval_InitThreads();

    Attack_Initialize();
    PyObject* pgdAttack = Attack_Net(net);
    PyGILState_Release(gstate);
    PyThreadState* tstate = PyEval_SaveThread();

    Vec x = interval.lower;
    int y = net.predict(x);

    bool verified = false;
    bool timeout = false;
    Vec counterexample(net.get_input_size());
    try {
        int num_calls = 0;
        verified = verify_with_strategy(
                x, interval, y, net,
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

    PyEval_RestoreThread(tstate);
    Attack_Finalize();

    return 0;
}
