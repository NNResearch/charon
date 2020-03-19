/* Training module for Charon
*/

#include "strategy.hpp"
#include "powerset.hpp"
#include "network.hpp"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <map>
#include <Python.h>
#include <zonotope.h>
#include <bayesopt/bayesopt.hpp>

// #define TIMEOUT 1000
#define TIMEOUT 1000
#define PENALTY 2

/** An enum describing all the networks used for training. */
enum NetType {
    ACAS_XU,
};
static const NetType AllNetTypes[] = {ACAS_XU};


/** A map from network descriptors to network objects. */
static std::map<NetType, Network> networks = {
    {NetType::ACAS_XU, read_network(CHARON_HOME + std::string("example/acas_xu_1_1.txt"))}
};

/** A map from network descriptors to the associated PGD methods. */
static std::map<NetType, PyObject*> networkAttacks;

typedef struct property {
    Interval itv;
    NetType net;
} Property;


static PyObject* pModule;
static PyObject* pAttackInit;
static PyObject* pFunc;

static int numSample = 0;
/** Holds information needed by the Bayesian optimization procedure. */
class CegarOptimizer: public bayesopt::ContinuousModel {
    private:
        /** A set of properties to train with. */
        std::vector<Property> properties;
        /** The meta-strategy we are training. */
        const StrategyInterpretation& strategy_interp;

    public:
        /**
         * Construct an optimizer.
         *
         * \param input_dimension The number of parameters to train.
         * \param params A set of parameters to use for training.
         * \param si The meta-strategy.
         * \param prop_file A file containing several training properties.
         */
        CegarOptimizer(size_t input_dimension, bayesopt::Parameters params, const StrategyInterpretation& si, std::string prop_file):
            bayesopt::ContinuousModel(input_dimension, params), strategy_interp(si) {
                std::string line;
                std::ifstream fd(prop_file);
                // Load a set of properties. Each line in prop_file is a filename
                // for a file containing some training property. These filenames
                // should be relative to the Charon home directory.
                std::vector<std::string> prop_files;
                while(std::getline(fd, line))
                    prop_files.push_back(line);

                for (std::string& s : prop_files) {
                    std::vector<std::string> results;
                    std::stringstream iss(s);
                    for(std::string s; iss >> s;) {
                        results.push_back(s);
                    }
                    Property p;
                    p.itv = Interval(CHARON_HOME + std::string(results[0]));
                    p.net = NetType::ACAS_XU;
                    properties.push_back(p);
                }

            }

        /** Get the training properties of this class. */
        const std::vector<Property>& getProperties() const {
            return this->properties;
        }

        /**
         * Determine how good a given strategy is.
         *
         * \param query The strategy to evaluate.
         * \return A score for the strategy.
         */
        double evaluateSample(const boost::numeric::ublas::vector<double>& query) {
            int no = ++numSample;
            // We should only get into this call when world_rank = 0
            // There might be a more efficient way to convert to an Eigen vector
            // Split the strategy vector into two matrices, one for choosing a domain
            // and one for choosing a partition.
            
            Mat domain_strat, split_strat;
            strategy_interp.vec_to_strategy_mats< boost::numeric::ublas::vector<double> >(query, domain_strat, split_strat);
            // std::cout << "job no #" << no << " to evaluate\n";
            std::cout << "******************** job no #" << no << " evaluting strategies ********************* \n";
            std::cout << " Strategy: \n";
            std::cout << "   |- domain: "<< mat_to_str(domain_strat) << "\n";
            std::cout << "   |- split : "<< mat_to_str(split_strat) << "\n";
            // std::cout << "Evaluating: (domain)" << mat_to_str(domain_strat) << std::endl;
            // std::cout << "and: (split)" << mat_to_str(split_strat) << std::endl;
            // For some given time budget (per property), see how many properties we can verify
            int count = 0;
            double total_time = 0.0;
            for (size_t i = 0; i < properties.size(); i++) {
                double res = handle_one_property(domain_strat, split_strat, properties[i]);
                if (res>0) {
                    count++;
                    total_time+=res;
                } else {
                    // The verificatino timed out.
                    total_time += PENALTY * TIMEOUT;
                }
            }

            std::cout << " Result: \n";
            std::cout << "   |- property solved: " << count << " (/" << properties.size() << ")\n";
            std::cout << "   |- time spent: " << total_time << "\n\n\n";
            return total_time;
        }

        bool checkReachability(const boost::numeric::ublas::vector<double>& query) {
            // If we need constraints besides a bounding box we can put them here.
            return true;
        }

    private:
        double handle_one_property(Mat& domain_strat, Mat& split_strat, Property& property) 
        {
            struct timespec start, end;
            int netType = property.net;
            Interval interval = property.itv;

            Network net = networks[static_cast<NetType>(netType)];
            PyObject *pgdAttack = networkAttacks[static_cast<NetType>(netType)];

            // Verify property
            Vec x = interval.lower;
            int y = net.predict(x);
            Vec counterexample(x.size());
            int num_calls = 0;

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            try {
                verify_with_strategy(x, interval, y, net, counterexample, num_calls, domain_strat, split_strat, 
                        strategy_interp, TIMEOUT, pgdAttack, pFunc);
            } catch (timeout_exception e) {
                // This exception indicates a timeout
                return 0.0;
            }
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            double elapsed = (double)(end.tv_sec - start.tv_sec);
            return elapsed;
        }
};

PyGILState_STATE gstate;
std::vector<PyObject*> liveObjects;
// static PyGILState_STATE gstate = PyGILState_Ensure();
void cleanupPyObjects() {
    for (PyObject* o: liveObjects)
        Py_DECREF(o);
    liveObjects.clear();
}

void Attack_Initialize() {
    std::string cwd = CHARON_HOME;
    Py_Initialize();
    PyEval_InitThreads();
    gstate = PyGILState_Ensure();
    char s[5] = "path";
    PyObject* sysPath = PySys_GetObject(s);
    PyObject* newElem = PyString_FromString((cwd + "/src").c_str());
    PyList_Append(sysPath, newElem);
    PySys_SetObject(s, sysPath);
    PyObject* pName = PyString_FromString("interface");
    PyObject* pModule = PyImport_Import(pName);
    liveObjects.insert(liveObjects.end(), {sysPath, newElem, pName, pModule});
    // Py_DECREF(pName);
    // initialize_pgd_class
    pAttackInit = PyObject_GetAttrString(pModule, "initialize_pgd_class");
    if (!pAttackInit || !PyCallable_Check(pAttackInit)) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        Py_XDECREF(pAttackInit);
        // Py_DECREF(pModule);
        cleanupPyObjects();
        throw std::runtime_error("Python error: Finding constructor");
    }
    liveObjects.push_back(pAttackInit);
    // prepare the pointer to function run_attack
    pFunc = PyObject_GetAttrString(pModule, "run_attack");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        Py_XDECREF(pFunc);
        // Py_DECREF(pModule);
        cleanupPyObjects();
        throw std::runtime_error("Python error: loading attack function");
    }
    liveObjects.push_back(pFunc);
}

PyObject* Attack_Net(Network& net) {
    try {
        PyObject* pgdAttack = create_attack_from_network(net, pAttackInit); // this return an object to IntervalPGDAttack
        liveObjects.push_back(pgdAttack);
        return pgdAttack;
    } catch (const std::runtime_error& e) {
        Py_DECREF(pAttackInit);
        Py_DECREF(pModule);
        throw e;
    }
}

void Attack_Finalize() {
    gstate = PyGILState_Ensure();
    cleanupPyObjects();
    Py_Finalize();
}

int main(int argc, char** argv) {
    std::cout << "start the program..." << std::endl << std::flush;
    if (argc != 2) {
        std::cout << "Usage: ./learn <property-file>" << std::endl;
        std::abort();
    }
#ifndef CHARON_HOME
    std::cout << "CHARON_HOME is undefined. If you compiled with the provided "
        << "CMake file this shouldn't happen, otherwise set CHARON_HOME" << std::endl;
    std::abort();
#endif

    Attack_Initialize();
    for (const auto e : AllNetTypes) {
        networkAttacks[e] = Attack_Net(networks[e]);
    }

    PyGILState_Release(gstate);
    PyThreadState* tstate = PyEval_SaveThread();

    BayesianStrategy bi;
    int dis,dos,sis,sos,dim;
    bi.fill_strategy_size(dis, dos, sis, sos, dim);

    // The main process takes care of the Bayesian optimization stuff
    std::cout << "STARTING ROOT" << std::endl;

    boost::numeric::ublas::vector<double> best_point(dim);
    boost::numeric::ublas::vector<double> lower_bound(dim);
    boost::numeric::ublas::vector<double> upper_bound(dim);
    for (int i = 0; i < dim; i++) { lower_bound(i) = -1.0; upper_bound(i) = 1.0; }

    std::string benchmarks = argv[1];
    bayesopt::Parameters params;
    params.n_iterations = 400;
    params.l_type = L_MCMC;
    params.n_iter_relearn = 20;
    params.load_save_flag = 2;
    CegarOptimizer opt(dim, params, bi, benchmarks);
    opt.setBoundingBox(lower_bound, upper_bound);
    opt.optimize(best_point);

    std::cout << "*********** Best Point is: (" << best_point(0);
    for (int i = 0; i < dim; i++) { std::cout << "," << best_point(i); }
    std::cout << ")\n";

    PyEval_RestoreThread(tstate);
    for (const auto e : AllNetTypes) {
        Py_DECREF(networkAttacks[e]);
    }
    Attack_Finalize();
    return 0;
}


