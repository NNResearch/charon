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
#ifdef TACC
#include <mpi.h>
#else
#include <mpi/mpi.h>
#endif

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


/*
void devectorize_to_mats(StrategyInterpretation& si, const boost::numeric::ublas::vector<double>& query, Mat& domain_strat, Mat& split_strat) {
    int dis, dos, sis, sos, dim;
    si.fill_strategy_size(dis, dos, sis, sos, dim);
    domain_strat.resize(dos, dis);
    split_strat.resize(sos, sis);
    int ds_size = dos*dis, ss_size = sos*sis;
    for (int i = 0; i < ds_size; i++)
        domain_strat(i/dis, i%dis) = query[i];
    for (int i = 0; i < ss_size; i++)
        split_strat(i/sis, i%sis) = query[ds_size+i];
}
*/

static int numSample = 0;
/** Holds information needed by the Bayesian optimization procedure. */
class CegarOptimizer: public bayesopt::ContinuousModel {
    private:
        /** A set of properties to train with. */
        std::vector<Property> properties;
        /** The meta-strategy we are training. */
        const StrategyInterpretation& strategy_interp;
        /** The number of MPI processes we're using. */
        int world_size;

    public:
        /**
         * Construct an optimizer.
         *
         * \param input_dimension The number of parameters to train.
         * \param params A set of parameters to use for training.
         * \param si The meta-strategy.
         * \param ws The number of MPI processes.
         * \param prop_file A file containing several training properties.
         */
        CegarOptimizer(size_t input_dimension, bayesopt::Parameters params, const StrategyInterpretation& si, int ws, std::string prop_file):
            bayesopt::ContinuousModel(input_dimension, params), strategy_interp(si), world_size(ws) {
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
            int numProperties, propertiesToAssign, propertiesEvaluated = 0;
            // Split the strategy vector into two matrices, one for choosing a domain
            // and one for choosing a partition.
            int dis, dos, sis, sos;
            strategy_interp.fill_strategy_size(dis, dos, sis, sos);
            Mat domain_strat(dos, dis);
            Mat split_strat(sos, sis);
            for (int i = 0; i < (int)query.size(); i++) {
                if (i < dos * dis) { domain_strat(i / dis, i % dis) = query(i); } 
                else { split_strat((i - dos*dis) / sis, (i - dos*dis) % sis) = query(i); }
            }
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
            int i = 0;
            numProperties = properties.size();
            propertiesToAssign = numProperties;
            int N = (world_size-1 > numProperties) ? numProperties : world_size-1;
            for (i = 0; i < N; i++) {
                //Distribute properties to available workers
                this->sendProperty(this->properties[i].net, i, i+1, query);
                propertiesToAssign--;
            }
            while(propertiesEvaluated < numProperties) {
                int solved, source;
                MPI_Status status;
                MPI_Recv(&solved, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                double elapsed;
                MPI_Recv(&elapsed, 1, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (solved) {
                    // The property was either verified or falsified.
                    count++;
                    total_time += elapsed;
                } else {
                    // The verificatino timed out.
                    total_time += PENALTY * TIMEOUT;
                }
                propertiesEvaluated++;
                if (propertiesToAssign > 0) {
                    // If we still have properties to evaluate, send one to the thread
                    // which we just got a result from.
                    source = status.MPI_SOURCE;
                    this->sendProperty(this->properties[i].net, i, source, query);
                    i++, propertiesToAssign--;
                }
            }

            // std::cout << "**** job no #" << no << " evaluted strategies: \n";
            // std::cout << " |- domain_strategy: "<< mat_to_str(domain_strat) << "\n";
            // std::cout << " |- split_strategy: "<< mat_to_str(split_strat) << "\n";
            // std::cout << " |- domain_strategy: "<< domain_strat << "\n";
            // std::cout << " |- split_strategy: "<< split_strat << "\n";
            std::cout << " Result: \n";
            std::cout << "   |- property solved: " << count << " (/" << numProperties << ")\n";
            std::cout << "   |- time spent: " << total_time << "\n\n\n";
            return total_time;
        }

        bool checkReachability(const boost::numeric::ublas::vector<double>& query) {
            // If we need constraints besides a bounding box we can put them here.
            return true;
        }

    private:
        void sendProperty(const NetType netType, int propId, int worker, const boost::numeric::ublas::vector<double>& query) {
            std::cout << " Sending network " << netType << " and property #" << propId << " to worker: " << worker << std::endl;
            std::cout << "   |- interval: " << properties[propId].itv << "\n";
            MPI_Send(&netType, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            int propertySize = properties[propId].itv.lower.size();
            int querySize = query.size();
            int msgSize = querySize + 2 * propertySize;
            std::vector<double> strategyAndProperty(querySize + 2 * propertySize);
            for (int j = 0; j < querySize; j++) {
                strategyAndProperty[j] = query[j];
            }
            for (int j = 0; j < propertySize; j++) {
                strategyAndProperty[querySize + j] = properties[propId].itv.lower[j];
                strategyAndProperty[querySize + propertySize + j] = properties[propId].itv.upper[j];
            }
            // std::cout << "Sending: " << std::endl;
            MPI_Send(&strategyAndProperty[0], msgSize, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);
            // std::cout << "Sent!" << std::endl;
        }
};

static PyObject* pModule;
static PyObject* pAttackInit;
static PyObject* pFunc;
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

    int world_rank, world_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0) {
        std::cout << "world size: " << world_size << std::endl;
        // The main process takes care of the Bayesian optimization stuff
        std::cout << "STARTING ROOT" << std::endl;
        //int dim = bi.input_size() * bi.output_size();
        // int dim = bi.domain_input_size() * bi.domain_output_size() + bi.split_input_size() * bi.split_output_size();
        // std::cout << "dim: " << dim << std::endl;

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
        CegarOptimizer opt(dim, params, bi, world_size, benchmarks);
        opt.setBoundingBox(lower_bound, upper_bound);
        opt.optimize(best_point);

        int done = -1;
        for (int i = 1; i < world_size; i++) {
            MPI_Send(&done, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        std::cout << "*********** Best Point is: (" << best_point(0);
        for (int i = 0; i < dim; i++) { std::cout << "," << best_point(i); }
        std::cout << ")\n";

    } else {
        struct timespec start, end;
        while(true) {
            //Receive properties from master process and then attempt to verify them.
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            int numElements, solved, netType;
            MPI_Status status;

            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &numElements);
            MPI_Recv(&netType, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (netType < 0) { //Root process is done with optimization. We can exit
                std::cout << "Exiting" << std::endl;
                break;
            }
            Network net = networks[static_cast<NetType>(netType)];
            PyObject *pgdAttack = networkAttacks[static_cast<NetType>(netType)];

            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &numElements);
            std::vector<double> strategyAndProperty(numElements);
            MPI_Recv(&strategyAndProperty[0], numElements, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // Interpret the given strategy and property.
            Mat domain_strat(dos, dis);
            Mat split_strat(sos, sis);
            int ds_size = dos*dis, ss_size = sos*sis;
            for (int i = 0; i < ds_size; i++)
                domain_strat(i/dis, i%dis) = strategyAndProperty[i];
            for (int i = 0; i < ss_size; i++)
                split_strat(i/sis, i%sis) = strategyAndProperty[ds_size+i];

            //Deserialize property
            // int propertyStart = bi.domain_output_size() * bi.domain_input_size() + bi.split_output_size() * bi.split_input_size();
            int propertyStart = dos*dis+sos*sis;
            int propertySize = numElements-propertyStart;
            Vec lower(propertySize/2);
            Vec upper(propertySize/2);
            int lowerStart = propertyStart, upperStart = propertyStart + propertySize/2;
            for (int i = 0; i < propertySize/2; i++) {
                lower(i) = strategyAndProperty[lowerStart+i];
                upper(i) = strategyAndProperty[upperStart+i];
            }

            // Verify property
            Interval itv(lower, upper);
            // We start by determining a target class. In order for the interval to be robust, 
            // all points must have the same label, so we can evaluate any point in the interval to get a target class.
            int max_ind = net.predict(lower);

            try {
                Vec counterexample(lower.size());
                int num_calls = 0;
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
                verify_with_strategy(lower, itv, max_ind, net,
                        counterexample, num_calls, domain_strat,
                        split_strat, bi, TIMEOUT, pgdAttack, pFunc);
                solved = 1;
            } catch (timeout_exception e) {
                // This exception indicates a timeout
                solved = 0;
            }
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            MPI_Send(&solved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            double elapsed = (double)(end.tv_sec - start.tv_sec);
            MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    PyEval_RestoreThread(tstate);
    for (const auto e : AllNetTypes) {
        Py_DECREF(networkAttacks[e]);
    }
    Attack_Finalize();

    MPI_Finalize();

    return 0;
}
