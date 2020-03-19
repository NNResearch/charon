/* Greg Anderson
* Classes for mapping a vector to a strategy.
*/

#include "strategy.hpp"
#include "powerset.hpp"
#include "network.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <utility>
#include <deque>
#include <ctime>
#include <future>
#include <memory>

#include <Python.h>

#define NUM_THREADS 16

using namespace std;

//Domain default_domain = Domain::IntervalDomain;
Domain default_domain = Domain::ZonotopeDomain;

StrategyInterpretation::~StrategyInterpretation() {
    // Do nothing
}

double getSplitOffset(const Interval &input_space, uint dimension, double strategy_param) {
    const Vec center = input_space.get_center();
    double eps = std::min(1.0 - EPSILON, std::max(-1.0 + EPSILON, strategy_param));
    double len = (input_space.upper(dimension) - input_space.lower(dimension)) / 2.0;
    return center(dimension) + len * eps;
}

Vec BayesianStrategy::domain_featurize( const Network& net, const Interval& input_space, const Vec& counterexample) const {
    Vec center = input_space.get_center();
    Vec diff = counterexample - center;
    Vec farthest = center - input_space.lower;
    // Get the distance from the center to the counterexample in a few different
    // norms and normalize these distances according to the distance from the
    // center to the corner.
    double l1 = diff.lpNorm<1>() / farthest.lpNorm<1>();
    double linf = diff.lpNorm<Eigen::Infinity>() / farthest.lpNorm<Eigen::Infinity>();
    // Get the difference between the two best scores at the point returned
    // by the counterexample search.
    Vec output = net.evaluate(counterexample);
    int best_ind = 0;
    double best_score = output(0);
    for (int i = 1; i < output.size(); i++) {
        if (output(i) > best_score) {
            best_ind = i;
            best_score = output(i);
        }
    }
    int second_best_ind = best_ind == 0 ? 1 : 0;
    double second_best_score = output(second_best_ind);
    for (int i = 0; i < output.size(); i++) {
        if (i == best_ind) {
            continue;
        }
        if (output(i) > second_best_score) {
            second_best_score = output(i);
        }
    }
    double score_diff = (best_score - second_best_score) / std::abs(best_score);
    // Get some information about the gradient at the counterexample point
    Vec gradient = net.gradient(counterexample);
    double tmp = std::abs(gradient(0));
    double gradSum = tmp;
    for (int i = 1; i < gradient.size(); i++) {
        gradSum += std::abs(gradient(i));
        if (std::abs(gradient(i)) > tmp) {
            tmp = std::abs(gradient(i));
        }
    }
    double ce_dim_largest_grad = tmp / gradSum;
    // Construct a strategy input from the gathered information
    Vec strategy_input(this->domain_input_size());
    strategy_input << l1, linf, score_diff, ce_dim_largest_grad, 1.0;
    return strategy_input;
}

Vec BayesianStrategy::split_featurize(
        const Network& net, const Interval& input_space,
        const Vec& counterexample) const {
    // Use the same featurization function for partitioning as we do for
    // choosing a domain.
    return domain_featurize(net, input_space, counterexample);
}

void BayesianStrategy::domain_extract(
        const Vec& strategy_output,
        const Network& net, const Vec& counterexample,
        Domain& domain, int& num_disjuncts) const {
    // Choose a domain
    if (strategy_output(0) >= 0) {
        domain = ZonotopeDomain;
    } else {
        domain = IntervalDomain;
    }
    // Clip the powerset size output and discretize it into one of
    // 64 options.
    double clipped = std::max(0.0, std::min(1.0, strategy_output(1)));
    num_disjuncts = 1 + (int) (clipped * 64);
}

void BayesianStrategy::split_extract(
        const Vec& strategy_output,
        const Network& net, const AbstractResult& ar,
        double& split_offset, uint& dimension) const {
    bool concretize;
    // Choose between the dimension of highest influence (dim1) and the longest
    // dimension (dim2)
    uint dim1 = back_prop(net, ar.interval, concretize, ar.maxInd, ar.layerOutputs);
    uint dim2 = ar.interval.longest_dim();
    if (strategy_output(1) <= 0) {
        dimension = dim1;
    } else {
        dimension = dim2;
    }
    if (concretize)
        split_offset = 0;
    else
        split_offset = getSplitOffset(ar.interval, dimension, strategy_output(0));
}

// Free a given hyperinterval
void free_interval(elina_interval_t** property, int dims) {
    for (int i = 0; i < dims; i++) {
        elina_interval_free(property[i]);
    }
    free(property);
}

PyObject* eigen_vector_to_python_list(const Vec& b) {
    PyObject* ret = PyList_New(b.size());
    for (int i = 0; i < b.size(); i++) {
        PyObject* pi = PyFloat_FromDouble(b(i));
        PyList_SetItem(ret, i, pi);
    }
    return ret;
}

PyObject* eigen_matrix_to_python_list(const Mat& w) {
    PyObject* ret = PyList_New(w.rows());
    for (int i = 0; i < w.rows(); i++) {
        PyObject* row = eigen_vector_to_python_list(w.row(i));
        PyList_SetItem(ret, i, row);
    }
    return ret;
}

Vec python_list_to_eigen_vector(PyObject* pValue) {
    int size = PyList_Size(pValue);
    Vec ret(size);
    for (int i = 0; i < size; i++) {
        PyObject* elem = PyList_GetItem(pValue, i);
        ret(i) = PyFloat_AsDouble(elem);
    }
    return ret;
}

// Split a given input region into two pieces to analyze
void split(const AbstractResult &ar,
        const Mat& split_strategy, const Network& net,
        AbstractInput& left, AbstractInput& right,
        const StrategyInterpretation& interp) {

    const Interval &itv = ar.interval;
    const Vec &counterexample = ar.counterexample;
    //const std::vector<Powerset> &layerOutputs = ar.layerOutputs;
    Vec strategy_input = interp.split_featurize(net, itv, counterexample);
    Vec strategy_output = split_strategy * strategy_input;

    double split_offset;
    uint dimension;
    interp.split_extract(strategy_output, net, ar, split_offset, dimension);

    // These four vectors describe the bounds of the two new regions.
    Vec left_lower(net.get_input_size());
    Vec left_upper(net.get_input_size());
    Vec right_lower(net.get_input_size());
    Vec right_upper(net.get_input_size());
    for (int i = 0; i < net.get_input_size(); i++) {
        if (i == (int)dimension) {
            // If i is the split dimension, then we change left_upper and right_lower
            // to partition the space.
            left_lower(i) = itv.lower(i);
            right_upper(i) = itv.upper(i);
            if (split_offset == 0) {
                left_lower(i) = itv.lower(i);
                right_lower(i) = itv.upper(i);
            } else {
                left_upper(i) = split_offset;
                right_lower(i) = split_offset;
            }
        } else {
            // All other bounds stay the same.
            left_lower(i) = itv.lower(i);
            left_upper(i) = itv.upper(i);
            right_lower(i) = itv.lower(i);
            right_upper(i) = itv.upper(i);
        }
    }
    left.property.set_bounds(left_lower, left_upper);
    right.property.set_bounds(right_lower, right_upper);
}

static int samples = 0;
// Search for a counterexample
Vec find_counterexample(Interval input, int max_ind, const Network& net, PyObject* pgdAttack, PyObject* pFunc) {
    // PGD from the center of this box
    Vec lower = input.lower;
    Vec upper = input.upper;
    Vec ce = input.get_center();
    
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject* pArgs = PyTuple_New(5);
    // the tuple will take our reference to the pgdAttack so in order to maintain
    // this for future calls we will need to incref.
    Py_INCREF(pgdAttack);
    PyTuple_SetItem(pArgs, 0, pgdAttack);
    PyTuple_SetItem(pArgs, 1, eigen_vector_to_python_list(ce));
    PyObject* pNum = PyInt_FromLong(max_ind);
    PyTuple_SetItem(pArgs, 2, pNum);
    PyTuple_SetItem(pArgs, 3, eigen_vector_to_python_list(lower));
    PyTuple_SetItem(pArgs, 4, eigen_vector_to_python_list(upper));

    // call run_attack(attack_class_object, x, y, lower, upper)
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    if (pValue == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        PyGILState_Release(gstate);
        throw std::runtime_error("Python error in find_counterexample");
    }

    ce = python_list_to_eigen_vector(pValue);
    // cout << "*********************************************************\n" << flush;
    // cout << ">>search counterexample around " << vec_to_str(ce) << ")\n" << flush;
    // cout << "  -- lower bound is " << vec_to_str(lower)<< "\n" << flush;
    // cout << "  -- upper bound is " << vec_to_str(upper)<< "\n" << flush;
    // cout << "  == counterexample found: " << vec_to_str(ce) << "\n" << flush;
    // cout << " #" << ++samples << ") " << input << " -->" << vec_to_str(ce) << "\n" << flush;
    Py_DECREF(pValue);
    PyGILState_Release(gstate);
    return ce;
}

PyObject* create_attack_from_network(const Network& net, PyObject* pAttackInit) {
    PyObject* ls = PyList_New(net.get_num_layers());
    for (int i = 0; i < net.get_num_layers(); i++) {
        std::shared_ptr<Layer> l = net.get_layer(i);
        LayerType lt = l->get_type();
        PyObject* in_size = PyTuple_New(3);
        PyTuple_SetItem(in_size, 0, PyInt_FromLong(l->get_input_height()));
        PyTuple_SetItem(in_size, 1, PyInt_FromLong(l->get_input_width()));
        PyTuple_SetItem(in_size, 2, PyInt_FromLong(l->get_input_depth()));
        PyObject* out_size = PyTuple_New(3);
        PyTuple_SetItem(out_size, 0, PyInt_FromLong(l->get_output_height()));
        PyTuple_SetItem(out_size, 1, PyInt_FromLong(l->get_output_width()));
        PyTuple_SetItem(out_size, 2, PyInt_FromLong(l->get_output_depth()));
        PyObject* pl = PyTuple_New(5);
        PyTuple_SetItem(pl, 1, in_size);
        PyTuple_SetItem(pl, 2, out_size);
        if (lt == CONV) {
            std::shared_ptr<ConvLayer> cl = std::static_pointer_cast<ConvLayer>(l);
            PyTuple_SetItem(pl, 0, PyString_FromString("conv"));

            PyObject* filter = PyList_New(cl->filter_height);
            for (int j = 0; j < cl->filter_height; j++) {
                PyObject* row = PyList_New(cl->filter_width);
                for (int k = 0; k < cl->filter_width; k++) {
                    PyObject* line = PyList_New(cl->filter_depth);
                    for (int m = 0; m < cl->filter_depth; m++) {
                        PyObject* filts = PyList_New(cl->num_filters);
                        for (int n = 0; n < cl->num_filters; n++) {
                            PyList_SetItem(filts, n, PyFloat_FromDouble(cl->filters[n].data[m](j, k)));
                        }
                        PyList_SetItem(line, m, filts);
                    }
                    PyList_SetItem(row, k, line);
                }
                PyList_SetItem(filter, j, row);
            }

            PyObject* biases = PyList_New(cl->num_filters);
            for (int j = 0; j < cl->num_filters; j++) {
                PyList_SetItem(biases, j, PyFloat_FromDouble(cl->biases[j]));
            }

            PyTuple_SetItem(pl, 3, filter);
            PyTuple_SetItem(pl, 4, biases);
        } else if (lt == FC) {
            std::shared_ptr<FCLayer> fcl = std::static_pointer_cast<FCLayer>(l);
            PyTuple_SetItem(pl, 0, PyString_FromString("fc"));
            PyTuple_SetItem(pl, 3, eigen_matrix_to_python_list(fcl->weight));
            PyTuple_SetItem(pl, 4, eigen_vector_to_python_list(fcl->bias));
        } else {
            std::shared_ptr<MaxPoolLayer> mpl = std::static_pointer_cast<MaxPoolLayer>(l);
            PyTuple_SetItem(pl, 0, PyString_FromString("maxpool"));
            PyTuple_SetItem(pl, 3, PyInt_FromLong(mpl->window_height));
            PyTuple_SetItem(pl, 4, PyInt_FromLong(mpl->window_width));
        }
        PyList_SetItem(ls, i, pl);
    }
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, ls);
    PyObject* pgdAttack = PyObject_CallObject(pAttackInit, pArgs);
    Py_DECREF(pArgs);
    if (pgdAttack == NULL) {
        PyErr_Print();
        throw std::runtime_error("Python error: Initializing attack");
    }
    return pgdAttack;
}

// See if an output value satisfies the robustness property.
bool check_abstract_value(Powerset output, int max_ind, int dims) {
    // Construct an array of linear constraints encoding the part of the output
    // space where o(max_ind) > o(k) for all other k.
    for (int i = 0; i < dims; i++) {
        elina_lincons0_array_t cons = elina_lincons0_array_make(1);
        if (i == max_ind) {
            continue;
        }
        elina_linexpr0_t* le = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE, 2);
        // Produce a lincons0 as x_i >= x_{max_ind}, i.e., x_i - x_{max_ind} >= 0
        elina_linexpr0_set_coeff_scalar_double(le, max_ind, -1.0);
        elina_linexpr0_set_coeff_scalar_double(le, i, 1.0);
        elina_linexpr0_set_cst_scalar_double(le, 0.0);
        cons.p[0].constyp = ELINA_CONS_SUPEQ;
        cons.p[0].linexpr0 = le;
        Powerset p = output.meet_lincons_array(&cons);
        elina_lincons0_array_clear(&cons);
        if (!p.is_bottom()) {
            // If p is not bottom, then it includes some point where
            // x_i >= x_{max_ind}
            return false;
        }
    }
    return true;
}

// Propagate a given input interval through a network.
std::vector<Powerset> propagate_through_network(Interval input, int disjuncts, const Network& net, elina_manager_t* man) {
    elina_interval_t** itv = (elina_interval_t**) malloc(input.lower.size() * sizeof(elina_interval_t*));
    for (int i = 0; i < input.lower.size(); i++) {
        itv[i] = elina_interval_alloc();
        elina_interval_set_double(itv[i], input.lower(i), input.upper(i));
    }
    elina_abstract0_t* t = elina_abstract0_of_box(man, 0, net.get_input_size(), itv);
    free_interval(itv, net.get_input_size());

    Powerset z(man, t, disjuncts);
    return net.propagate_powerset(z);
}

static int input_no=0;
// Run one iteration of the Charon loop.
AbstractResult verify_abstract(AbstractInput ai, int max_ind,
        const Network& net, PyObject* pgdAttack, PyObject* pFunc,
        const StrategyInterpretation* interp, const Mat strategy) {
    int no=input_no++;
    // std::cout << "  #" << no << ") verify_abstract input: " << ai.property << " ->" << max_ind << "\n" << std::flush;
    Vec ce = find_counterexample(ai.property, max_ind, net, pgdAttack, pFunc);

    // We need to determine if ce is actually a counterexample
    Vec outp = net.evaluate(ce);
    int mi = max_ind;
    double max = outp(max_ind);
    double softmax_total = 0.0;
    for (int i = 0; i < net.get_output_size(); i++) {
        if (i == max_ind) {
            softmax_total += std::exp(outp(i));
            continue;
        }
        if (outp(i) >= max) {
            max = outp(i);
            mi = i;
        }
        softmax_total += std::exp(outp(i));
    }
    Vec softmax(outp.size());
    for (int i = 0; i < net.get_output_size(); i++) {
        softmax(i) = std::exp(outp(i)) / softmax_total;
    }

    Vec center(net.get_input_size());
    Vec lower_corner = ai.property.lower;
    Vec upper_corner = ai.property.upper;
    for (int i = 0; i < net.get_input_size(); i++) {
        center(i) = (lower_corner(i) + upper_corner(i)) / 2.0;
    }
    AbstractResult ar;
    ar.counterexample = ce;
    ar.interval = ai.property;
    ar.maxInd = max_ind;
    if (mi != max_ind) {
        ar.falsified = true;
        ar.verified = false;
        std::cout << "x" << std::flush;
        // std::cout << " " << no << "|-> falsified!\n";
        // std::cout << std::flush << "#" << no << ") verify " << ai.property << ":" << max_ind << " |-> falsified!\n" << std::flush;
        return ar;
    }

    // We did not find a counterexample, so now we need to choose a domain
    Vec domain_inp = interp->domain_featurize(net, ai.property, ce);
    Vec domain_outp = strategy * domain_inp;
    interp->domain_extract(domain_outp, net, ce, ai.domain, ai.disjuncts);

    elina_manager_t *man;

    if (ai.domain == Domain::IntervalDomain)
        man = elina_box_manager_alloc();
    else
        man = zonotope_manager_alloc();

    std::vector<Powerset> output = propagate_through_network(ai.property, ai.disjuncts, net, man);

    // Check if the output is verified by abstract interpretation.
    if (check_abstract_value(output[output.size()-1], max_ind, net.get_output_size())) {
        // If it is, we can just free this interval and move on to the next one
        AbstractResult ar;
        ar.verified = true;
        ar.falsified = false;
        ar.interval = ai.property;
        ar.layerOutputs = output;
        ar.maxInd = max_ind;
        elina_manager_free(man);
        std::cout << "o" << std::flush;
        // std::cout << std::flush << "#" << no << ") verify " << ai.property << ":" << max_ind << " |-> verified!\n" << std::flush;
        return ar;
    }

    // Otherwise we did not find a proof or a counterexample.
    ar.falsified = false;
    ar.verified = false;
    ar.maxInd = max_ind;
    ar.layerOutputs = output;
    elina_manager_free(man);
    std::cout << "?" << std::flush;
    // std::cout << " " << no << "|-> unknown!\n";
    // std::cout << std::flush << "#" << no << ") verify " << ai.property << ":" << max_ind << " |-> unknown!\n" << std::flush;
    return ar;
}

// Given an interval, determine whether the network is robust on that interval.
// If not, give a counterexample.
bool verify_with_strategy(const Vec& original,
        const Interval& property, int max_ind, const Network& net,
        Vec& counterexample, int& num_calls,
        const Mat& domain_strategy, const Mat& split_strategy,
        const StrategyInterpretation& interp,
        double timeout, PyObject* pgdAttack, PyObject* pFunc) {
    std::cout << "--> verify with strategy... \n";
    struct timespec start, current;
    AbstractInput init;
    init.property = property;
    init.domain = default_domain;
    init.disjuncts = POWERSET_SIZE;
    // to_verify will hold a bunch of intervals on which the property has not yet been verified
    std::deque<AbstractInput> to_verify;
    // Initially, to_verify is a single interval with the the entire input space
    to_verify.push_back(init);
    num_calls = 0;
    // We use std::future for parallelism. It's possible to extend this with
    // some more heavy duty framework like MPI.
    std::vector<std::future<AbstractResult>> results;
    bool verified = false;
    bool falsified = false;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    try {
        while (true) {
            // Start up to NUM_THREADS processes to use to verify properties
            while (!to_verify.empty() && results.size() < NUM_THREADS) {
                AbstractInput input = to_verify.front();
                to_verify.pop_front();
                num_calls++;
                // verify_abstract(input, max_ind, net, pgdAttack, pFunc, &interp, domain_strategy) 
                results.push_back(std::async(std::launch::async, verify_abstract, input,
                            max_ind, net, pgdAttack, pFunc, &interp, domain_strategy));
                /*
                 * std::string msg = "?"; 
                 * index=results.size()-1; 
                 * if (results[index].get().verified) msg="o"; 
                 * else if (results[index].get().falsified) msg="x"; 
                 * std::cout << msg << std::flush;
                 */
            }
            // If there are no more regions to verify then the property has been
            // proven.
            if (results.empty()) {
                verified = true;
                std::cout << "<-- [verified] verify with strategy... \n";
                return verified;
            }
            // See if any of our computations have finished
            for (unsigned int i = 0; i < results.size(); i++) {
                std::future_status st = results[i].wait_for(std::chrono::seconds(5));
                if (st == std::future_status::ready) {
                    AbstractResult ar = results[i].get();
                    results.erase(results.begin() + i);
                    i--;
                    if (ar.verified) {
                        // nothing to do
                    } else if (ar.falsified) {
                        falsified = true;
                        counterexample = ar.counterexample;
                        std::cout << "<-- [falsified] verify with strategy... \n";
                        return verified;
                    } else {
                        AbstractInput left;
                        AbstractInput right;
                        split(ar, split_strategy, net, left, right, interp);
                        to_verify.push_back(left);
                        to_verify.push_back(right);
                    }
                }
            }
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current);
            double elapsed = (double)(current.tv_sec - start.tv_sec);
            if (elapsed > timeout) {
                std::cout << "<-- [timeout] verify with strategy... \n";
                throw timeout_exception();
            }
        }
    } catch (const timeout_exception& e) {
        std::cout << "<-- [timeout] verify with strategy... \n";
        if (verified || falsified) {
            return verified;
        }
        throw e;
    }

    std::cout << "<-- [verified] verify with strategy... \n";
    return verified;
}

// Populate Eigen vector bounds from an ELINA interval
void intervalBounds(Vec &lower, Vec &upper,
        elina_interval_t** &box) {
    int size = upper.rows();
    for (int i = 0; i < size; i++) {
        elina_interval_t *itv = box[i];
        lower(i) = itv->inf->val.dbl;
        upper(i) = itv->sup->val.dbl;
    }
}

// Find the most influential dimension. This is based on work in ReluVal, see
// github.com/tcwangshiqi-columbia/ReluVal
uint back_prop(const Network &net, const Interval &input, bool &concretize,
        const int trueClass, const std::vector<Powerset> &layerOutputs) {
    concretize = false;
    const Tensor weights = net.get_weights();
    std::vector<uint> positiveWidthIntervals;

    for (uint i = 0; i < input.lower.size(); i++) {
        if (input.upper(i) > input.lower(i))
            positiveWidthIntervals.push_back(i);
    }
    int numLayers = net.get_num_layers();
    Mat dLower, dUpper;
    for(int i = numLayers-2; i > -1; i--) {
        Vec tempLower1 = Vec::Zero(net.get_layer_sizes()[i]);
        Vec tempUpper1 = Vec::Zero(net.get_layer_sizes()[i]);
        Mat dLower1, dUpper1;
        elina_interval_t **bbox = layerOutputs[i+1].bounding_box();
        Vec lower(layerOutputs[i+1].dims()), upper(layerOutputs[i+1].dims());
        intervalBounds(lower, upper, bbox);
        if (i == numLayers-2) {
            dUpper1 = weights[weights.size()-1].row(trueClass);
            dLower1 = weights[weights.size()-1].row(trueClass);
        } else {
            dUpper1 = dUpper;
            dLower1 = dLower;
        }
        Mat w = weights[i];
        dLower = tempLower1;
        dUpper = tempUpper1;
        if (weights[i].cols() == 0) {
            elina_interval_t** bbox2 = layerOutputs[i].bounding_box();
            Vec lower2(layerOutputs[i].dims()), upper2(layerOutputs[i].dims());
            intervalBounds(lower2, upper2, bbox2);
            // This is a max pooling layer
            for (int j = 0; j < net.get_layer_sizes()[i+1]; j++) {
                // Get the 3D coordinates of j
                int plane = j % net.get_layer_depth(i+1);
                int col = (j / net.get_layer_depth(i+1)) % net.get_layer_width(i+1);
                int row = (j / net.get_layer_depth(i+1)) / net.get_layer_width(i+1);
                col *= 2;
                row *= 2;
                int iw = net.get_layer_width(i);
                // int ih = net.get_layer_height(i);
                int id = net.get_layer_depth(i);
                // row and col are now the first coordinates in the input space
                int ind00 = iw * id * row + id * col + plane;
                int ind01 = iw * id * row + id * (col + 1) + plane;
                int ind10 = iw * id * (row + 1) + id * col + plane;
                int ind11 = iw * id * (row + 1) + id * (col + 1) + plane;
                bool couldBeMax00 = upper2(ind00) > lower2(ind01) &&
                    upper2(ind00) > lower2(ind10) && upper2(ind00) > lower2(ind11);
                bool couldBeMax01 = upper2(ind01) > lower2(ind00) &&
                    upper2(ind01) > lower2(ind10) && upper2(ind01) > lower2(ind11);
                bool couldBeMax10 = upper2(ind10) > lower2(ind00) &&
                    upper2(ind10) > lower2(ind01) && upper2(ind10) > lower2(ind11);
                bool couldBeMax11 = upper2(ind11) > lower2(ind00) &&
                    upper2(ind11) > lower2(ind01) && upper2(ind11) > lower2(ind10);
                bool twoCouldBe = (couldBeMax00 && couldBeMax01) ||
                    (couldBeMax00 && couldBeMax10) || (couldBeMax00 && couldBeMax11) ||
                    (couldBeMax01 && couldBeMax10) || (couldBeMax01 && couldBeMax11) ||
                    (couldBeMax10 && couldBeMax11);
                if (couldBeMax00) {
                    if (twoCouldBe) {
                        dLower(ind00) = dLower1(j) < 0 ? dLower1(j) : 0;
                        dUpper(ind00) = dUpper1(j) < 0 ? 0 : dUpper1(j);
                    }
                } else {
                    dLower(ind00) = dLower1(j);
                    dUpper(ind00) = dUpper1(j);
                }
                if (couldBeMax01) {
                    if (twoCouldBe) {
                        dLower(ind01) = dLower1(j) < 0 ? dLower1(j) : 0;
                        dUpper(ind01) = dUpper1(j) < 0 ? 0 : dUpper1(j);
                    }
                } else {
                    dLower(ind01) = dLower1(j);
                    dUpper(ind01) = dUpper1(j);
                }
                if (couldBeMax10) {
                    if (twoCouldBe) {
                        dLower(ind10) = dLower1(j) < 0 ? dLower1(j) : 0;
                        dUpper(ind10) = dUpper1(j) < 0 ? 0 : dUpper1(j);
                    }
                } else {
                    dLower(ind10) = dLower1(j);
                    dUpper(ind10) = dUpper1(j);
                }
                if (couldBeMax11) {
                    if (twoCouldBe) {
                        dLower(ind11) = dLower1(j) < 0 ? dLower1(j) : 0;
                        dUpper(ind11) = dUpper1(j) < 0 ? 0 : dUpper1(j);
                    }
                } else {
                    dLower(ind11) = dLower1(j);
                    dUpper(ind11) = dUpper1(j);
                }
            }
            free_interval(bbox2, layerOutputs[i].dims());
            continue;
        }
        for (int j = 0; j < net.get_layer_sizes()[i+1]; j++) {
            if (upper(j) <= 0) {
                dUpper1(j) = 0, dLower1(j) = 0;
            } else if (upper(j) > 0 && lower(j) < 0) {
                dUpper1(j) = dUpper1(j) > 0 ? dUpper1(j) : 0;
                dLower1(j) = dLower1(j) < 0 ? dLower1(j) : 0;
            }

            for (int k = 0; k < net.get_layer_sizes()[i]; k++) {
                if (w(j, k) >= 0) {
                    dUpper(k) += w(j, k) * dUpper1(j);
                    dLower(k) += w(j, k) * dLower1(j);
                } else {
                    dUpper(k) += w(j, k) * dLower1(j);
                    dLower(k) += w(j, k) * dUpper1(j);
                }

            }
        }
        free_interval(bbox, layerOutputs[i+1].dims());
    }

    uint dim = 0;
    double max_val = -1.0;

    for (uint i = 0; i < positiveWidthIntervals.size(); i++) {
        uint itv = positiveWidthIntervals[i];
        double up = dUpper(itv), low = -dLower(itv);
        double valBefore = up > low ? up : low;
        valBefore *= (input.upper(itv) - input.lower(itv));
        if (valBefore > max_val) {
            if (dUpper(itv) >= 0 && dLower(itv) >= 0) //monotonic gradient
                concretize = true;
            else if (dUpper(itv) <= 0 && dLower(itv) <= 0) //monotonic gradient
                concretize = true;
            else
                concretize = false;
            max_val = valBefore;
            dim = itv;
        }
    }
    return dim;
}
