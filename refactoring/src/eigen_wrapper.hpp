/* Greg Anderson
 * Define a neural network.
 */

#ifndef _EIGEN_WRAPPER_H_
#define _EIGEN_WRAPPER_H_

#include <cstdlib>
#include <string>
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Python.h>

/* A tensor is represented as a `std::vector<Eigen::MatrixXd>`.
*/
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
typedef std::vector<Mat> Tensor; 

static PyObject* eigen_vector_to_python_list(const Vec& b) {
    PyObject* ret = PyList_New(b.size());
    for (int i = 0; i < b.size(); i++) {
        PyObject* pi = PyFloat_FromDouble(b(i));
        PyList_SetItem(ret, i, pi);
    }
    return ret;
}

static PyObject* eigen_matrix_to_python_list(const Mat& w) {
    PyObject* ret = PyList_New(w.rows());
    for (int i = 0; i < w.rows(); i++) {
        PyObject* row = eigen_vector_to_python_list(w.row(i));
        PyList_SetItem(ret, i, row);
    }
    return ret;
}

static Vec python_list_to_eigen_vector(PyObject* pValue) {
    int size = PyList_Size(pValue);
    Vec ret(size);
    for (int i = 0; i < size; i++) {
        PyObject* elem = PyList_GetItem(pValue, i);
        ret(i) = PyFloat_AsDouble(elem);
    }
    return ret;
}
static std::string vec_to_str(Vec& v) {
    if (v.size()<=0) return "()";
    std::stringstream ss;
    ss << "(" << v(0);
    for (unsigned int i = 1; i < v.size(); i++) {
        ss << "," << v(i);
    }
    ss << ")";
    return ss.str();
}


#endif
