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

/* A tensor is represented as a `std::vector<Eigen::MatrixXd>`.
*/
typedef Eigen::VectorXd Vec;
typedef Eigen::MatrixXd Mat;
typedef std::vector<Mat> Tensor; 

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
