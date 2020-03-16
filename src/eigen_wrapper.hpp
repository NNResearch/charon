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


#endif
