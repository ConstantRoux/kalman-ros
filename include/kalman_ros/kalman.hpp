/**
* Kalman filter implementation using Eigen. Based on the following
* introductory paper:
*
*     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <eigen3/Eigen/Dense>

#pragma once

class KalmanFilter {

public:

  /**
  * Create a Kalman filter with the specified matrices.
  *   A - System dynamics matrix
  *   C - Output matrix
  *   Q - Process noise covariance
  *   R - Measurement noise covariance
  *   P - Estimate error covariance
  */
  KalmanFilter(
      double dt,
      const Eigen::MatrixXd& A,
      const Eigen::MatrixXd& C,
      const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R,
      const Eigen::MatrixXd& P
  );

  /**
  * Create a blank estimator.
  */
  KalmanFilter();

  /**
  * Initialize the filter with initial states as zero.
  */
  void init();

  /**
  * Initialize the filter with a guess for initial states.
  */
  void init(double t0, const Eigen::VectorXd& x0);

  /**
  * Update the estimated state based on measured values. The
  * time step is assumed to remain constant.
  */
  void predict_and_update(const Eigen::VectorXd& y);

  /**
  * Update the estimated state based on measured values,
  * using the given time step and the dynamics matrix.
  */
  void predict_and_update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A);

  /**
  * Update the estimated state based on measured values,
  * using the given time step, the dynamics matrix and the measurement noise covariance.
  */
  void predict_and_update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A, const Eigen::MatrixXd R);

  /**
  * Update the estimated state based on a estimated possible measure,
  * using the given time step and the dynamics matrix.
  */
  void predict_and_estimate(double dt, const Eigen::MatrixXd A);

  /**
  * Return the current state, covariance and time.
  */
  Eigen::VectorXd state() 
  { 
    return x_hat; 
  };

  Eigen::MatrixXd covariance() 
  { 
    return P; 
  };

  double time() 
  { 
    return t; 
  };

  /**
   * Compute quadratic normalized innovation using the
   * measure at k+1, the state at k and the matrices
   * A and C at k+1.
  */
  Eigen::VectorXd compute_innovation(const Eigen::VectorXd& y, const Eigen::MatrixXd A, const Eigen::MatrixXd C);

private:

  // Matrices for computation
  Eigen::MatrixXd A, C, Q, R, P, K, P0;

  // System dimensions
  int m, n;

  // Initial and current time
  double t0, t;

  // Discrete time step
  double dt;

  // Is the filter initialized?
  bool initialized;

  // n-size identity
  Eigen::MatrixXd I;

  // Estimated states
  Eigen::VectorXd x_hat, x_hat_new;
};
