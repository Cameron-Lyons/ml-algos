#include "../matrix.h"
#include <cmath>
#include <vector>

class GaussianProcessRegressor {
private:
  double l;       // Length scale for RBF kernel
  double sigma_n; // Noise level
  Matrix X_train;
  Vector y_train;
  Matrix K_inv;

  double rbf_kernel(const Vector &x1, const Vector &x2) const {
    double sum = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
      sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return exp(-sum / (2 * l * l));
  }

public:
  GaussianProcessRegressor(double l, double sigma_n) : l(l), sigma_n(sigma_n) {}

  void fit(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;

    int n = X_train.size();
    Matrix K(n, Vector(n, 0.0));

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        K[i][j] = rbf_kernel(X_train[i], X_train[j]);
        if (i == j) {
          K[i][j] += sigma_n * sigma_n;
        }
      }
    }

    K_inv = invert_matrix(K);
  };

  double predict(const Vector &X_test) {
    int n = X_train.size();
    Matrix k_star(n, Vector(1));
    for (int i = 0; i < n; ++i) {
      k_star[i][0] = rbf_kernel(X_test, X_train[i]);
    }
    Matrix y_col(n, Vector(1));
    for (int i = 0; i < n; ++i)
      y_col[i][0] = y_train[i];
    Matrix alpha = multiply(K_inv, y_col); // (n x 1)
    double mu = 0.0;
    for (int i = 0; i < n; ++i) {
      mu += k_star[i][0] * alpha[i][0];
    }
    return mu;
  }
};
