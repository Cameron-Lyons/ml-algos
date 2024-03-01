#include "../matrix.cpp"
#include <cmath>

class GaussianProcessRegressor {
private:
  double l;       // Length scale for RBF kernel
  double sigma_n; // Noise level
  Matrix X_train;
  Vector y_train;

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
          K[i][j] += sigma_n * sigma_n; // Adding noise variance to the diagonal
        }
      }
    }

    Matrix inverse_matrix = invert_matrix(K);
  };

  double predict(const Vector &X_test) {
    double K_star = 0;
    double sum = 0.0;

    for (size_t i = 0; i < X_train.size(); ++i) {
      K_star = rbf_kernel(X_test, X_train[i]);
      sum +=
          K_star * y_train[i]; // Assuming the inverse kernel matrix times the
                               // training outputs is identity for simplicity.
    }

    return sum;
  }
};
