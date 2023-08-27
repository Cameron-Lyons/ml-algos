#include "../matrix.h"
#include <cmath>

class LinearModel {
protected:
  Vector coefficients;

  // Add a column of 1s to the left of the matrix to account for the bias term
  Matrix addBias(const Matrix &X) {
    Matrix X_bias(X.size(), Vector(X[0].size() + 1, 1.0));
    for (size_t i = 0; i < X.size(); i++) {
      for (size_t j = 0; j < X[0].size(); j++) {
        X_bias[i][j + 1] = X[i][j];
      }
    }
    return X_bias;
  }

public:
  virtual void fit(const Matrix &X, const Vector &y) = 0;

  Vector predict(const Matrix &X) const {
    return multiply(X, {coefficients})[0];
  }

  Vector getCoefficients() const {
    Vector coeff(coefficients.begin() + 1, coefficients.end());
    return coeff;
  }

  // The bias term is the first coefficient
  double getBias() const { return coefficients[0]; }
};

class LinearRegression : public LinearModel {
public:
  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_bias = addBias(X);

    Matrix Xt = transpose(X_bias);
    Matrix XtX = multiply(Xt, X_bias);
    Matrix XtX_inv = inverse(XtX);
    Matrix XtX_inv_Xt = multiply(XtX_inv, Xt);
    Matrix betaMatrix = multiply(XtX_inv_Xt, {y});

    coefficients.clear();
    for (const auto &row : betaMatrix) {
      coefficients.push_back(row[0]);
    }
  }
};

class RidgeRegression : public LinearModel {
private:
  double lambda; // Regularization parameter

public:
  RidgeRegression(double lambda_val) : lambda(lambda_val) {}

  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_bias = addBias(X);
    Matrix Xt = transpose(X_bias);
    Matrix XtX = multiply(Xt, X_bias);

    // Add regularization term
    Matrix I(XtX.size(), Vector(XtX[0].size(), 0.0));
    for (size_t i = 0; i < I.size(); i++) {
      I[i][i] = lambda;
    }
    Matrix regularizedMatrix = add(XtX, I);

    Matrix regularizedMatrix_inv = inverse(regularizedMatrix);
    Matrix XtX_inv_Xt = multiply(regularizedMatrix_inv, Xt);
    Matrix betaMatrix = multiply(XtX_inv_Xt, {y});

    coefficients.clear();
    for (const auto &row : betaMatrix) {
      coefficients.push_back(row[0]);
    }
  }
};

class LassoRegression : public LinearModel {
private:
  double lambda; // Regularization parameter
  double tol;    // Tolerance for stopping criterion
  int max_iter;

  double soft_threshold(double value, double threshold) const {
    if (value > threshold)
      return value - threshold;
    if (value < -threshold)
      return value + threshold;
    return 0.0;
  }

public:
  LassoRegression(double lambda_val, double tol_val = 1e-6,
                  int max_iterations = 1000)
      : lambda(lambda_val), tol(tol_val), max_iter(max_iterations) {}

  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_bias = addBias(X);
    size_t num_samples = X_bias.size();
    size_t num_features = X_bias[0].size();

    coefficients = Vector(num_features, 0.0); // Initialize coefficients

    for (int iteration = 0; iteration < max_iter; ++iteration) {
      Vector old_coefficients = coefficients;

      for (size_t j = 0; j < num_features; ++j) {
        double tmp = y[j];
        for (size_t k = 0; k < num_features; ++k) {
          if (j != k)
            tmp -= coefficients[k] * X_bias[j][k];
        }

        if (j == 0) // Bias term, don't penalize
        {
          coefficients[j] = tmp;
        } else {
          coefficients[j] = soft_threshold(tmp, lambda) / num_samples;
        }
      }

      // Check for convergence
      double max_change = 0.0;
      for (size_t j = 0; j < num_features; ++j) {
        double change = fabs(old_coefficients[j] - coefficients[j]);
        if (change > max_change)
          max_change = change;
      }

      if (max_change < tol)
        break;
    }
  }
};

class ElasticNet : public LinearModel {
private:
  double alpha; // Regularization strength
  double rho;   // Mix ratio for L1 vs L2 regularization
  double tol;   // Tolerance for convergence
  int max_iter;

  double soft_threshold(double value, double threshold) const {
    if (value > threshold)
      return value - threshold;
    if (value < -threshold)
      return value + threshold;
    return 0.0;
  }

public:
  ElasticNet(double alpha_val, double rho_val, double tol_val = 1e-6,
             int max_iterations = 1000)
      : alpha(alpha_val), rho(rho_val), tol(tol_val), max_iter(max_iterations) {
  }

  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_bias = addBias(X);
    size_t num_samples = X_bias.size();
    size_t num_features = X_bias[0].size();

    coefficients = Vector(num_features, 0.0); // Initialize coefficients

    for (int iteration = 0; iteration < max_iter; ++iteration) {
      Vector old_coefficients = coefficients;

      for (size_t j = 0; j < num_features; ++j) {
        double tmp = y[j];
        for (size_t k = 0; k < num_features; ++k) {
          if (j != k)
            tmp -= coefficients[k] * X_bias[j][k];
        }

        if (j == 0) // Bias term, don't penalize
        {
          coefficients[j] = tmp;
        } else {
          coefficients[j] =
              soft_threshold(tmp, alpha * rho) / (1 + alpha * (1 - rho));
        }
      }

      // Check for convergence
      double max_change = 0.0;
      for (size_t j = 0; j < num_features; ++j) {
        double change = fabs(old_coefficients[j] - coefficients[j]);
        if (change > max_change)
          max_change = change;
      }

      if (max_change < tol)
        break;
    }
  }
};
