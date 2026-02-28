#include "../matrix.h"
#include <algorithm>
#include <cmath>

class LinearModel {
protected:
  Vector coefficients;

  Matrix addBias(const Matrix &X) const {
    Matrix X_bias(X.size(), Vector(X[0].size() + 1, 1.0));
    for (size_t i = 0; i < X.size(); i++) {
      for (size_t j = 0; j < X[0].size(); j++) {
        X_bias[i][j + 1] = X[i][j];
      }
    }
    return X_bias;
  }

public:
  virtual ~LinearModel() = default;
  virtual void fit(const Matrix &X, const Vector &y) = 0;

  Vector predict(const Matrix &X) const {
    if (X.empty() || coefficients.empty()) {
      return {};
    }

    Vector preds(X.size(), coefficients[0]);
    for (size_t i = 0; i < X.size(); ++i) {
      double sum = coefficients[0];
      const size_t n_features =
          std::min(X[i].size(), coefficients.size() - static_cast<size_t>(1));
      for (size_t j = 0; j < n_features; ++j) {
        sum += coefficients[j + 1] * X[i][j];
      }
      preds[i] = sum;
    }
    return preds;
  }

  Vector getCoefficients() const {
    Vector coeff(coefficients.begin() + 1, coefficients.end());
    return coeff;
  }

  double getBias() const { return coefficients[0]; }

  void setCoefficients(const Vector &coefs) { coefficients = coefs; }

  const Vector &getRawCoefficients() const { return coefficients; }
};

class LinearRegression : public LinearModel {
public:
  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_bias = addBias(X);

    Matrix Xt = transpose(X_bias);
    Matrix XtX = multiply(Xt, X_bias);
    Matrix XtX_inv = invert_matrix(XtX);
    Matrix XtX_inv_Xt = multiply(XtX_inv, Xt);
    Matrix y_col(y.size(), Vector(1));
    for (size_t i = 0; i < y.size(); ++i) {
      y_col[i][0] = y[i];
    }
    Matrix betaMatrix = multiply(XtX_inv_Xt, y_col);

    coefficients.clear();
    for (const auto &row : betaMatrix) {
      coefficients.push_back(row[0]);
    }
  }
};

class RidgeRegression : public LinearModel {
private:
  double lambda;

public:
  RidgeRegression(double lambda_val) : lambda(lambda_val) {}

  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_bias = addBias(X);
    Matrix Xt = transpose(X_bias);
    Matrix XtX = multiply(Xt, X_bias);

    Matrix I(XtX.size(), Vector(XtX[0].size(), 0.0));
    for (size_t i = 0; i < I.size(); i++) {
      I[i][i] = lambda;
    }
    Matrix regularizedMatrix = add(XtX, I);

    Matrix regularizedMatrix_inv = invert_matrix(regularizedMatrix);
    Matrix XtX_inv_Xt = multiply(regularizedMatrix_inv, Xt);
    Matrix y_col(y.size(), Vector(1));
    for (size_t i = 0; i < y.size(); ++i) {
      y_col[i][0] = y[i];
    }
    Matrix betaMatrix = multiply(XtX_inv_Xt, y_col);

    coefficients.clear();
    for (const auto &row : betaMatrix) {
      coefficients.push_back(row[0]);
    }
  }
};

class LassoRegression : public LinearModel {
private:
  double lambda;
  double tol;
  int max_iter;

  double soft_threshold(double value, double threshold) const {
    if (value > threshold) {
      return value - threshold;
    }
    if (value < -threshold) {
      return value + threshold;
    }
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

    if (num_samples == 0 || y.empty()) {
      coefficients.clear();
      return;
    }

    coefficients = Vector(num_features, 0.0);
    Vector predictions(num_samples, 0.0);

    for (int iteration = 0; iteration < max_iter; ++iteration) {
      Vector old_coefficients = coefficients;

      for (size_t j = 0; j < num_features; ++j) {
        double old_beta_j = coefficients[j];
        double numerator = 0.0;
        double denominator = 0.0;

        for (size_t i = 0; i < num_samples; ++i) {
          const double x_ij = X_bias[i][j];
          const double residual = y[i] - predictions[i] + (x_ij * old_beta_j);
          numerator += x_ij * residual;
          denominator += x_ij * x_ij;
        }

        if (denominator <= 0.0) {
          continue;
        }

        if (j == 0) {
          coefficients[j] = numerator / denominator;
        } else {
          coefficients[j] = soft_threshold(numerator, lambda) / denominator;
        }

        const double delta = coefficients[j] - old_beta_j;
        if (delta != 0.0) {
          for (size_t i = 0; i < num_samples; ++i) {
            predictions[i] += X_bias[i][j] * delta;
          }
        }
      }

      double max_change = 0.0;
      for (size_t j = 0; j < num_features; ++j) {
        double change = fabs(old_coefficients[j] - coefficients[j]);
        if (change > max_change) {
          max_change = change;
        }
      }

      if (max_change < tol) {
        break;
      }
    }
  }
};

class ElasticNet : public LinearModel {
private:
  double alpha;
  double rho;
  double tol;
  int max_iter;

  double soft_threshold(double value, double threshold) const {
    if (value > threshold) {
      return value - threshold;
    }
    if (value < -threshold) {
      return value + threshold;
    }
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

    if (num_samples == 0 || y.empty()) {
      coefficients.clear();
      return;
    }

    coefficients = Vector(num_features, 0.0);
    Vector predictions(num_samples, 0.0);

    for (int iteration = 0; iteration < max_iter; ++iteration) {
      Vector old_coefficients = coefficients;

      for (size_t j = 0; j < num_features; ++j) {
        double old_beta_j = coefficients[j];
        double numerator = 0.0;
        double denominator = 0.0;

        for (size_t i = 0; i < num_samples; ++i) {
          const double x_ij = X_bias[i][j];
          const double residual = y[i] - predictions[i] + (x_ij * old_beta_j);
          numerator += x_ij * residual;
          denominator += x_ij * x_ij;
        }

        if (denominator <= 0.0) {
          continue;
        }

        if (j == 0) {
          coefficients[j] = numerator / denominator;
        } else {
          const double l1 = alpha * rho;
          const double l2 = alpha * (1.0 - rho);
          coefficients[j] = soft_threshold(numerator, l1) / (denominator + l2);
        }

        const double delta = coefficients[j] - old_beta_j;
        if (delta != 0.0) {
          for (size_t i = 0; i < num_samples; ++i) {
            predictions[i] += X_bias[i][j] * delta;
          }
        }
      }

      double max_change = 0.0;
      for (size_t j = 0; j < num_features; ++j) {
        double change = fabs(old_coefficients[j] - coefficients[j]);
        if (change > max_change) {
          max_change = change;
        }
      }

      if (max_change < tol) {
        break;
      }
    }
  }
};
