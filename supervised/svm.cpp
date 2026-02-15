#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>

class SVM {
protected:
  Vector weights;
  double learningRate;
  int maxIterations;

  SVM(size_t n_features, double learningRate, int maxIterations)
      : learningRate(learningRate), maxIterations(maxIterations) {
    weights.resize(n_features, 0.0);
  }

public:
  virtual double predict(const Vector &x) const = 0;
  virtual void fit(const Matrix &X, const Vector &y) = 0;
  virtual ~SVM() = default;
};

class SVC : public SVM {
private:
  int num_classes;
  std::vector<Vector> class_weights;

public:
  SVC(size_t n_features, int n_classes, double learningRate = 0.01,
      int maxIterations = 1000)
      : SVM(n_features, learningRate, maxIterations), num_classes(n_classes) {
    class_weights.resize(static_cast<size_t>(n_classes),
                         Vector(n_features, 0.0));
  }

  double predict(const Vector &x) const override {
    double max_dotProduct = std::numeric_limits<double>::lowest();
    int predicted_class = -1;

    for (int k = 0; k < num_classes; k++) {
      double dotProduct = 0.0;
      for (size_t i = 0; i < x.size(); i++) {
        dotProduct += x[i] * class_weights[static_cast<size_t>(k)][i];
      }
      if (dotProduct > max_dotProduct) {
        max_dotProduct = dotProduct;
        predicted_class = k;
      }
    }

    return static_cast<double>(predicted_class);
  }

  void fit(const Matrix &X, const Vector &y) override {
    for (int k = 0; k < num_classes; k++) {
      for (int iter = 0; iter < maxIterations; iter++) {
        bool allClassifiedCorrectly = true;
        for (size_t i = 0; i < X.size(); i++) {
          int true_label = (y[i] == k) ? 1 : -1;
          int prediction = (predict(X[i]) == k) ? 1 : -1;
          if (prediction != true_label) {
            allClassifiedCorrectly = false;
            for (size_t j = 0; j < X[i].size(); j++) {
              class_weights[static_cast<size_t>(k)][j] +=
                  learningRate * true_label * X[i][j];
            }
          }
        }
        if (allClassifiedCorrectly) {
          break;
        }
      }
    }
  }
};

class SVR : public SVM {
private:
  double bias = 0.0;
  double epsilon;

public:
  SVR(size_t n_features, double learningRate = 0.01, double epsilon = 0.1,
      int maxIterations = 1000)
      : SVM(n_features, learningRate, maxIterations), epsilon(epsilon) {}

  double predict(const Vector &x) const override {
    double result = bias;
    for (size_t i = 0; i < x.size(); i++) {
      result += x[i] * weights[i];
    }
    return result;
  }

  void fit(const Matrix &X, const Vector &y) override {
    for (int iter = 0; iter < maxIterations; iter++) {
      for (size_t i = 0; i < X.size(); i++) {
        double prediction = predict(X[i]);
        double error = prediction - y[i];
        if (std::abs(error) > epsilon) {
          for (size_t j = 0; j < X[i].size(); j++) {
            weights[j] -= learningRate * error * X[i][j];
          }
          bias -= learningRate * error;
        }
      }
    }
  }
};

double rbf_kernel(const Vector &x, const Vector &z, double sigma) {
  double norm_sq = 0.0;
  for (size_t i = 0; i < x.size(); i++) {
    norm_sq += (x[i] - z[i]) * (x[i] - z[i]);
  }
  return exp(-norm_sq / (2 * sigma * sigma));
}

double linear_kernel(const Vector &x, const Vector &z) {
  double result = 0.0;
  for (size_t i = 0; i < x.size(); i++)
    result += x[i] * z[i];
  return result;
}

double polynomial_kernel(const Vector &x, const Vector &z, int degree,
                         double coef0) {
  double dot = 0.0;
  for (size_t i = 0; i < x.size(); i++)
    dot += x[i] * z[i];
  double normalized_dot =
      dot / static_cast<double>(std::max<size_t>(x.size(), 1));
  return pow(normalized_dot + coef0, degree);
}

enum class KernelType { RBF, Linear, Polynomial };

class KernelSVM : public SVM {
private:
  Matrix support_vectors;
  Vector alphas;
  double bias = 0.0;
  double sigma;
  double epsilon;
  KernelType kernel_type;
  int poly_degree;
  double poly_coef0;

  double computeKernel(const Vector &x, const Vector &z) const {
    double value = 0.0;
    switch (kernel_type) {
    case KernelType::Linear:
      value = linear_kernel(x, z);
      break;
    case KernelType::Polynomial:
      value = polynomial_kernel(x, z, poly_degree, poly_coef0);
      break;
    case KernelType::RBF:
    default:
      value = rbf_kernel(x, z, sigma);
      break;
    }
    if (!std::isfinite(value))
      return 0.0;
    return std::clamp(value, -1e6, 1e6);
  }

public:
  KernelSVM(size_t n_features, double learningRate, double sigma,
            double epsilon = 0.1, int maxIterations = 1000,
            KernelType kernel = KernelType::RBF, int degree = 3,
            double coef0 = 1.0)
      : SVM(n_features, learningRate, maxIterations), sigma(sigma),
        epsilon(epsilon), kernel_type(kernel), poly_degree(degree),
        poly_coef0(coef0) {}

  double predict(const Vector &x) const override {
    double result = bias;
    for (size_t i = 0; i < support_vectors.size(); i++) {
      result += alphas[i] * computeKernel(x, support_vectors[i]);
    }
    return result;
  }

  void fit(const Matrix &X, const Vector &y) override {
    support_vectors = X;
    alphas.assign(X.size(), 0.0);
    bias = 0.0;

    if (X.empty())
      return;

    for (int iter = 0; iter < maxIterations; iter++) {
      bool any_update = false;
      for (size_t i = 0; i < X.size(); i++) {
        double prediction = predict(X[i]);
        double error = y[i] - prediction;

        if (std::abs(error) <= epsilon)
          continue;

        // Bound each dual update to keep kernels stable across feature scales.
        double delta = learningRate * std::clamp(error, -1.0, 1.0);
        alphas[i] += delta;
        bias += delta;
        any_update = true;
      }

      if (!any_update)
        break;
    }
  }
};
