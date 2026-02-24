#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <ranges>

class LogisticRegression {
private:
  Vector weights;
  double bias = 0.0;
  double learningRate;
  int maxIterations;

  double sigmoid(double z) const { return 1.0 / (1.0 + std::exp(-z)); }

public:
  LogisticRegression(double learningRate = 0.01, int maxIterations = 1000)
      : learningRate(learningRate), maxIterations(maxIterations) {}

  void fit(const Matrix &X, const Vector &y) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    weights = Vector(n_features, 0.0);

    for (int iter = 0; iter < maxIterations; iter++) {
      Vector dw(n_features, 0.0);
      double db = 0.0;

      for (size_t i = 0; i < n_samples; i++) {
        double z = bias;
        for (size_t j = 0; j < n_features; j++) {
          z += weights[j] * X[i][j];
        }
        double pred = sigmoid(z);
        double error = pred - y[i];

        for (size_t j = 0; j < n_features; j++) {
          dw[j] += error * X[i][j];
        }
        db += error;
      }

      auto n = static_cast<double>(n_samples);
      for (size_t j = 0; j < n_features; j++) {
        weights[j] -= learningRate * dw[j] / n;
      }
      bias -= learningRate * db / n;
    }
  }

  double predictProbability(const Vector &x) const {
    double z = bias;
    for (size_t j = 0; j < x.size(); j++) {
      z += weights[j] * x[j];
    }
    return sigmoid(z);
  }

  double predict(const Vector &x, double threshold = 0.5) const {
    return predictProbability(x) >= threshold ? 1.0 : 0.0;
  }

  Vector predict(const Matrix &X, double threshold = 0.5) const {
    Vector predictions;
    for (const auto &x : X) {
      predictions.push_back(predict(x, threshold));
    }
    return predictions;
  }

  Vector getWeights() const { return weights; }
  double getBias() const { return bias; }
  void setWeights(const Vector &w) { weights = w; }
  void setBias(double b) { bias = b; }
};

class SoftmaxRegression {
private:
  Matrix weights;
  Vector biases;
  double learningRate;
  int maxIterations;
  size_t nClasses = 0;

  Vector softmax(const Vector &z) const {
    double maxZ = std::ranges::max(z);
    Vector exp_z(z.size());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); i++) {
      exp_z[i] = std::exp(z[i] - maxZ);
      sum += exp_z[i];
    }
    for (double &v : exp_z) {
      v /= sum;
    }
    return exp_z;
  }

public:
  SoftmaxRegression(double learningRate = 0.01, int maxIterations = 1000)
      : learningRate(learningRate), maxIterations(maxIterations) {}

  void fit(const Matrix &X, const Vector &y) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    nClasses = 0;
    for (double v : y) {
      nClasses = std::max(nClasses, static_cast<size_t>(v) + 1);
    }

    weights = Matrix(n_features, Vector(nClasses, 0.0));
    biases = Vector(nClasses, 0.0);

    for (int iter = 0; iter < maxIterations; iter++) {
      Matrix dw(n_features, Vector(nClasses, 0.0));
      Vector db(nClasses, 0.0);

      for (size_t i = 0; i < n_samples; i++) {
        Vector z(nClasses, 0.0);
        for (size_t c = 0; c < nClasses; c++) {
          z[c] = biases[c];
          for (size_t j = 0; j < n_features; j++) {
            z[c] += weights[j][c] * X[i][j];
          }
        }

        Vector probs = softmax(z);
        size_t label = static_cast<size_t>(y[i]);

        for (size_t c = 0; c < nClasses; c++) {
          double grad = probs[c] - (c == label ? 1.0 : 0.0);
          for (size_t j = 0; j < n_features; j++) {
            dw[j][c] += grad * X[i][j];
          }
          db[c] += grad;
        }
      }

      auto n = static_cast<double>(n_samples);
      for (size_t j = 0; j < n_features; j++) {
        for (size_t c = 0; c < nClasses; c++) {
          weights[j][c] -= learningRate * dw[j][c] / n;
        }
      }
      for (size_t c = 0; c < nClasses; c++) {
        biases[c] -= learningRate * db[c] / n;
      }
    }
  }

  double predict(const Vector &x) const {
    Vector z(nClasses, 0.0);
    for (size_t c = 0; c < nClasses; c++) {
      z[c] = biases[c];
      for (size_t j = 0; j < x.size(); j++) {
        z[c] += weights[j][c] * x[j];
      }
    }
    Vector probs = softmax(z);
    return static_cast<double>(std::ranges::max_element(probs) - probs.begin());
  }
};
