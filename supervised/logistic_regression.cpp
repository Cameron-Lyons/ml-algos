#include "../matrix.h"
#include <cmath>

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
