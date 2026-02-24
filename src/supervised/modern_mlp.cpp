#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

enum class Activation { ReLU, Sigmoid, Tanh };

namespace modern_mlp_detail {

double clipFinite(double value, double limit = 1e6) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::clamp(value, -limit, limit);
}

double activate(double x, Activation act) {
  switch (act) {
  case Activation::ReLU:
    return x > 0.0 ? x : 0.0;
  case Activation::Sigmoid:
    x = std::clamp(x, -60.0, 60.0);
    return 1.0 / (1.0 + std::exp(-x));
  case Activation::Tanh:
    return std::tanh(x);
  }
  return x;
}

double activateDerivative(double output, Activation act) {
  switch (act) {
  case Activation::ReLU:
    return output > 0.0 ? 1.0 : 0.0;
  case Activation::Sigmoid:
    return output * (1.0 - output);
  case Activation::Tanh:
    return 1.0 - (output * output);
  }
  return 1.0;
}

} // namespace modern_mlp_detail

class ModernMLP {
private:
  std::vector<size_t> layerSizes_;
  Activation activation_;
  double learningRate_;
  int maxEpochs_;
  double l2Lambda_;
  size_t batchSize_;
  std::vector<Matrix> weights_;
  std::vector<Vector> biases_;
  size_t inputSize_ = 0;

  void initWeights(std::mt19937 &rng) {
    weights_.clear();
    biases_.clear();

    std::vector<size_t> sizes;
    sizes.push_back(inputSize_);
    for (size_t s : layerSizes_) {
      sizes.push_back(s);
    }
    sizes.push_back(1);

    for (size_t l = 0; l + 1 < sizes.size(); l++) {
      size_t fanIn = sizes[l];
      size_t fanOut = sizes[l + 1];

      double stddev;
      if (activation_ == Activation::ReLU) {
        stddev = std::sqrt(2.0 / static_cast<double>(fanIn));
      } else {
        stddev = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
      }

      auto dist = std::normal_distribution<double>(0.0, stddev);
      Matrix w(fanIn, Vector(fanOut));
      for (size_t i = 0; i < fanIn; i++) {
        for (size_t j = 0; j < fanOut; j++) {
          w[i][j] = dist(rng);
        }
      }
      weights_.push_back(std::move(w));
      biases_.emplace_back(fanOut, 0.0);
    }
  }

  Vector forward(const Vector &input, std::vector<Vector> &layerOutputs) const {
    layerOutputs.clear();
    layerOutputs.push_back(input);

    Vector current = input;
    for (size_t l = 0; l < weights_.size(); l++) {
      size_t outSize = weights_[l][0].size();
      Vector next(outSize, 0.0);
      for (size_t j = 0; j < outSize; j++) {
        double sum = biases_[l][j];
        for (size_t i = 0; i < current.size(); i++) {
          sum += current[i] * weights_[l][i][j];
        }
        if (l + 1 < weights_.size()) {
          next[j] = modern_mlp_detail::activate(sum, activation_);
        } else {
          next[j] = sum;
        }
      }
      layerOutputs.push_back(next);
      current = next;
    }
    return current;
  }

public:
  ModernMLP(std::vector<size_t> layerSizes,
            Activation activation = Activation::ReLU,
            double learningRate = 0.01, int maxEpochs = 500,
            double l2Lambda = 0.0, size_t batchSize = 32)
      : layerSizes_(std::move(layerSizes)), activation_(activation),
        learningRate_(learningRate), maxEpochs_(maxEpochs), l2Lambda_(l2Lambda),
        batchSize_(batchSize) {}

  void fit(const Matrix &X, const Vector &y) {
    if (X.empty() || y.empty()) {
      return;
    }

    inputSize_ = X[0].size();
    std::mt19937 rng(42);
    initWeights(rng);

    size_t n = X.size();
    auto indices = std::vector<size_t>(n);
    for (size_t i = 0; i < n; i++) {
      indices[i] = i;
    }

    for (int epoch = 0; epoch < maxEpochs_; epoch++) {
      double epochLearningRate = learningRate_ / (1.0 + 0.01 * epoch);
      std::ranges::shuffle(indices, rng);

      for (size_t batchStart = 0; batchStart < n; batchStart += batchSize_) {
        size_t batchEnd = std::min(batchStart + batchSize_, n);
        size_t bSize = batchEnd - batchStart;
        double batchScale = 1.0 / static_cast<double>(bSize);

        std::vector<Matrix> wGrad(weights_.size());
        std::vector<Vector> bGrad(biases_.size());
        for (size_t l = 0; l < weights_.size(); l++) {
          wGrad[l] =
              Matrix(weights_[l].size(), Vector(weights_[l][0].size(), 0.0));
          bGrad[l] = Vector(biases_[l].size(), 0.0);
        }

        for (size_t b = batchStart; b < batchEnd; b++) {
          size_t idx = indices[b];
          std::vector<Vector> layerOutputs;
          Vector pred = forward(X[idx], layerOutputs);

          double error = modern_mlp_detail::clipFinite(pred[0] - y[idx], 100.0);
          if (!std::isfinite(error)) {
            continue;
          }

          std::vector<Vector> deltas(weights_.size());
          deltas[weights_.size() - 1] = {error};

          for (int l = static_cast<int>(weights_.size()) - 2; l >= 0; l--) {
            size_t lu = static_cast<size_t>(l);
            size_t nextSize = layerOutputs[lu + 1].size();
            Vector delta(nextSize, 0.0);
            for (size_t i = 0; i < nextSize; i++) {
              double sum = 0.0;
              for (size_t j = 0; j < deltas[lu + 1].size(); j++) {
                sum += deltas[lu + 1][j] * weights_[lu + 1][i][j];
              }
              double grad = modern_mlp_detail::activateDerivative(
                  layerOutputs[lu + 1][i], activation_);
              delta[i] = modern_mlp_detail::clipFinite(sum * grad, 10.0);
            }
            deltas[lu] = delta;
          }

          for (size_t l = 0; l < weights_.size(); l++) {
            for (size_t i = 0; i < weights_[l].size(); i++) {
              for (size_t j = 0; j < weights_[l][0].size(); j++) {
                double grad = layerOutputs[l][i] * deltas[l][j];
                wGrad[l][i][j] += modern_mlp_detail::clipFinite(grad, 10.0);
              }
            }
            for (size_t j = 0; j < biases_[l].size(); j++) {
              bGrad[l][j] += modern_mlp_detail::clipFinite(deltas[l][j], 10.0);
            }
          }
        }

        for (size_t l = 0; l < weights_.size(); l++) {
          for (size_t i = 0; i < weights_[l].size(); i++) {
            for (size_t j = 0; j < weights_[l][0].size(); j++) {
              double update =
                  epochLearningRate *
                  (wGrad[l][i][j] * batchScale + l2Lambda_ * weights_[l][i][j]);
              update = modern_mlp_detail::clipFinite(update, 1.0);
              weights_[l][i][j] = modern_mlp_detail::clipFinite(
                  weights_[l][i][j] - update, 1e6);
            }
          }
          for (size_t j = 0; j < biases_[l].size(); j++) {
            double update =
                epochLearningRate *
                modern_mlp_detail::clipFinite(bGrad[l][j] * batchScale, 1.0);
            biases_[l][j] =
                modern_mlp_detail::clipFinite(biases_[l][j] - update, 1e6);
          }
        }
      }
    }
  }

  double predict(const Vector &x) const {
    std::vector<Vector> layerOutputs;
    Vector out = forward(x, layerOutputs);
    return modern_mlp_detail::clipFinite(out[0], 1e6);
  }
};
