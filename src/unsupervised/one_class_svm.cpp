#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

class OneClassSVM {
private:
  Matrix supportVectors_;
  double gamma_;
  double nu_;
  double rho_ = 0.0;

  double kernel(const Vector &x, const Vector &z) const {
    double sq = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
      const double d = x[i] - z[i];
      sq += d * d;
    }
    return std::exp(-gamma_ * sq);
  }

public:
  OneClassSVM(double gamma = 0.5, double nu = 0.1)
      : supportVectors_(), gamma_(std::max(1e-6, gamma)),
        nu_(std::clamp(nu, 1e-3, 0.5)), rho_(0.0) {}

  void fit(const Matrix &X) {
    supportVectors_ = X;
    if (supportVectors_.empty()) {
      rho_ = 0.0;
      return;
    }

    Vector scores(supportVectors_.size(), 0.0);
    for (size_t i = 0; i < supportVectors_.size(); i++) {
      double sum = 0.0;
      for (const auto &sv : supportVectors_) {
        sum += kernel(supportVectors_[i], sv);
      }
      scores[i] = sum / static_cast<double>(supportVectors_.size());
    }

    Vector sortedScores = scores;
    std::sort(sortedScores.begin(), sortedScores.end());

    const size_t rank = static_cast<size_t>(
        std::floor(nu_ * static_cast<double>(sortedScores.size() - 1)));
    rho_ = sortedScores[rank];
  }

  double decisionFunction(const Vector &x) const {
    if (supportVectors_.empty()) {
      return 0.0;
    }

    double sum = 0.0;
    for (const auto &sv : supportVectors_) {
      sum += kernel(x, sv);
    }
    const double meanScore = sum / static_cast<double>(supportVectors_.size());
    return meanScore - rho_;
  }

  double scoreSample(const Vector &x) const { return decisionFunction(x); }

  double predict(const Vector &x) const {
    return decisionFunction(x) >= 0.0 ? 1.0 : -1.0;
  }

  Vector predict(const Matrix &X) const {
    Vector labels;
    labels.reserve(X.size());
    for (const auto &x : X) {
      labels.push_back(predict(x));
    }
    return labels;
  }
};
