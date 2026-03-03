#include "preprocessing.h"
#include <cmath>
#include <ranges>

class StandardScaler {
  Vector mean_, std_;

public:
  void fit(const Matrix &X) {
    size_t n = X.size();
    size_t d = X[0].size();
    mean_.assign(d, 0.0);
    std_.assign(d, 0.0);

    for (size_t j = 0; j < d; j++) {
      for (size_t i = 0; i < n; i++) {
        mean_[j] += X[i][j];
      }
      mean_[j] /= static_cast<double>(n);
    }

    for (size_t j = 0; j < d; j++) {
      for (size_t i = 0; i < n; i++) {
        double diff = X[i][j] - mean_[j];
        std_[j] += diff * diff;
      }
      std_[j] = std::sqrt(std_[j] / static_cast<double>(n));
      if (std_[j] == 0.0) {
        std_[j] = 1.0;
      }
    }
  }

  Matrix transform(const Matrix &X) const {
    Matrix result = X;
    for (auto &row : result) {
      for (auto [value, mean, stddev] : std::views::zip(row, mean_, std_)) {
        value = (value - mean) / stddev;
      }
    }
    return result;
  }

  Matrix fit_transform(const Matrix &X) {
    fit(X);
    return transform(X);
  }
};

ScaledData scaleData(const Matrix &X_train, const Matrix &X_test) {
  StandardScaler scaler;
  scaler.fit(X_train);
  return {.train = scaler.transform(X_train), .test = scaler.transform(X_test)};
}
