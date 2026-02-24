#include "matrix.h"
#include <cmath>

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
      for (size_t j = 0; j < row.size(); j++) {
        row[j] = (row[j] - mean_[j]) / std_[j];
      }
    }
    return result;
  }

  Matrix fit_transform(const Matrix &X) {
    fit(X);
    return transform(X);
  }
};

class MinMaxScaler {
  Vector min_, range_;

public:
  void fit(const Matrix &X) {
    size_t d = X[0].size();
    min_.assign(d, std::numeric_limits<double>::max());
    Vector max(d, std::numeric_limits<double>::lowest());

    for (const auto &row : X) {
      for (size_t j = 0; j < d; j++) {
        if (row[j] < min_[j]) {
          min_[j] = row[j];
        }
        if (row[j] > max[j]) {
          max[j] = row[j];
        }
      }
    }

    range_.resize(d);
    for (size_t j = 0; j < d; j++) {
      range_[j] = max[j] - min_[j];
      if (range_[j] == 0.0) {
        range_[j] = 1.0;
      }
    }
  }

  Matrix transform(const Matrix &X) const {
    Matrix result = X;
    for (auto &row : result) {
      for (size_t j = 0; j < row.size(); j++) {
        row[j] = (row[j] - min_[j]) / range_[j];
      }
    }
    return result;
  }

  Matrix fit_transform(const Matrix &X) {
    fit(X);
    return transform(X);
  }
};

std::pair<Matrix, Matrix> scaleData(const Matrix &X_train,
                                    const Matrix &X_test) {
  StandardScaler scaler;
  scaler.fit(X_train);
  return {scaler.transform(X_train), scaler.transform(X_test)};
}
