#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

class KNNClassifier {
private:
  int k;
  Matrix X_train;
  Vector y_train;

  double distance(const Vector &a, const Vector &b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
      double diff = a[i] - b[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

public:
  KNNClassifier(int k) : k(k) {}

  void fit(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;
  }

  double predict(const Vector &x) const {
    std::vector<std::pair<double, double>> distances;
    for (size_t i = 0; i < X_train.size(); i++) {
      distances.emplace_back(distance(x, X_train[i]), y_train[i]);
    }

    std::partial_sort(
        distances.begin(), distances.begin() + k, distances.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    std::unordered_map<int, int> votes;
    for (int i = 0; i < k; i++) {
      votes[static_cast<int>(distances[i].second)]++;
    }

    int best_class = -1;
    int best_count = 0;
    for (const auto &[cls, count] : votes) {
      if (count > best_count) {
        best_class = cls;
        best_count = count;
      }
    }
    return static_cast<double>(best_class);
  }

  Vector predict(const Matrix &X) const {
    Vector predictions;
    for (const auto &x : X) {
      predictions.push_back(predict(x));
    }
    return predictions;
  }
};

class KNNRegressor {
private:
  int k;
  Matrix X_train;
  Vector y_train;

  double distance(const Vector &a, const Vector &b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
      double diff = a[i] - b[i];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

public:
  KNNRegressor(int k) : k(k) {}

  void fit(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;
  }

  double predict(const Vector &x) const {
    std::vector<std::pair<double, double>> distances;
    for (size_t i = 0; i < X_train.size(); i++) {
      distances.emplace_back(distance(x, X_train[i]), y_train[i]);
    }

    std::partial_sort(
        distances.begin(), distances.begin() + k, distances.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    double sum = 0.0;
    for (int i = 0; i < k; i++) {
      sum += distances[i].second;
    }
    return sum / k;
  }

  Vector predict(const Matrix &X) const {
    Vector predictions;
    for (const auto &x : X) {
      predictions.push_back(predict(x));
    }
    return predictions;
  }
};
