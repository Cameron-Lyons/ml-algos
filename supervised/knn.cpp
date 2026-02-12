#include "../matrix.h"
#include <algorithm>
#include <unordered_map>

class KNNClassifier {
private:
  int k;
  Matrix X_train;
  Vector y_train;

public:
  KNNClassifier(int k) : k(k) {}

  int getK() const { return k; }
  const Matrix &getXTrain() const { return X_train; }
  const Vector &getYTrain() const { return y_train; }
  void setTrainingData(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;
  }

  void fit(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;
  }

  double predict(const Vector &x) const {
    std::vector<std::pair<double, double>> distances;
    distances.reserve(X_train.size());
    for (size_t i = 0; i < X_train.size(); i++) {
      distances.emplace_back(squaredEuclideanDistance(x, X_train[i]),
                             y_train[i]);
    }

    std::nth_element(
        distances.begin(), distances.begin() + k, distances.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    std::unordered_map<int, int> votes;
    for (size_t i = 0; i < static_cast<size_t>(k); i++) {
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
    predictions.reserve(X.size());
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

public:
  KNNRegressor(int k) : k(k) {}

  int getK() const { return k; }
  const Matrix &getXTrain() const { return X_train; }
  const Vector &getYTrain() const { return y_train; }
  void setTrainingData(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;
  }

  void fit(const Matrix &X, const Vector &y) {
    X_train = X;
    y_train = y;
  }

  double predict(const Vector &x) const {
    std::vector<std::pair<double, double>> distances;
    distances.reserve(X_train.size());
    for (size_t i = 0; i < X_train.size(); i++) {
      distances.emplace_back(squaredEuclideanDistance(x, X_train[i]),
                             y_train[i]);
    }

    std::nth_element(
        distances.begin(), distances.begin() + k, distances.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    double sum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(k); i++) {
      sum += distances[i].second;
    }
    return sum / static_cast<double>(k);
  }

  Vector predict(const Matrix &X) const {
    Vector predictions;
    predictions.reserve(X.size());
    for (const auto &x : X) {
      predictions.push_back(predict(x));
    }
    return predictions;
  }
};
